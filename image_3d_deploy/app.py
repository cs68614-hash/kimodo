from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import spaces

# ZeroGPU patches torch when ``spaces`` is imported. Keep this import order.
import torch
import trimesh
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent
HUNYUAN_ROOT = ROOT / "third_party" / "Hunyuan3D-2"
if HUNYUAN_ROOT.exists():
    sys.path.insert(0, str(HUNYUAN_ROOT))

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from server import colorproc, meshproc, preprocess

MODEL_ID = "tencent/Hunyuan3D-2"
MODEL_SUBFOLDER = "hunyuan3d-dit-v2-0"
OUTPUT_ROOT = Path(tempfile.gettempdir()) / "image3d_outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _load_pipeline() -> Hunyuan3DDiTFlowMatchingPipeline:
    """Link the two required cached files, then load directly onto CUDA.

    Hunyuan3D's stock loader snapshots every file in the model subfolder, which
    includes several duplicate 4.9GB checkpoints. The Space only needs the
    config and fp16 safetensors checkpoint.
    """
    local_root = Path(tempfile.gettempdir()) / "hy3dgen_models"
    model_dir = local_root / MODEL_ID / MODEL_SUBFOLDER
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("config.yaml", "model.fp16.safetensors"):
        cached_path = Path(
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=f"{MODEL_SUBFOLDER}/{filename}",
            )
        )
        destination = model_dir / filename
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        destination.symlink_to(cached_path)

    os.environ["HY3DGEN_MODELS"] = str(local_root)
    print(f"Loading {MODEL_ID}/{MODEL_SUBFOLDER} on ZeroGPU...", flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        MODEL_ID,
        subfolder=MODEL_SUBFOLDER,
        device="cuda",
        dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    print("Hunyuan3D shape pipeline is ready.", flush=True)
    return pipeline


PIPELINE = None if os.getenv("IMAGE3D_SKIP_MODEL_LOAD") == "1" else _load_pipeline()


def _prepare_image(image: Image.Image, remove_bg: bool) -> tuple[Image.Image, bool]:
    image = image.convert("RGBA")
    bg_removed = False

    if remove_bg and not preprocess.has_removed_background(image):
        image, bg_removed = preprocess.remove_uniform_background(image)
        if not bg_removed:
            image, bg_removed = preprocess.remove_background(image)

    return preprocess.resize_to_square(image, size=1024), bg_removed


def _first_mesh(result: object) -> trimesh.Trimesh:
    while isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("模型没有返回可用的 3D 网格。")
        result = result[0]

    if isinstance(result, trimesh.Trimesh):
        return result

    vertices = getattr(result, "vertices", None)
    faces = getattr(result, "faces", None)
    if vertices is None or faces is None:
        vertices = getattr(result, "mesh_v", None)
        faces = getattr(result, "mesh_f", None)
    if vertices is None or faces is None:
        raise RuntimeError(f"无法识别模型输出类型：{type(result)!r}")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _export_mesh(
    mesh: trimesh.Trimesh,
    processed_image: Image.Image,
    add_vertex_color: bool,
) -> tuple[str, list[str]]:
    output_dir = OUTPUT_ROOT / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=False)

    if add_vertex_color:
        colors = colorproc.project_multiview_colors(mesh, processed_image)
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)

    paths: list[str] = []
    for suffix, file_type in (
        ("glb", "glb"),
        ("obj", "obj"),
        ("stl", "stl"),
        ("3mf", "3mf"),
    ):
        path = output_dir / f"image3d.{suffix}"
        payload = mesh.export(file_type=file_type)
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_bytes(payload)
        paths.append(str(path))

    return paths[0], paths


def _duration(
    _image: Image.Image,
    steps: int,
    _guidance: float,
    _octree: str,
    _seed: int,
    _height: float,
    _max_faces: str,
    _remove_bg: bool,
    _add_vertex_color: bool,
) -> int:
    return min(240, max(90, int(steps) * 5))


@spaces.GPU(duration=_duration, size="large")
def generate(
    image: Image.Image,
    steps: int,
    guidance: float,
    octree: str,
    seed: int,
    target_height_mm: float,
    max_faces: str,
    remove_bg: bool,
    add_vertex_color: bool,
):
    if image is None:
        raise gr.Error("请先上传一张主体清晰的图片。")
    if PIPELINE is None:
        raise gr.Error("模型未加载。")

    try:
        processed, bg_removed = _prepare_image(image, remove_bg)
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        result = PIPELINE(
            image=processed,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            octree_resolution=int(octree),
            generator=generator,
            output_type="trimesh",
        )

        mesh = _first_mesh(result)
        # Hunyuan3D uses Y-up; Blender and image-3d's post-processing use Z-up.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
        )
        mesh, stats = meshproc.process(
            mesh,
            target_height_mm=float(target_height_mm),
            max_faces=int(max_faces),
        )
        preview, files = _export_mesh(mesh, processed, add_vertex_color)
        info = stats.to_dict()
        info.update(
            {
                "model": MODEL_ID,
                "background_removed": bg_removed,
                "seed": int(seed),
                "steps": int(steps),
                "octree_resolution": int(octree),
                "note": "GLB/OBJ are recommended for Blender. STL/3MF use millimetre scale.",
            }
        )
        return preview, files, processed, info
    except gr.Error:
        raise
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise gr.Error("GPU 显存不足。请把 Octree 调成 256，或把最大面数调低后重试。") from exc
    except Exception as exc:
        raise gr.Error(f"生成失败：{exc}") from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


CSS = """
.legal-note {font-size: 0.86rem; opacity: 0.82;}
.gradio-container {max-width: 1240px !important;}
"""

with gr.Blocks(title="Image → 3D (ZeroGPU)", css=CSS) as demo:
    gr.Markdown(
        """
        # Image → 3D
        上传单张图片，使用 Hunyuan3D-2 生成 3D 形状，并导出 Blender 可用的
        **GLB / OBJ** 与 3D 打印常用的 **STL / 3MF**。

        当前是 ZeroGPU 稳定版：只启用 Shape，不启用高显存的 Paint 纹理模型。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                image_mode="RGBA",
                label="输入图片",
                sources=["upload", "clipboard"],
            )
            with gr.Accordion("生成参数", open=False):
                steps = gr.Slider(10, 50, value=30, step=1, label="推理步数")
                guidance = gr.Slider(1.0, 10.0, value=5.5, step=0.1, label="引导强度")
                octree = gr.Dropdown(
                    choices=["256", "384"],
                    value="384",
                    label="Octree 分辨率",
                )
                seed = gr.Number(value=1234, precision=0, label="随机种子")
                target_height = gr.Number(value=100, label="导出高度（毫米）")
                max_faces = gr.Dropdown(
                    choices=["50000", "100000", "200000"],
                    value="100000",
                    label="最大面数",
                )
                remove_bg = gr.Checkbox(value=True, label="自动移除背景")
                add_color = gr.Checkbox(
                    value=True,
                    label="将输入图颜色投影到 GLB/OBJ 顶点（轻量预览色）",
                )

            generate_button = gr.Button("生成 3D", variant="primary")

        with gr.Column(scale=1):
            model_preview = gr.Model3D(label="3D 预览", clear_color=[0.05, 0.05, 0.06, 1.0])
            downloads = gr.Files(label="下载 GLB / OBJ / STL / 3MF")
            processed_image = gr.Image(label="实际送入模型的图片", interactive=False)
            stats = gr.JSON(label="网格信息")

    gr.Markdown(
        """
        <div class="legal-note">
        本私有演示由 Space 所有者自行提供，并非腾讯赞助、认可或关联的官方服务。
        基于 animede/image-3d 与 Tencent Hunyuan3D-2；使用前请阅读仓库内
        LICENSE、NOTICE 与 LICENSES/ 目录。Hunyuan3D-2 许可不适用于欧盟、英国和韩国，
        且包含地域、用途及规模限制。
        </div>
        """
    )

    generate_button.click(
        fn=generate,
        inputs=[
            input_image,
            steps,
            guidance,
            octree,
            seed,
            target_height,
            max_faces,
            remove_bg,
            add_color,
        ],
        outputs=[model_preview, downloads, processed_image, stats],
        concurrency_limit=1,
        api_name="generate_3d",
    )

demo.queue(max_size=10).launch(mcp_server=True)
