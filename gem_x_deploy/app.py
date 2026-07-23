from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr
import spaces

PROJECT_ROOT = Path(__file__).resolve().parent
MAX_SECONDS = 2


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _prepare_video(source: str, destination: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            source,
            "-t",
            str(MAX_SECONDS),
            "-vf",
            "fps=30,scale='min(720,iw)':-2:force_original_aspect_ratio=decrease",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    )


@spaces.GPU(duration=120)
def capture_motion(
    video_path: str | None,
    camera_mode: str,
    use_ddim: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not video_path:
        raise gr.Error("请先上传一段视频。")

    progress(0.03, desc="准备视频")
    job_dir = Path(tempfile.mkdtemp(prefix="gemx_", dir="/tmp"))
    input_path = job_dir / "input.mp4"
    output_root = job_dir / "outputs"

    try:
        _prepare_video(video_path, input_path)
        progress(0.1, desc="运行 GEM-X")

        command = [
            sys.executable,
            str(PROJECT_ROOT / "space_runner.py"),
            "--video",
            str(input_path),
            "--output-root",
            str(output_root),
        ]
        if camera_mode == "固定相机":
            command.append("--static-camera")
        if use_ddim:
            command.append("--ddim")

        result = _run(command)
        marker = "GEMX_RESULT="
        result_line = next(
            (line[len(marker) :] for line in result.stdout.splitlines() if line.startswith(marker)),
            None,
        )
        if result_line is None:
            raise RuntimeError("推理已结束，但没有返回结果清单。\n" + result.stdout[-4000:])
        files = json.loads(result_line)

        progress(0.95, desc="整理下载文件")
        file_keys = ("preview", "bvh77", "bvh78", "raw", "archive")
        missing = [name for name in file_keys if not Path(files[name]).exists()]
        if missing:
            raise RuntimeError(f"缺少输出文件：{', '.join(missing)}")

        summary = (
            f"完成：处理 {files['frames']} 帧，{files['fps']:.2f} FPS。"
            " Blender 推荐导入 soma77_blender.bvh；"
            " soma78_virtual_root.bvh 保留 GEM-X 虚拟 Root。"
        )
        progress(1.0, desc="完成")
        return (
            files["preview"],
            files["bvh77"],
            files["bvh78"],
            files["raw"],
            files["archive"],
            summary,
        )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        details = (exc.stdout or str(exc))[-6000:]
        raise gr.Error("GEM-X 运行失败：\n" + details) from exc
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="GEM-X Motion Capture") as demo:
    gr.Markdown(
        """
        # GEM-X 视频动作捕捉

        上传单人短视频，生成 NVIDIA SOMA 全身动作和 Blender BVH。
        为适配 ZeroGPU，视频会自动截取前 2 秒、转为 30 FPS。
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="输入视频", sources=["upload"])
            camera = gr.Radio(
                ["固定相机", "移动相机"],
                value="固定相机",
                label="相机类型",
            )
            ddim = gr.Checkbox(
                value=False,
                label="DDIM 50 步（更慢，兼容模式下姿态通常更好）",
            )
            run_button = gr.Button("生成动作", variant="primary")
        with gr.Column(scale=1):
            preview = gr.Video(label="77 点跟踪预览")
            status = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        bvh77 = gr.File(label="Blender BVH（77 关节）")
        bvh78 = gr.File(label="SOMA BVH（78 节点，含虚拟 Root）")
        raw = gr.File(label="GEM-X 原始结果 (.pt)")
        archive = gr.File(label="全部结果 (.zip)")

    gr.Markdown(
        """
        Blender：`文件 → 导入 → Motion Capture (.bvh)`。通常先用 77 关节版本，
        再通过 Blender 的 Retarget 工具或插件映射到你的角色骨架。

        建议画面中只有一个人，身体和双手尽量完整可见。首次运行需要下载模型，
        会明显慢于后续运行。
        """
    )

    run_button.click(
        capture_motion,
        inputs=[video, camera, ddim],
        outputs=[preview, bvh77, bvh78, raw, archive, status],
    )

demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch()
