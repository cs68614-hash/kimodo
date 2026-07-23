from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent
SOMA_SOURCE = PROJECT_ROOT / "third_party" / "soma"
for source_dir in (PROJECT_ROOT, SOMA_SOURCE):
    source = str(source_dir)
    if source not in sys.path:
        sys.path.insert(0, source)


def ensure_soma_assets() -> Path:
    target = Path("inputs/soma_assets")
    required = target / "SOMA_neutral.npz"
    if not required.exists():
        snapshot_download(
            repo_id="nvidia/soma-x",
            repo_type="model",
            local_dir=target,
        )
    return target


def run(args: argparse.Namespace) -> dict:
    ensure_soma_assets()

    from gem.utils.kp2d_utils import render_2d_keypoints
    from scripts.demo import demo_soma_onnx as pipeline
    from space_bvh import export_both_bvhs

    demo_args = SimpleNamespace(
        video=str(Path(args.video).resolve()),
        output_root=str(Path(args.output_root).resolve()),
        static_cam=args.static_camera,
        verbose=False,
        ckpt=None,
        exp="gem_soma_regression",
    )
    cfg = pipeline._build_cfg(demo_args)
    no_imgfeat = True

    pipeline.run_preprocess_fast(cfg, force_pytorch=args.ddim, no_imgfeat=no_imgfeat)
    data = pipeline.load_data_dict(cfg, no_imgfeat=no_imgfeat)
    pred = pipeline.run_inference_fast(
        cfg,
        data,
        force_pytorch=args.ddim,
        no_imgfeat=no_imgfeat,
        use_ddim=args.ddim,
    )
    raw_path = Path(cfg.paths.hpe_results)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, raw_path)

    cap = cv2.VideoCapture(str(cfg.video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    output_dir = Path(cfg.output_dir)
    preview = output_dir / "gemx_77_keypoints.mp4"
    render_2d_keypoints(
        video_path=cfg.video_path,
        vitpose_path=cfg.paths.vitpose,
        bbx_path=cfg.paths.bbx,
        output_path=str(preview),
        fps=fps,
    )

    body_params = pred.get("body_params_global") or pred.get("pred_body_params_global")
    if body_params is None:
        raise RuntimeError("GEM-X output has no world-space SOMA body parameters.")

    bvh77 = output_dir / "soma77_blender.bvh"
    bvh78 = output_dir / "soma78_virtual_root.bvh"
    export_both_bvhs(body_params, fps, bvh77, bvh78)

    archive_base = Path(args.output_root) / "gemx_motion_results"
    archive_path = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=output_dir,
        )
    )
    return {
        "preview": str(preview),
        "bvh77": str(bvh77),
        "bvh78": str(bvh78),
        "raw": str(raw_path),
        "archive": str(archive_path),
        "fps": fps,
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--static-camera", action="store_true")
    parser.add_argument("--ddim", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print("GEMX_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
