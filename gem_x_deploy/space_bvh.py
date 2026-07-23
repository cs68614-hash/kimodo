from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _read_reference_hierarchy(path: Path) -> tuple[list[str], list[int]]:
    names: list[str] = []
    parents: list[int] = []
    stack: list[int] = []
    pending: int | None = None
    in_end_site = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = re.match(r"^(ROOT|JOINT)\s+(.+)$", line)
        if match:
            parent = stack[-1] if stack else -1
            names.append(match.group(2).strip())
            parents.append(parent)
            pending = len(names) - 1
            in_end_site = False
            continue
        if line.startswith("End Site"):
            pending = None
            in_end_site = True
            continue
        if line == "{":
            if pending is not None:
                stack.append(pending)
                pending = None
            elif in_end_site:
                stack.append(-2)
                in_end_site = False
            continue
        if line == "}" and stack:
            stack.pop()

    if len(names) != 78:
        raise RuntimeError(f"Expected 78 reference nodes, found {len(names)}.")
    return names, parents


def _skeleton(body_params: dict):
    from gem.utils.soma_utils.soma_layer import SomaLayer

    identity = body_params.get("identity_coeffs")
    scales = body_params.get("scale_params")
    if identity is None:
        identity = torch.zeros(1, 64)
    if scales is None:
        scales = torch.zeros(1, 69)

    identity = torch.as_tensor(_numpy(identity), dtype=torch.float32).reshape(-1, 64)[:1]
    scales = torch.as_tensor(_numpy(scales), dtype=torch.float32).reshape(-1, 69)[:1]
    soma = SomaLayer(
        data_root="inputs/soma_assets",
        low_lod=True,
        device="cpu",
        identity_model_type="mhr",
        mode="warp",
    )
    with torch.no_grad():
        joints = soma.get_skeleton(identity, scales)[0].cpu().numpy()
    return joints


def _motion_data(body_params: dict):
    global_orient = _numpy(body_params["global_orient"]).reshape(-1, 3)
    body_pose = _numpy(body_params["body_pose"]).reshape(len(global_orient), 76, 3)
    transl = _numpy(body_params["transl"]).reshape(-1, 3)
    all_rotvecs = np.concatenate([global_orient[:, None], body_pose], axis=1)
    return all_rotvecs, transl


def _write_bvh(
    path: Path,
    names: list[str],
    parents: list[int],
    offsets: np.ndarray,
    euler: np.ndarray,
    translation: np.ndarray,
    fps: float,
) -> None:
    children = [[] for _ in names]
    root = parents.index(-1)
    for joint, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(joint)

    def write_joint(handle, joint: int, depth: int):
        indent = "  " * depth
        handle.write(f"{indent}{'ROOT' if joint == root else 'JOINT'} {names[joint]}\n")
        handle.write(f"{indent}{{\n")
        x, y, z = offsets[joint] * 100.0
        handle.write(f"{indent}  OFFSET {x:.6f} {y:.6f} {z:.6f}\n")
        if joint == root:
            handle.write(
                f"{indent}  CHANNELS 6 Xposition Yposition Zposition "
                "Zrotation Yrotation Xrotation\n"
            )
        else:
            handle.write(f"{indent}  CHANNELS 3 Zrotation Yrotation Xrotation\n")
        if children[joint]:
            for child in children[joint]:
                write_joint(handle, child, depth + 1)
        else:
            handle.write(f"{indent}  End Site\n{indent}  {{\n")
            handle.write(f"{indent}    OFFSET 0.000000 0.000000 0.000000\n")
            handle.write(f"{indent}  }}\n")
        handle.write(f"{indent}}}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("HIERARCHY\n")
        write_joint(handle, root, 0)
        handle.write("MOTION\n")
        handle.write(f"Frames: {len(euler)}\n")
        handle.write(f"Frame Time: {1.0 / max(float(fps), 1.0):.8f}\n")
        for frame in range(len(euler)):
            values: list[float] = []
            for joint in range(len(names)):
                if joint == root:
                    values.extend((translation[frame] * 100.0).tolist())
                values.extend(euler[frame, joint].tolist())
            handle.write(" ".join(f"{value:.6f}" for value in values) + "\n")


def export_both_bvhs(
    body_params: dict,
    fps: float,
    blender_path: Path,
    virtual_root_path: Path,
) -> None:
    reference = Path("assets/soma_zero_frame0.bvh")
    names78, parents78 = _read_reference_hierarchy(reference)
    positions77 = _skeleton(body_params)
    positions78 = np.concatenate([np.zeros((1, 3), np.float32), positions77], axis=0)

    rig = np.load("inputs/soma_assets/SOMA_neutral.npz", allow_pickle=False)
    orient = Rotation.from_matrix(rig["t_pose_world"][..., :3, :3])
    rotvecs, translation = _motion_data(body_params)
    frames = len(rotvecs)

    offsets78 = np.zeros((78, 3), np.float32)
    local_quat78 = np.zeros((frames, 78, 4), np.float64)
    local_quat78[:, 0, 3] = 1.0
    for joint in range(1, 78):
        parent = parents78[joint]
        world_offset = positions78[joint] - positions78[parent]
        offsets78[joint] = orient[parent].inv().apply(world_offset)
        body_rotation = Rotation.from_rotvec(rotvecs[:, joint - 1])
        local_quat78[:, joint] = (
            orient[parent].inv() * body_rotation * orient[joint]
        ).as_quat()

    # Translation represents the moving pelvis, so keep Hips offset at zero.
    offsets78[1] = 0.0
    euler78 = Rotation.from_quat(local_quat78.reshape(-1, 4)).as_euler(
        "ZYX", degrees=True
    ).reshape(frames, 78, 3)
    _write_bvh(
        virtual_root_path,
        names78,
        parents78,
        offsets78,
        euler78,
        translation,
        fps,
    )

    names77 = names78[1:]
    parents77 = [-1 if parent <= 0 else parent - 1 for parent in parents78[1:]]
    offsets77 = offsets78[1:]
    euler77 = euler78[:, 1:]
    _write_bvh(
        blender_path,
        names77,
        parents77,
        offsets77,
        euler77,
        translation,
        fps,
    )
