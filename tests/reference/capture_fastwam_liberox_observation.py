#!/usr/bin/env python3
"""Capture one frozen LIBERO-X observation without changing upstream code."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(quat[3]) / denominator)


def write_no_overwrite(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liberox-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.liberox_root.resolve()
    sys.path[:0] = [str(root), str(root / "packages/openpi-client/src")]
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.parse_bddl import parse_bddl_file
    from openpi_client import image_tools

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "wam-liberox-parity-manifest-v1":
        raise ValueError("unsupported LIBERO-X parity manifest")
    task = manifest["task"]
    init = manifest["init"]
    evaluation = manifest["evaluation"]
    if not evaluation["flip_images"] or evaluation["resize_size"] != 224:
        raise ValueError("reference capture requires flipped 224x224 images")

    bddl_path = (root / "libero/libero_x/bddl" /
                 manifest["scene_group"] / task["bddl_file"])
    init_path = (root / "libero/libero_x/init" /
                 manifest["scene_group"] / init["source_file"])
    if sha256(bddl_path) != task["bddl_sha256"]:
        raise ValueError("BDDL digest differs from the frozen manifest")
    if sha256(init_path) != init["source_sha256"]:
        raise ValueError("init-state digest differs from the frozen manifest")

    states = torch.load(init_path, map_location="cpu")
    source_index = int(init["source_index"])
    if source_index < 0 or source_index >= len(states):
        raise ValueError("manifest init-state index is out of range")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path, camera_heights=256, camera_widths=256,
        horizon=(int(evaluation["max_steps"]) +
                 int(evaluation["num_steps_wait"]) + 1))
    try:
        env.seed(int(init["simulator_seed"]))
        env.reset()
        state_vector = states[source_index]
        observation = env.regenerate_obs_from_state(
            state_vector.numpy() if hasattr(state_vector, "numpy")
            else state_vector)
    finally:
        env.close()

    size = int(evaluation["resize_size"])
    scene = np.ascontiguousarray(
        observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1])
    scene = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(scene, size, size))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, size, size))
    state = np.concatenate((
        observation["robot0_eef_pos"],
        quat_to_axis_angle(observation["robot0_eef_quat"]),
        observation["robot0_gripper_qpos"],
    )).astype("<f4")
    if scene.shape != (224, 224, 3) or wrist.shape != (224, 224, 3):
        raise ValueError("captured image geometry differs from PolicySpec")
    if state.shape != (8,):
        raise ValueError("captured state is not 8D")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_no_overwrite(output / "camera0.rgb", scene.tobytes())
    write_no_overwrite(output / "camera1.rgb", wrist.tobytes())
    write_no_overwrite(output / "state.f32", state.tobytes())
    instruction = str(parse_bddl_file(bddl_path)["language"])
    record = {
        "format": "wam-fastwam-liberox-observation-v1",
        "manifest_sha256": sha256(args.manifest.resolve()),
        "bddl_sha256": sha256(bddl_path),
        "init_sha256": sha256(init_path),
        "source_init_index": source_index,
        "instruction": instruction,
        "scene_shape": list(scene.shape),
        "wrist_shape": list(wrist.shape),
        "state_shape": list(state.shape),
    }
    write_no_overwrite(
        output / "observation.json",
        (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
