#!/usr/bin/env python3
"""Create or verify the path-free GWP05 donor parity manifest."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import sys
from pathlib import Path


BASE_STAGES = (
    "image_input",
    "normalized_state",
    "vae_conv_in",
    "vae_down0",
    "vae_down1",
    "vae_down2",
    "vae_down3",
    "vae_latent",
    "t5_embedding",
    "action_tokens",
    "visual_tokens",
    "action_condition",
    "block0_action",
    "velocity_step0",
    "flow_timesteps",
    "flow_sigmas",
)

CACHE_STAGES = (
    "prefix_state_hidden_in",
    "prefix_visual_hidden_in",
    "prefix_projected_action_prompt",
    "prefix_layer_00_state_hidden_out",
    "prefix_layer_00_visual_hidden_out",
    "prefix_layer_00_self_key",
    "prefix_layer_00_self_value",
    "prefix_layer_00_prompt_key",
    "prefix_layer_00_prompt_value",
    "prefix_layer_29_state_hidden_out",
    "prefix_layer_29_visual_hidden_out",
    "prefix_layer_29_self_key",
    "prefix_layer_29_self_value",
    "prefix_layer_29_prompt_key",
    "prefix_layer_29_prompt_value",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_record(data: bytes, shape: list[int], derived_from: list[str]) -> dict[str, object]:
    return {
        "dtype": "float32",
        "shape": shape,
        "byte_order": "little",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "derived_from": derived_from,
    }


def read_f32(path: Path) -> array:
    result = array("f")
    with path.open("rb") as source:
        result.fromfile(source, path.stat().st_size // 4)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def little_endian_bytes(values: array) -> bytes:
    if sys.byteorder == "little":
        return values.tobytes()
    copy = array("f", values)
    copy.byteswap()
    return copy.tobytes()


def stage_names() -> tuple[str, ...]:
    denoise = []
    for step in range(10):
        prefix = f"denoise_{step:02d}"
        denoise.extend(
            (f"{prefix}_action_in", f"{prefix}_dt",
             f"{prefix}_velocity", f"{prefix}_action_out")
        )
    return (*BASE_STAGES, *denoise, "action")


def read_stage(root: Path, name: str) -> dict[str, object]:
    data = root / f"{name}.f32"
    metadata_path = root / f"{name}.json"
    if not data.is_file() or not metadata_path.is_file():
        raise ValueError(f"missing donor stage {name!r}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = metadata.get("shape")
    if metadata.get("format") != "gwp-f32-dump-v1" or \
            metadata.get("dtype") != "float32" or \
            metadata.get("byte_order") != "little" or \
            not isinstance(shape, list) or not shape:
        raise ValueError(f"invalid donor stage metadata: {metadata_path}")
    elements = math.prod(shape)
    if metadata.get("elements") != elements or data.stat().st_size != elements * 4:
        raise ValueError(f"donor stage size mismatch: {name}")
    return {
        "path": f"{name}.f32",
        "dtype": "float32",
        "shape": shape,
        "byte_order": "little",
        "bytes": data.stat().st_size,
        "sha256": sha256(data),
    }


def replay_record(root: Path) -> dict[str, object]:
    manifest_path = root / "fixed" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "wam-gwp05-replay-manifest-v1":
        raise ValueError("unsupported donor replay manifest")
    files = manifest["files"]
    selected = {
        "camera_high": files["camera_high"],
        "camera_left_wrist": files["camera_left_wrist"],
        "camera_right_wrist": files["camera_right_wrist"],
        "prompt_embedding": files["prompt_embedding"],
        "state": files["state"],
        "action_noise": files["action_noise"],
    }
    for record in selected.values():
        path = root / "fixed" / str(record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"replay payload differs: {path.name}")
    return {
        "case": "fixed",
        "camera_order": manifest["contract"]["camera_order"],
        "scheduler": manifest["contract"]["scheduler"],
        "denoise_steps": manifest["contract"]["denoise_steps"],
        "payloads": selected,
    }


def derived_boundaries(replay_root: Path, stage_root: Path) -> dict[str, object]:
    patchified = read_f32(stage_root / "image_input.f32")
    canvas = array("f", [0.0]) * (384 * 320 * 3)
    for channel in range(3):
        for dx in range(2):
            for dy in range(2):
                patch_channel = (channel * 2 + dx) * 2 + dy
                for y in range(192):
                    for x in range(160):
                        source = (patch_channel * 192 + y) * 160 + x
                        target = ((y * 2 + dy) * 320 + x * 2 + dx) * 3 + channel
                        canvas[target] = patchified[source]

    normalized = read_f32(stage_root / "denoise_09_action_out.f32")
    if len(normalized) != 48 * 32:
        raise ValueError("final normalized action has the wrong shape")
    trimmed = array("f")
    for step in range(48):
        trimmed.extend(normalized[step * 32:step * 32 + 14])

    recovered = read_f32(stage_root / "action.f32")
    state = read_f32(replay_root / "fixed" / "state.f32")
    if len(recovered) != 48 * 14 or len(state) != 14:
        raise ValueError("action recovery inputs have the wrong shape")
    reference_indices = (0, 1, 2, 3, 4, 5, -1, 7, 8, 9, 10, 11, 12, -1)
    mixed = array("f", recovered)
    for step in range(48):
        for dimension, state_index in enumerate(reference_indices):
            if state_index >= 0:
                mixed[step * 14 + dimension] -= state[state_index]

    return {
        "composed_image": bytes_record(
            little_endian_bytes(canvas), [384, 320, 3], ["image_input"]),
        "normalized_model_action": bytes_record(
            little_endian_bytes(normalized), [48, 32],
            ["denoise_09_action_out"]),
        "normalized_trimmed_action": bytes_record(
            little_endian_bytes(trimmed), [48, 14],
            ["denoise_09_action_out"]),
        "unnormalized_mixed_action": bytes_record(
            little_endian_bytes(mixed), [48, 14], ["action", "input.state"]),
        "joint_position_action": bytes_record(
            little_endian_bytes(recovered), [48, 14], ["action"]),
        "reference_state_indices": list(reference_indices),
    }


def build_manifest(replay_root: Path, stage_root: Path,
                   cache_stage_root: Path,
                   donor_executable: Path) -> dict[str, object]:
    stages = {name: read_stage(stage_root, name) for name in stage_names()}
    cache_stages = {
        name: read_stage(cache_stage_root, name) for name in CACHE_STAGES
    }
    replay = replay_record(replay_root)
    if stages["denoise_00_action_in"]["sha256"] != \
            replay["payloads"]["action_noise"]["sha256"]:
        raise ValueError("stage-0 action does not equal the frozen explicit noise")
    if stages["t5_embedding"]["sha256"] != \
            replay["payloads"]["prompt_embedding"]["sha256"]:
        raise ValueError("stage text embedding does not equal the frozen request")
    return {
        "format": "wam-gwp05-slice4a-reference-v1",
        "scope": "legacy F32 32-dim quantile donor engine; not the 14-dim RoboTwin checkpoint",
        "donor_executable": {
            "name": donor_executable.name,
            "bytes": donor_executable.stat().st_size,
            "sha256": sha256(donor_executable),
        },
        "input": replay,
        "engine_oracle": {
            "execution": "F32 CPU complete MoT graph with explicit action noise",
            "stages": stages,
        },
        "cache_oracle": {
            "execution": "audited F32 CUDA prefix-cache capture",
            "stages": cache_stages,
            "note": "kept separate because the current container exposed no CUDA device",
        },
        "policy_boundaries": derived_boundaries(replay_root, stage_root),
        "tolerances": {
            "f32_final_action": {"mean_abs": 0.001, "max_abs": 0.001},
            "default_intermediate_mean_abs": 0.0002,
            "denoise_09_velocity_mean_abs": 0.005,
            "exact": ["flow_timesteps", "flow_sigmas", *(
                f"denoise_{step:02d}_dt" for step in range(10))],
        },
        "distribution": {
            "payloads_committed": False,
            "reason": "checkpoint and replay redistribution rights are not recorded",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--cache-stage-root", type=Path, required=True)
    parser.add_argument("--donor-executable", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = build_manifest(
        args.replay_root.resolve(), args.stage_root.resolve(),
        args.cache_stage_root.resolve(), args.donor_executable.resolve())
    encoded = json.dumps(actual, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(encoded, encoding="utf-8")
        return 0
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("donor reference differs from the frozen manifest")
    print("GWP05 donor parity reference: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
