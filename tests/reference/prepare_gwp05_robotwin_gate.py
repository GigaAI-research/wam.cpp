#!/usr/bin/env python3
"""Freeze one RoboTwin observation for the GWP05 14D Gate A parity."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
from PIL import Image


VIEW_COLUMNS = {
    "camera_high.ppm": "observation.images.cam_high",
    "camera_left_wrist.ppm": "observation.images.cam_left_wrist",
    "camera_right_wrist.ppm": "observation.images.cam_right_wrist",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)


def ppm_payload(image: Image.Image) -> tuple[bytes, list[int]]:
    array = np.ascontiguousarray(np.asarray(image.convert("RGB")), dtype=np.uint8)
    height, width, channels = array.shape
    if channels != 3:
        raise RuntimeError("decoded image is not RGB")
    return f"P6\n{width} {height}\n255\n".encode("ascii") + array.tobytes(), [height, width, 3]


def load_instruction(tasks_path: Path, task_index: int) -> str:
    with tasks_path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if int(row["task_index"]) == task_index:
                instruction = str(row["task"]).strip()
                if not instruction:
                    break
                return instruction
    raise RuntimeError(f"task index {task_index} is absent from {tasks_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=20)
    parser.add_argument("--noise-seed", type=int, default=20260725)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.frame_index < 0:
        parser.error("frame-index must be non-negative")
    if args.output_dir.exists():
        if any(args.output_dir.iterdir()):
            raise SystemExit(f"refusing to write into nonempty directory: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)

    table = parquet.read_table(args.parquet)
    if args.frame_index >= table.num_rows:
        raise SystemExit(
            f"frame {args.frame_index} is outside parquet length {table.num_rows}")
    row = args.frame_index
    if int(table["frame_index"][row].as_py()) != row:
        raise RuntimeError("parquet row and frame_index differ")
    task_index = int(table["task_index"][row].as_py())
    instruction = load_instruction(args.tasks, task_index)

    outputs: dict[str, dict[str, object]] = {}
    for output_name, column in VIEW_COLUMNS.items():
        encoded = table[column][row].as_py()
        payload, shape = ppm_payload(Image.open(io.BytesIO(encoded["bytes"])))
        path = args.output_dir / output_name
        write_new(path, payload)
        outputs[output_name] = {
            "dtype": "uint8",
            "shape": shape,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    state = np.asarray(table["observation.state"][row].as_py(), dtype="<f4")
    if state.shape != (14,) or not np.isfinite(state).all():
        raise RuntimeError(f"state has invalid shape or values: {state.shape}")
    state_path = args.output_dir / "state.f32"
    write_new(state_path, state.tobytes())
    outputs[state_path.name] = {
        "dtype": "float32-le",
        "shape": [14],
        "size_bytes": state_path.stat().st_size,
        "sha256": sha256_file(state_path),
    }

    generator = np.random.default_rng(args.noise_seed)
    noise = generator.standard_normal((48, 14), dtype=np.float32).astype("<f4")
    noise_path = args.output_dir / "noise.f32"
    write_new(noise_path, noise.tobytes())
    outputs[noise_path.name] = {
        "dtype": "float32-le",
        "shape": [48, 14],
        "size_bytes": noise_path.stat().st_size,
        "sha256": sha256_file(noise_path),
    }

    prompt_path = args.output_dir / "prompt.txt"
    write_new(prompt_path, (instruction + "\n").encode("utf-8"))
    outputs[prompt_path.name] = {
        "encoding": "utf-8",
        "size_bytes": prompt_path.stat().st_size,
        "sha256": sha256_file(prompt_path),
    }

    manifest = {
        "format": "wam-gwp05-robotwin-gate-input-v1",
        "source": {
            "parquet": str(args.parquet.resolve()),
            "frame_index": row,
            "episode_index": int(table["episode_index"][row].as_py()),
            "task_index": task_index,
            "instruction": instruction,
        },
        "noise": {
            "generator": "numpy.default_rng.PCG64",
            "seed": args.noise_seed,
        },
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "input-manifest.json"
    write_new(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
