#!/usr/bin/env python3
"""Capture the independent PyTorch BF16 oracle for GWP05 RoboTwin Gate A."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as output:
        output.write(payload)


def load_module(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("wam_robotwin_gate_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import inference module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if value.size != expected:
        raise RuntimeError(f"{path} has {value.size} values, expected {expected}")
    return np.ascontiguousarray(value.reshape(shape), dtype=np.float32)


def install_fixed_noise(pipeline, noise: np.ndarray) -> dict[str, torch.Tensor]:
    original = pipeline.prepare_latents
    fixed = torch.from_numpy(noise).unsqueeze(0)
    captured: dict[str, torch.Tensor] = {}

    def prepare_latents(instance, *args, **kwargs):
        outputs = original(*args, **kwargs)
        if len(outputs) != 4:
            raise RuntimeError(
                f"expand_timesteps prepare_latents returned {len(outputs)} values")
        captured["vae_latent"] = outputs[1][:, :, 0].detach()
        action = fixed.to(device=outputs[-1].device, dtype=outputs[-1].dtype)
        return (*outputs[:-1], action)

    pipeline.prepare_latents = types.MethodType(prepare_latents, pipeline)
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-script", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    if args.output_dir.exists():
        if any(args.output_dir.iterdir()):
            raise SystemExit(f"refusing to write into nonempty directory: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("Gate A PyTorch reference requires an available CUDA device")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(device)

    inference = load_module(args.inference_script.resolve())
    state = read_f32(args.input_dir / "state.f32", (14,))
    noise = read_f32(args.input_dir / "noise.f32", (48, 14))
    prompt = (args.input_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    images = [
        Image.open(args.input_dir / name).convert("RGB")
        for name in (
            "camera_high.ppm", "camera_left_wrist.ppm",
            "camera_right_wrist.ppm")
    ]
    reference_image = inference.compose_three_views(*images, width=320, height=384)
    stats = inference.load_norm_stats(str(args.norm_stats), device)

    load_begin = time.perf_counter()
    pipeline = inference.build_pipeline(
        str(args.checkpoint), str(args.base_model), device, torch.bfloat16)
    load_milliseconds = (time.perf_counter() - load_begin) * 1000.0
    captured = install_fixed_noise(pipeline, noise)

    with torch.inference_mode():
        processed_image = pipeline.video_processor.preprocess(
            reference_image, height=384, width=320).to(
                device=device, dtype=torch.float32)
        if tuple(processed_image.shape) != (1, 3, 384, 320):
            raise RuntimeError(
                f"processed image shape is {tuple(processed_image.shape)}")
        prompt_embedding = pipeline._get_t5_prompt_embeds(
            prompt=[prompt], num_videos_per_prompt=1,
            max_sequence_length=64, device=device, dtype=torch.bfloat16)
        if tuple(prompt_embedding.shape) != (1, 64, 4096):
            raise RuntimeError(
                f"prompt embedding shape is {tuple(prompt_embedding.shape)}")
        normalized_state = (
            torch.as_tensor(state, device=device) - stats["state_mean"]
        ) / stats["state_std"]
        torch.cuda.synchronize(device)
        predict_begin = time.perf_counter()
        normalized_action = pipeline(
            height=384,
            width=320,
            action_chunk=48,
            action_dim=14,
            state=normalized_state.unsqueeze(0),
            num_frames=5,
            guidance_scale=0.0,
            num_inference_steps=10,
            image=reference_image,
            return_dict=False,
            prompt_embeds=prompt_embedding,
            generator=None,
            max_sequence_length=64,
        )
        torch.cuda.synchronize(device)
        predict_milliseconds = (time.perf_counter() - predict_begin) * 1000.0
        normalized_action = normalized_action[0, :48, :14].float()
        if "vae_latent" not in captured:
            raise RuntimeError("prepare_latents did not expose the VAE latent")
        vae_latent = captured["vae_latent"]
        if tuple(vae_latent.shape) != (1, 48, 24, 20):
            raise RuntimeError(f"VAE latent shape is {tuple(vae_latent.shape)}")
        action_delta = (
            normalized_action * stats["action_std"] + stats["action_mean"])
        delta_mask = torch.as_tensor(
            [True] * 6 + [False] + [True] * 6 + [False],
            device=device, dtype=action_delta.dtype)
        action = action_delta + torch.as_tensor(
            state, device=device).unsqueeze(0) * delta_mask.unsqueeze(0)

    arrays = {
        "processed_image.f32": np.ascontiguousarray(
            processed_image[0].cpu().numpy(), dtype="<f4"),
        "normalized_state.f32": np.ascontiguousarray(
            normalized_state.float().cpu().numpy(), dtype="<f4"),
        "prompt_embedding.f32": np.ascontiguousarray(
            prompt_embedding[0].float().cpu().numpy(), dtype="<f4"),
        "attention_mask.i32": np.ones((64,), dtype="<i4"),
        "vae_latent.f32": np.ascontiguousarray(
            vae_latent[0].float().cpu().numpy(), dtype="<f4"),
        "normalized_action.f32": np.ascontiguousarray(
            normalized_action.cpu().numpy(), dtype="<f4"),
        "action.f32": np.ascontiguousarray(action.cpu().numpy(), dtype="<f4"),
    }
    outputs: dict[str, dict[str, object]] = {}
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN or Inf")
        path = args.output_dir / name
        write_new(path, value.tobytes())
        outputs[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "format": "wam-gwp05-robotwin-pytorch-bf16-reference-v1",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "deterministic_algorithms": True,
            "allow_tf32": False,
            "dtype": "bfloat16",
            "seed": args.seed,
        },
        "contract": {
            "profile": "gwp05_robotwin_dual_arm_14d_zscore",
            "prompt_tokens": 64,
            "action_shape": [48, 14],
            "explicit_action_noise": True,
            "inference_steps": 10,
            "flow_shift": 5.0,
        },
        "inputs": {
            "input_manifest": str((args.input_dir / "input-manifest.json").resolve()),
            "input_manifest_sha256": sha256_file(args.input_dir / "input-manifest.json"),
            "checkpoint": str(args.checkpoint.resolve()),
            "base_model": str(args.base_model.resolve()),
            "normalization": str(args.norm_stats.resolve()),
            "inference_script": str(args.inference_script.resolve()),
            "inference_script_sha256": sha256_file(args.inference_script),
        },
        "timing": {
            "load_milliseconds": load_milliseconds,
            "predict_milliseconds": predict_milliseconds,
            "peak_device_memory_bytes": torch.cuda.max_memory_allocated(device),
        },
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "reference-manifest.json"
    write_new(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
