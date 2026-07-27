#!/usr/bin/env python3
"""Generate an independent FastWAM LIBERO-X action reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_array(path: Path, dtype, shape) -> np.ndarray:
    result = np.fromfile(path, dtype=dtype)
    if result.size != int(np.prod(shape)):
        raise ValueError(f"invalid fixture array: {path}")
    return np.ascontiguousarray(result.reshape(shape))


def write_no_overwrite(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as stream:
        stream.write(np.ascontiguousarray(array).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fastwam-root", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    args = parser.parse_args()

    fastwam_root = args.fastwam_root.resolve()
    sys.path[:0] = [str(fastwam_root), str(fastwam_root / "src")]
    from experiments.liberox.fastwam_liberox_policy import (
        build_policy_from_args)
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    source = args.observation_dir.resolve()
    observation_record = json.loads(
        (source / "observation.json").read_text(encoding="utf-8"))
    scene = read_array(source / "camera0.rgb", np.uint8, (224, 224, 3))
    wrist = read_array(source / "camera1.rgb", np.uint8, (224, 224, 3))
    state = read_array(source / "state.f32", "<f4", (8,))
    instruction = observation_record["instruction"]

    policy_args = SimpleNamespace(
        sim_cfg_name="sim_liberox.yaml",
        task="liberox_train_2cam224_1e-4",
        ckpt=str(args.checkpoint.resolve()),
        dataset_stats=str(args.dataset_stats.resolve()),
        device=args.device, mixed_precision="bf16", action_horizon=32,
        replan_steps=10, num_inference_steps=args.num_inference_steps,
        sigma_shift=None, seed=args.seed, text_cfg_scale=1.0,
        negative_prompt="", rand_device="cpu", tiled=False,
        binarize_gripper=True, invert_gripper=True, gripper_debug=False,
        gripper_filter="none", gripper_switch_threshold=0.2,
        gripper_switch_count=3, timing_enabled=True)
    policy = build_policy_from_args(policy_args)
    observation = {
        "observation/image": scene,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": instruction,
    }
    image = policy._build_image_tensor(observation)
    proprio = policy._normalize_state(state)
    prompt = DEFAULT_PROMPT.format(task=instruction)
    debug = {"action_velocity": [], "action_state": []}
    model = policy.model
    original_encode = model._encode_input_image_latents_tensor
    original_prefill = model.mot.prefill_video_cache
    original_predict = model._predict_action_noise_with_cache
    original_step = model.infer_action_scheduler.step

    def capture_encode(self, *capture_args, **capture_kwargs):
        value = original_encode(*capture_args, **capture_kwargs)
        debug["vae_latent"] = value.detach().cpu().clone()
        return value

    def capture_prefill(*capture_args, **capture_kwargs):
        value = original_prefill(*capture_args, **capture_kwargs)
        debug["video_kv"] = [
            {name: tensor.detach().cpu().clone()
             for name, tensor in layer.items()}
            for layer in value]
        return value

    def capture_predict(self, *capture_args, **capture_kwargs):
        value = original_predict(*capture_args, **capture_kwargs)
        debug["action_velocity"].append(value.detach().cpu().clone())
        return value

    def capture_step(*capture_args, **capture_kwargs):
        value = original_step(*capture_args, **capture_kwargs)
        debug["action_state"].append(value.detach().cpu().clone())
        return value

    model._encode_input_image_latents_tensor = MethodType(capture_encode, model)
    model.mot.prefill_video_cache = capture_prefill
    model._predict_action_noise_with_cache = MethodType(capture_predict, model)
    model.infer_action_scheduler.step = capture_step
    with torch.inference_mode():
        context, context_mask = model.encode_prompt(prompt)
        prediction = model.infer_action(
            prompt=None, input_image=image, action_horizon=32,
            proprio=proprio, context=context, context_mask=context_mask,
            num_inference_steps=args.num_inference_steps, sigma_shift=None,
            seed=args.seed, rand_device="cpu", tiled=False)["action"]
    denormalized = policy._denormalize_action(prediction)[0]

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn((32, 7), generator=generator, dtype=torch.float32)
    noise = noise.to(dtype=torch.bfloat16).to(dtype=torch.float32)
    context_u16 = (context[0].detach().to(device="cpu", dtype=torch.bfloat16)
                   .contiguous().view(torch.uint16).numpy().copy())
    mask_i32 = (context_mask[0].detach().to(device="cpu", dtype=torch.int32)
                .contiguous().numpy().copy())
    normalized = np.ascontiguousarray(
        prediction.detach().to(device="cpu", dtype=torch.float32).numpy())
    denormalized = np.ascontiguousarray(denormalized, dtype=np.float32)

    if context_u16.shape != (128, 4096) or mask_i32.shape != (128,):
        raise ValueError("donor language geometry differs from PolicySpec")
    if normalized.shape != (32, 7) or denormalized.shape != (32, 7):
        raise ValueError("donor action geometry differs from PolicySpec")
    if not np.all(np.isfinite(denormalized)):
        raise ValueError("donor action contains NaN or Inf")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("camera0.rgb", "camera1.rgb", "state.f32"):
        write_no_overwrite(output / name, np.fromfile(source / name, np.uint8))
    write_no_overwrite(output / "context.bf16", context_u16.astype("<u2"))
    write_no_overwrite(output / "context-mask.i32", mask_i32.astype("<i4"))
    write_no_overwrite(output / "noise.f32", noise.numpy().astype("<f4"))
    write_no_overwrite(
        output / "expected-action-normalized.f32",
        normalized.astype("<f4"))
    write_no_overwrite(
        output / "expected-action.f32",
        denormalized.astype("<f4"))
    write_no_overwrite(
        output / "normalized-state.f32",
        proprio.detach().to(dtype=torch.float32).numpy().astype("<f4"))
    torch.save({
        **debug,
        "input_image": image.detach().cpu(),
        "context": context.detach().cpu(),
        "context_mask": context_mask.detach().cpu(),
        "proprio": proprio.detach().cpu(),
        "noise": noise,
        "action": torch.from_numpy(normalized.copy()),
        "action_denormalized": torch.from_numpy(denormalized.copy()),
    }, output / "intermediates.pt")
    record = {
        "format": "wam-fastwam-liberox-pytorch-reference-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "dataset_stats": str(args.dataset_stats.resolve()),
        "dataset_stats_sha256": sha256(args.dataset_stats.resolve()),
        "observation_sha256": sha256(source / "observation.json"),
        "instruction": instruction,
        "prompt": prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "normalization": "z_score",
        "action_shape": list(denormalized.shape),
        "parity_tolerances": {
            "mean_absolute_error": 9.0e-4,
            "maximum_absolute_error": 1.0e-2,
        },
    }
    with (output / "reference.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
