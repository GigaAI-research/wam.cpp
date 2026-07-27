#!/usr/bin/env python3
"""Convert one action-only FastWAM checkpoint to a PolicySpec GGUF."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ModuleNotFoundError:
    load_safetensors = None

try:
    import gguf
except ModuleNotFoundError as exc:
    raise SystemExit("missing gguf-py; add llama.cpp/gguf-py to PYTHONPATH") from exc

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.fastwam_gguf import (  # noqa: E402
    ARCH,
    CONVERSION_POLICY,
    CONVERTER_REVISION,
    EXPECTED_COMPONENT_COUNTS,
    load_policy_profile,
    publish_no_overwrite,
)
from common.policy_spec import add_policy_spec_metadata  # noqa: E402


def _load_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return payload


def _load_state(path: Path) -> Mapping[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        if load_safetensors is None:
            raise ValueError("safetensors is required for the VAE input")
        state = load_safetensors(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model_state", "state_dict", "model"):
            if isinstance(payload, Mapping) and isinstance(payload.get(key), Mapping):
                payload = payload[key]
                break
        if not isinstance(payload, Mapping):
            raise ValueError(f"VAE must contain a state dict: {path}")
        state = payload
    return {str(key): value for key, value in state.items()
            if isinstance(value, torch.Tensor)}


def _bf16_bytes(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().contiguous().cpu().view(torch.uint16).numpy()


def _add_tensor(writer: Any, name: str, tensor: torch.Tensor,
                *, force_f32: bool = False) -> None:
    tensor = tensor.detach().contiguous().cpu()
    if tensor.ndim > 4:
        raise ValueError(
            f"GGML supports at most four dimensions: {name} has {tuple(tensor.shape)}")
    if force_f32:
        writer.add_tensor(name, tensor.to(torch.float32).numpy(),
                          raw_dtype=gguf.GGMLQuantizationType.F32)
    else:
        writer.add_tensor(name, _bf16_bytes(tensor.to(torch.bfloat16)),
                          raw_shape=list(tensor.shape),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)


def _compact(component: str, source: str) -> str:
    value = source
    for old, new in (
        ("patch_embedding", "patch"), ("action_encoder", "encoder"),
        ("text_embedding", "text"), ("time_embedding", "time"),
        ("time_projection", "timep"), ("self_attn", "sa"),
        ("cross_attn", "ca"), ("norm_q", "qn"), ("norm_k", "kn"),
        ("modulation", "mod"), ("blocks", "blk"), ("ffn", "ff"),
        ("encoder.conv1", "encoder.in"),
        ("encoder.downsamples", "encoder.down"),
        ("encoder.middle", "encoder.mid"), ("encoder.head", "encoder.out"),
        ("conv1", "quant"), ("residual", "res"), ("shortcut", "skip"),
        ("resample", "rs"), ("time_conv", "tc"), ("to_qkv", "qkv"),
        ("proj", "p"),
    ):
        value = value.replace(old, new)
    value = value.removeprefix("model.")
    result = f"fastwam.{component}.{value}"
    if len(result.encode("utf-8")) >= 64:
        raise ValueError(f"compact tensor name is too long: {result}")
    return result


def _iter_mot(state: Mapping[str, torch.Tensor]) -> Iterable[tuple[str, str, torch.Tensor]]:
    for source, tensor in sorted(state.items()):
        if source.startswith("mixtures.video."):
            component = "video"
            body = source[len("mixtures.video."):]
        elif source.startswith("mixtures.action."):
            component = "action"
            body = source[len("mixtures.action."):]
        else:
            continue
        if component == "video" and body == "patch_embedding.weight" and tensor.ndim == 5:
            if tensor.shape[2] != 1:
                raise ValueError(f"unsupported temporal patch kernel {tuple(tensor.shape)}")
            tensor = tensor.squeeze(2)
        yield component, _compact(component, body), tensor


def _iter_vae(state: Mapping[str, torch.Tensor]) -> Iterable[tuple[str, str, torch.Tensor]]:
    for source, tensor in sorted(state.items()):
        source = source.removeprefix("model.")
        if not source.startswith("encoder.") and not source.startswith("conv1."):
            continue
        if tensor.ndim == 5:
            tensor = tensor[:, :, -1, :, :]
        yield "vae", _compact("vae", source), tensor


def _stats(path: Path, profile: Mapping[str, Any]) -> dict[str, np.ndarray]:
    root = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, np.ndarray] = {}
    epsilon = float(profile["normalization"]["epsilon"])
    for domain, size in (("action", profile["action"]["model_dim"]),
                         ("state", profile["state"]["model_dim"])):
        group = root.get(domain, {}).get("default")
        if not isinstance(group, Mapping):
            raise ValueError(f"normalization JSON has no {domain}.default group")
        kind = profile["normalization"][domain]["kind"]
        fields = (("global_min", "lower"), ("global_max", "upper")) \
            if kind == "min_max" else \
            (("global_mean", "mean"), ("global_std", "std"))
        statistics = []
        for source, target in fields:
            values = np.asarray(group.get(source), dtype=np.float32).reshape(-1)
            if values.size < size:
                raise ValueError(
                    f"{domain}.{source} has {values.size} values, expected {size}")
            values = np.ascontiguousarray(values[:size])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{domain}.{source} contains NaN or Inf")
            result[f"wam.norm.{domain}.{target}"] = values
            statistics.append(values)
        if kind == "min_max" and np.any(
                statistics[1] - statistics[0] <= epsilon):
            raise ValueError(f"{domain} normalization range is too small")
        if kind == "z_score" and np.any(statistics[1] <= epsilon):
            raise ValueError(f"{domain} normalization std is too small")
    return result


def _geometry(mot: Mapping[str, torch.Tensor],
              proprio: Mapping[str, torch.Tensor],
              action_dim: int, proprio_dim: int) -> dict[str, int]:
    def need(name: str) -> torch.Tensor:
        value = mot.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint is missing tensor {name}")
        return value

    video_patch = need("mixtures.video.patch_embedding.weight")
    action_encoder = need("mixtures.action.action_encoder.weight")
    video_text = need("mixtures.video.text_embedding.0.weight")
    if video_patch.ndim != 5 or action_encoder.ndim != 2 or video_text.ndim != 2:
        raise ValueError("FastWAM checkpoint has unexpected projection ranks")
    if int(action_encoder.shape[1]) != action_dim:
        raise ValueError("PolicySpec action dimension differs from checkpoint")
    layer_ids = {
        int(match.group(1)) for name in mot
        if (match := re.match(r"mixtures\.video\.blocks\.(\d+)\.", str(name)))
    }
    action_layers = {
        int(match.group(1)) for name in mot
        if (match := re.match(r"mixtures\.action\.blocks\.(\d+)\.", str(name)))
    }
    if not layer_ids or layer_ids != set(range(len(layer_ids))) or action_layers != layer_ids:
        raise ValueError("video/action transformer layer indices are inconsistent")
    weight = proprio.get("weight")
    bias = proprio.get("bias")
    text_dim = int(video_text.shape[1])
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (text_dim, proprio_dim):
        raise ValueError("proprio encoder weight shape differs from PolicySpec")
    if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (text_dim,):
        raise ValueError("proprio encoder bias shape differs from checkpoint")
    return {
        "video_hidden_dim": int(video_patch.shape[0]),
        "action_hidden_dim": int(action_encoder.shape[0]),
        "text_dim": text_dim,
        "num_layers": len(layer_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--policy-profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--action-shift", type=float, default=5.0)
    parser.add_argument("--video-shift", type=float, default=5.0)
    parser.add_argument("--norm-eps", type=float, default=1.0e-6)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--attn-head-dim", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = load_policy_profile(args.policy_profile)
    payload = _load_payload(args.checkpoint)
    mot = payload.get("mot")
    proprio = payload.get("proprio_encoder")
    if not isinstance(mot, Mapping) or not isinstance(proprio, Mapping):
        raise ValueError("checkpoint requires mot and proprio_encoder mappings")
    vae = _load_state(args.vae)
    action_dim = int(profile["action"]["model_dim"])
    proprio_dim = int(profile["state"]["model_dim"])
    geometry = _geometry(mot, proprio, action_dim, proprio_dim)
    if geometry["video_hidden_dim"] != args.num_heads * args.attn_head_dim:
        raise ValueError("video attention geometry is inconsistent")
    if (args.num_inference_steps <= 0 or args.action_shift <= 0 or
            args.video_shift <= 0 or args.norm_eps <= 0):
        raise ValueError("scheduler values must be positive")

    tensors: list[tuple[str, str, torch.Tensor, bool]] = []
    names: set[str] = set()
    counts = {name: 0 for name in EXPECTED_COMPONENT_COUNTS}
    for component, name, tensor in _iter_mot(mot):
        if name in names:
            raise ValueError(f"tensor name collision: {name}")
        names.add(name)
        counts[component] += 1
        tensors.append((component, name, tensor, False))
    for source, tensor in sorted(proprio.items()):
        if source == "weight" or source.endswith(".weight"):
            name = "fastwam.proprio.weight"
        elif source == "bias" or source.endswith(".bias"):
            name = "fastwam.proprio.bias"
        else:
            continue
        if name in names:
            raise ValueError(f"tensor name collision: {name}")
        names.add(name)
        counts["proprio"] += 1
        tensors.append(("proprio", name, tensor, False))
    for component, name, tensor in _iter_vae(vae):
        if name in names:
            raise ValueError(f"tensor name collision: {name}")
        names.add(name)
        counts[component] += 1
        tensors.append((component, name, tensor, False))
    for name, values in _stats(args.norm_stats, profile).items():
        counts["stats"] += 1
        tensors.append(("stats", name, torch.from_numpy(values), True))
    if counts != EXPECTED_COMPONENT_COUNTS:
        raise ValueError(
            f"FastWAM component counts differ from the Gate B contract: {counts}")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    print(f"validated {len(tensors)} tensors; model BF16, PolicySpec stats F32")
    if args.dry_run:
        return 0

    composition = profile["images"]["composition"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.incomplete.{os.getpid()}")
    writer = gguf.GGUFWriter(str(temporary), arch=ARCH)
    writer.add_name("FastWAM action-only PolicySpec BF16")
    writer.add_string("fastwam.conversion_policy", CONVERSION_POLICY)
    writer.add_string("fastwam.converter_revision", CONVERTER_REVISION)
    writer.add_string("fastwam.variant", "uncond_action_only")
    for key, value in (
        ("image_height", composition["height"]),
        ("image_width", composition["width"]),
        ("num_cameras", len(profile["images"]["roles"])),
        ("action_dim", action_dim), ("proprio_dim", proprio_dim),
        ("action_horizon", profile["action"]["horizon"]),
        ("inference_steps", args.num_inference_steps),
        ("latent_channels", 48), ("spatial_downsample", 16),
        ("temporal_downsample", 4),
        ("context_len", profile["language"]["max_tokens"]),
        ("text_dim", geometry["text_dim"]),
        ("video_hidden_dim", geometry["video_hidden_dim"]),
        ("action_hidden_dim", geometry["action_hidden_dim"]),
        ("num_layers", geometry["num_layers"]),
        ("num_heads", args.num_heads),
        ("attn_head_dim", args.attn_head_dim),
    ):
        writer.add_uint32(f"fastwam.{key}", int(value))
    writer.add_float32("fastwam.action_shift", float(args.action_shift))
    writer.add_float32("fastwam.video_shift", float(args.video_shift))
    writer.add_float32("fastwam.norm_eps", float(args.norm_eps))
    add_policy_spec_metadata(writer, profile)
    for _, name, tensor, force_f32 in tensors:
        _add_tensor(writer, name, tensor, force_f32=force_f32)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        publish_no_overwrite(temporary, args.out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
