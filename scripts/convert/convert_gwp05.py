#!/usr/bin/env python3
"""Convert a GWP-0.5 policy and its Wan front-end to one GGUF file.

The produced file keeps source tensor names below three stable namespaces:
``gwp.*`` for the policy, ``t5.*`` for UMT5, and ``vae.*`` for the Wan VAE.
Those names plus the ``gwp05.*`` metadata form the on-disk ABI consumed by the
C++ loader.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import torch
from safetensors import safe_open

import gguf

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.gwp05_gguf import (  # noqa: E402
    ARCH,
    CONVERTER_REVISION,
    DTYPE_BYTES,
    POLICIES,
    add_policy_metadata,
    component_expected_counts,
    kv,
    policy,
    publish_no_overwrite,
    sha256_file,
    write_json_no_overwrite,
)
import common.gwp05_gguf as gguf_contract  # noqa: E402


KV = kv
DEFAULT_TRANSFORMER_FILE = "diffusion_pytorch_model.bin"
DEFAULT_STATS_KEYS = ("mean", "std", "q01", "q99")
TORCH_DTYPE_NAMES = {torch.float32: "F32", torch.bfloat16: "BF16"}


def _replace_all(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    for source, target in replacements:
        value = value.replace(source, target)
    return value


def _compact_name(component: str, source_name: str) -> str:
    """Map framework names to the <=63-byte GGUF tensor-name ABI."""
    if component == "gwp":
        name = _replace_all(
            source_name,
            (
                ("action_condition_embedder.", "acond."),
                ("condition_embedder.", "cond."),
                ("patch_embedding.", "patch."),
                ("state_encoder.", "state."),
                ("action_encoder.", "action."),
                ("action_decoder.", "decode."),
                ("blocks.", "blk."),
                (".action_expert.", ".a."),
                (".visual_expert.", ".v."),
                ("time_embedder.linear_1.", "time.0."),
                ("time_embedder.linear_2.", "time.1."),
                ("text_embedder.linear_1.", "text.0."),
                ("text_embedder.linear_2.", "text.1."),
                ("time_proj.", "time.out."),
                (".ffn.net.0.proj.", ".ffn.in."),
                (".ffn.net.2.", ".ffn.out."),
                (".attn1.", ".sa."),
                (".attn2.", ".ca."),
                (".to_out.0.", ".o."),
                (".to_qkv.", ".qkv."),
                (".to_q.", ".q."),
                (".to_k.", ".k."),
                (".to_v.", ".vproj."),
                (".norm_q.", ".qn."),
                (".norm_k.", ".kn."),
                (".norm2.", ".ca_norm."),
                ("in_proj.", "in."),
                ("mid_proj.", "mid."),
                ("out_proj.", "out."),
                ("scale_shift_table", "mod"),
            ),
        )
    elif component == "t5":
        name = _replace_all(
            source_name,
            (
                ("shared.", "token_embd."),
                ("encoder.final_layer_norm.", "final_norm."),
                ("encoder.block.", "blk."),
                (".layer.0.SelfAttention.", ".sa."),
                (".layer.0.layer_norm.", ".sa_norm."),
                (".layer.1.DenseReluDense.", ".ffn."),
                (".layer.1.layer_norm.", ".ffn_norm."),
                ("relative_attention_bias.", "rel."),
                ("wi_0.", "wi0."),
                ("wi_1.", "wi1."),
            ),
        )
    elif component == "vae":
        name = _replace_all(
            source_name,
            (
                ("encoder.", "enc."),
                ("down_blocks.", "down."),
                ("downsampler.resample.1.", "ds."),
                ("resnets.", "res."),
                ("mid_block.", "mid."),
                ("attentions.0.", "attn."),
                ("conv_shortcut.", "skip."),
                ("conv_in.", "in."),
                ("conv_out.", "out."),
                ("norm_out.gamma", "out_norm"),
                ("to_qkv.", "qkv."),
                ("quant_conv.", "quant."),
                ("norm1.gamma", "n1"),
                ("norm2.gamma", "n2"),
                ("norm.gamma", "norm"),
                ("conv1.", "c1."),
                ("conv2.", "c2."),
                ("proj.", "proj."),
            ),
        )
    else:
        raise ValueError(f"unknown tensor component {component}")
    result = f"{component}.{name}"
    if len(result.encode("utf-8")) >= 64:
        raise ValueError(f"compact tensor name is still too long: {result!r}")
    return result


def _validate_compact_names(groups: Mapping[str, Iterable[str]]) -> None:
    seen: dict[str, tuple[str, str]] = {}
    for component, names in groups.items():
        for source_name in names:
            compact = _compact_name(component, source_name)
            previous = seen.get(compact)
            if previous is not None:
                raise SystemExit(
                    f"tensor-name collision: {previous} and {(component, source_name)} -> {compact}"
                )
            seen[compact] = (component, source_name)


def _bf16_bytes(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16, got {tensor.dtype}")
    return tensor.view(torch.uint16).numpy()


def _policy(name: str) -> dict:
    return policy(name)


def _convert_tensor(component: str, tensor: torch.Tensor, policy: dict) -> torch.Tensor:
    actual = TORCH_DTYPE_NAMES.get(tensor.dtype)
    expected = policy["source"][component]
    if actual != expected:
        raise TypeError(
            f"{component} source tensor has dtype {tensor.dtype}; policy requires {expected}"
        )
    target = policy["stored"][component]
    if target == "BF16" and tensor.dtype != torch.bfloat16:
        return tensor.to(torch.bfloat16)
    if target == "F32" and tensor.dtype != torch.float32:
        return tensor.to(torch.float32)
    return tensor


def _packed_self_qkv_name(name: str) -> str | None:
    for suffix in ("weight", "bias"):
        marker = f".attn1.to_q.{suffix}"
        if name.endswith(marker):
            return name[:-len(marker)] + f".attn1.to_qkv.{suffix}"
    return None


def _iter_transformer_tensors(
    state: Mapping[str, torch.Tensor], policy: dict
) -> Iterator[tuple[str, torch.Tensor]]:
    if not policy.get("pack_self_qkv"):
        for name in sorted(state):
            yield name, state[name]
        return
    skipped = (".attn1.to_k.weight", ".attn1.to_v.weight",
               ".attn1.to_k.bias", ".attn1.to_v.bias")
    for name in sorted(state):
        packed_name = _packed_self_qkv_name(name)
        if packed_name is not None:
            marker = ".attn1.to_q."
            prefix, suffix = name.split(marker, 1)
            tensors = [state[f"{prefix}.attn1.to_{part}.{suffix}"] for part in "qkv"]
            yield packed_name, torch.cat(tensors, dim=0)
        elif not name.endswith(skipped):
            yield name, state[name]


def _transformer_descriptors(
    state: Mapping[str, torch.Tensor], policy: dict
) -> list[tuple[str, str, tuple[int, ...]]]:
    if not policy.get("pack_self_qkv"):
        return [(name, TORCH_DTYPE_NAMES.get(state[name].dtype, str(state[name].dtype)),
                 _shape(state[name])) for name in state]
    result = []
    skipped = (".attn1.to_k.weight", ".attn1.to_v.weight",
               ".attn1.to_k.bias", ".attn1.to_v.bias")
    for name in state:
        packed_name = _packed_self_qkv_name(name)
        if packed_name is not None:
            marker = ".attn1.to_q."
            prefix, suffix = name.split(marker, 1)
            tensors = [state[f"{prefix}.attn1.to_{part}.{suffix}"] for part in "qkv"]
            shape = list(_shape(tensors[0]))
            shape[0] *= 3
            result.append((packed_name,
                           TORCH_DTYPE_NAMES.get(tensors[0].dtype, str(tensors[0].dtype)),
                           tuple(shape)))
        elif not name.endswith(skipped):
            result.append((name, TORCH_DTYPE_NAMES.get(state[name].dtype, str(state[name].dtype)),
                           _shape(state[name])))
    return result


def _add_tensor(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    """Add a tensor without silently changing its source precision."""
    tensor = tensor.detach().contiguous().cpu()
    if tensor.ndim > 4:
        raise ValueError(f"GGML supports at most four dimensions: {name} has {tensor.shape}")
    if tensor.dtype == torch.float32:
        writer.add_tensor(name, tensor.numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)
    elif tensor.dtype == torch.bfloat16:
        writer.add_tensor(
            name,
            _bf16_bytes(tensor),
            raw_shape=list(tensor.shape),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    else:
        raise TypeError(f"unsupported dtype {tensor.dtype} for {name}")


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def _need_shape(tensors: Mapping[str, torch.Tensor], name: str, expected: tuple[int, ...]) -> None:
    if name not in tensors:
        raise SystemExit(f"GWP checkpoint is missing tensor {name}")
    actual = _shape(tensors[name])
    if actual != expected:
        raise SystemExit(f"{name}: shape {actual}, expected {expected}")


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing required JSON file: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot parse {path}: {exc}") from exc


def _load_transformer(checkpoint: Path) -> tuple[dict, Mapping[str, torch.Tensor]]:
    cfg = _load_json(checkpoint / "config.json")
    if cfg.get("_class_name") != "CasualWorldActionTransformer_MoT":
        raise SystemExit(
            "only CasualWorldActionTransformer_MoT is supported; "
            f"checkpoint declares {cfg.get('_class_name')!r}"
        )
    weight_path = checkpoint / DEFAULT_TRANSFORMER_FILE
    if not weight_path.is_file():
        raise SystemExit(f"missing GWP weights: {weight_path}")
    print(f"loading mmap checkpoint metadata: {weight_path}")
    try:
        state = torch.load(weight_path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:
        raise SystemExit(f"cannot load {weight_path}: {exc}") from exc
    if not isinstance(state, Mapping):
        raise SystemExit(f"{weight_path} does not contain a state dict")
    return cfg, state


def _validate_transformer(cfg: dict, state: Mapping[str, torch.Tensor]) -> None:
    n_layers = int(cfg["num_layers"])
    n_heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["attention_head_dim"])
    hidden = n_heads * head_dim
    action_hidden = int(cfg["action_expert_dim"])
    action_dim = int(cfg["in_action_channels"])
    embodiments = int(cfg["num_embodiments"])
    patch = tuple(int(v) for v in cfg["patch_size"])

    if n_layers <= 0 or n_heads <= 0 or head_dim <= 0:
        raise SystemExit("invalid transformer layer/head configuration")
    if int(cfg["out_action_channels"]) != action_dim:
        raise SystemExit("GWP in/out action channels differ; v1 requires equality")
    if cfg.get("image_dim") is not None or cfg.get("added_kv_proj_dim") is not None:
        raise SystemExit("GWP v1 only supports text-conditioned MoT checkpoints")

    _need_shape(state, "patch_embedding.weight", (hidden, int(cfg["in_channels"]), *patch))
    _need_shape(state, "state_encoder.in_proj.weight", (embodiments, action_dim, 128))
    _need_shape(state, "action_encoder.in_proj.weight", (embodiments, action_dim, 128))
    _need_shape(state, "action_decoder.out_proj.weight", (embodiments, 128, action_dim))
    _need_shape(state, "action_scale_shift_table", (1, 2, action_hidden))

    block_ids: set[int] = set()
    for name in state:
        if name.startswith("blocks."):
            try:
                block_ids.add(int(name.split(".", 2)[1]))
            except (IndexError, ValueError):
                raise SystemExit(f"malformed transformer block tensor name: {name}")
    if block_ids != set(range(n_layers)):
        raise SystemExit(f"transformer blocks are {sorted(block_ids)}, expected 0..{n_layers - 1}")

    for index in range(n_layers):
        prefix = f"blocks.{index}"
        _need_shape(state, f"{prefix}.action_expert.attn1.to_q.weight", (hidden, action_hidden))
        _need_shape(
            state,
            f"{prefix}.action_expert.ffn.net.0.proj.weight",
            (int(cfg["action_ffn_dim"]), action_hidden),
        )
        _need_shape(state, f"{prefix}.visual_expert.attn1.to_q.weight", (hidden, hidden))
        _need_shape(
            state,
            f"{prefix}.visual_expert.ffn.net.0.proj.weight",
            (int(cfg["ffn_dim"]), hidden),
        )

    bad_dtype = [
        (name, tensor.dtype)
        for name, tensor in state.items()
        if tensor.dtype not in (torch.float32, torch.bfloat16)
    ]
    if bad_dtype:
        raise SystemExit(f"unsupported GWP tensor dtype(s): {bad_dtype[:8]}")


def _safetensor_index(component: Path) -> tuple[dict, dict[str, Path]]:
    index_path = component / "model.safetensors.index.json"
    if not index_path.is_file():
        index_path = component / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SystemExit(f"invalid safetensors index: {index_path}")
        paths = {name: component / shard for name, shard in weight_map.items()}
        missing = sorted({path for path in paths.values() if not path.is_file()})
        if missing:
            raise SystemExit(f"missing safetensors shard(s): {missing}")
        return index, paths

    candidates = sorted(component.glob("*.safetensors"))
    if len(candidates) != 1:
        raise SystemExit(
            f"expected one safetensors file or an index in {component}, got {candidates}"
        )
    with safe_open(candidates[0], framework="pt", device="cpu") as source:
        paths = {name: candidates[0] for name in source.keys()}
    return {}, paths


def _iter_safetensors(weight_map: Mapping[str, Path]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one shard at a time to cap temporary file mappings."""
    by_path: dict[Path, list[str]] = {}
    for name, path in weight_map.items():
        by_path.setdefault(path, []).append(name)
    for path in sorted(by_path):
        print(f"opening {path}")
        with safe_open(path, framework="pt", device="cpu") as source:
            available = set(source.keys())
            for name in sorted(by_path[path]):
                if name not in available:
                    raise SystemExit(f"{path} does not contain indexed tensor {name}")
                yield name, source.get_tensor(name)


def _output_element_count(component: str, name: str, shape: Iterable[int]) -> int:
    dimensions = [int(value) for value in shape]
    if component == "gwp" and name == "patch_embedding.weight":
        if len(dimensions) != 5 or dimensions[2] != 1:
            raise SystemExit(f"unsupported GWP patch kernel shape {tuple(dimensions)}")
        dimensions.pop(2)
    if component == "vae" and len(dimensions) == 5:
        dimensions.pop(2)
    result = 1
    for value in dimensions:
        result *= value
    return result


def _inspect_safetensors(weight_map: Mapping[str, Path]) -> dict[str, tuple[str, tuple[int, ...]]]:
    result: dict[str, tuple[str, tuple[int, ...]]] = {}
    by_path: dict[Path, list[str]] = {}
    for name, path in weight_map.items():
        by_path.setdefault(path, []).append(name)
    for path in sorted(by_path):
        with safe_open(path, framework="pt", device="cpu") as source:
            available = set(source.keys())
            for name in sorted(by_path[path]):
                if name not in available:
                    raise SystemExit(f"{path} does not contain indexed tensor {name}")
                view = source.get_slice(name)
                result[name] = (view.get_dtype(), tuple(int(v) for v in view.get_shape()))
    return result


def _component_stats(
    transformer_state: Mapping[str, torch.Tensor],
    t5_map: Mapping[str, Path],
    vae_map: Mapping[str, Path],
    stats: Mapping[str, np.ndarray],
    policy: dict,
    *,
    transformer_layers: int,
    t5_layers: int,
) -> dict[str, dict[str, int | str]]:
    transformer_descriptors = _transformer_descriptors(transformer_state, policy)
    descriptors: dict[str, list[tuple[str, Iterable[int]]]] = {
        "gwp": [(dtype, shape) for _, dtype, shape in transformer_descriptors],
        "stats": [("F32" if value.dtype == np.float32 else str(value.dtype), value.shape)
                  for value in stats.values()],
    }
    for component, weight_map in (("t5", t5_map), ("vae", vae_map)):
        inspected = _inspect_safetensors(weight_map)
        descriptors[component] = [inspected[name] for name in sorted(inspected)]

    expected_counts = component_expected_counts(
        transformer_layers, t5_layers, bool(policy.get("pack_self_qkv")))
    result = {}
    for component, items in descriptors.items():
        expected_dtype = policy["source"][component]
        bad = [dtype for dtype, _ in items if dtype != expected_dtype]
        if bad:
            raise SystemExit(
                f"{component} policy requires uniform source dtype {expected_dtype}; "
                f"found {sorted(set(bad))}"
            )
        if len(items) != expected_counts[component]:
            raise SystemExit(
                f"{component} has {len(items)} tensors; expected {expected_counts[component]}"
            )
        names = (
            [name for name, _, _ in transformer_descriptors] if component == "gwp" else
            list(stats) if component == "stats" else
            sorted(t5_map if component == "t5" else vae_map)
        )
        elements = sum(
            _output_element_count(component, name, shape)
            for name, (_, shape) in zip(names, items)
        )
        source_dtype = policy["source"][component]
        stored_dtype = policy["stored"][component]
        result[component] = {
            "source_dtype": source_dtype,
            "stored_dtype": stored_dtype,
            "tensor_count": len(items),
            "elements": elements,
            "source_bytes": elements * DTYPE_BYTES[source_dtype],
            "stored_bytes": elements * DTYPE_BYTES[stored_dtype],
        }
    return result


def _validate_t5(cfg: dict, names: set[str]) -> None:
    if cfg.get("model_type") != "umt5" or cfg.get("architectures") != ["UMT5EncoderModel"]:
        raise SystemExit("base model text_encoder is not UMT5EncoderModel")
    required = {"shared.weight", "encoder.final_layer_norm.weight"}
    for index in range(int(cfg["num_layers"])):
        prefix = f"encoder.block.{index}.layer"
        required.update(
            {
                f"{prefix}.0.SelfAttention.q.weight",
                f"{prefix}.0.SelfAttention.k.weight",
                f"{prefix}.0.SelfAttention.v.weight",
                f"{prefix}.0.SelfAttention.o.weight",
                f"{prefix}.0.SelfAttention.relative_attention_bias.weight",
                f"{prefix}.0.layer_norm.weight",
                f"{prefix}.1.DenseReluDense.wi_0.weight",
                f"{prefix}.1.DenseReluDense.wi_1.weight",
                f"{prefix}.1.DenseReluDense.wo.weight",
                f"{prefix}.1.layer_norm.weight",
            }
        )
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"UMT5 is missing {len(missing)} required tensor(s): {missing[:12]}")


def _validate_vae(cfg: dict, names: set[str]) -> None:
    if cfg.get("_class_name") != "AutoencoderKLWan":
        raise SystemExit("base model vae is not AutoencoderKLWan")
    required = {
        "encoder.conv_in.weight",
        "encoder.conv_in.bias",
        "encoder.conv_out.weight",
        "encoder.conv_out.bias",
        "quant_conv.weight",
        "quant_conv.bias",
    }
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"Wan VAE is missing required tensor(s): {missing}")


def _load_stats(path: Path, action_dim: int) -> dict[str, np.ndarray]:
    root = _load_json(path)
    stats = root.get("norm_stats", root)
    result: dict[str, np.ndarray] = {}
    for source_name, output_name in (("observation.state", "state"), ("action", "action")):
        group = stats.get(source_name)
        if not isinstance(group, dict):
            raise SystemExit(f"normalization JSON has no {source_name!r} group")
        for stat_name in DEFAULT_STATS_KEYS:
            value = np.asarray(group.get(stat_name), dtype=np.float32)
            if value.ndim != 1 or value.size < action_dim:
                raise SystemExit(
                    f"{source_name}.{stat_name} has shape {value.shape}; "
                    f"expected at least {action_dim} values"
                )
            value = np.ascontiguousarray(value[:action_dim])
            if not np.all(np.isfinite(value)):
                raise SystemExit(f"{source_name}.{stat_name} contains NaN/Inf")
            result[f"{output_name}_{stat_name}"] = value
    return result


def _add_metadata(
    writer: gguf.GGUFWriter,
    transformer: dict,
    t5: dict,
    vae: dict,
    *,
    image_height: int,
    image_width: int,
    action_chunk: int,
    inference_steps: int,
    policy: dict | None = None,
    component_stats: Mapping[str, Mapping[str, int | str]] | None = None,
) -> None:
    policy = policy or POLICIES["source-f32"]
    hidden = int(transformer["num_attention_heads"]) * int(transformer["attention_head_dim"])
    writer.add_name("GigaWorld Policy 0.5 MoT")
    writer.add_string(KV("architecture"), ARCH)
    writer.add_string(KV("transformer_class"), transformer["_class_name"])
    add_policy_metadata(writer, policy, component_stats)
    for key, value in (
        ("hidden", hidden),
        ("n_layers", transformer["num_layers"]),
        ("n_heads", transformer["num_attention_heads"]),
        ("head_dim", transformer["attention_head_dim"]),
        ("ffn_dim", transformer["ffn_dim"]),
        ("action_hidden", transformer["action_expert_dim"]),
        ("action_ffn_dim", transformer["action_ffn_dim"]),
        ("action_dim", transformer["in_action_channels"]),
        ("real_state_dim", 14),
        ("real_action_dim", 14),
        ("num_embodiments", transformer["num_embodiments"]),
        ("embodiment_id", 0),
        ("in_channels", transformer["in_channels"]),
        ("text_dim", transformer["text_dim"]),
        ("time_freq_dim", transformer["freq_dim"]),
        ("rope_max_seq_len", transformer["rope_max_seq_len"]),
        ("image_height", image_height),
        ("image_width", image_width),
        ("num_views", 3),
        ("action_chunk", action_chunk),
        ("inference_steps", inference_steps),
        ("train_timesteps", 1000),
    ):
        writer.add_uint32(KV(key), int(value))
    writer.add_array(KV("patch_size"), [int(v) for v in transformer["patch_size"]])
    writer.add_float32(KV("norm_eps"), float(transformer["eps"]))
    writer.add_float32(KV("flow_shift"), 5.0)
    writer.add_string(KV("norm_mode"), "quantiles")
    writer.add_bool(KV("vae_single_frame_specialized"), True)

    for key, value in (
        ("t5_vocab_size", t5["vocab_size"]),
        ("t5_hidden", t5["d_model"]),
        ("t5_ffn_dim", t5["d_ff"]),
        ("t5_head_dim", t5["d_kv"]),
        ("t5_heads", t5["num_heads"]),
        ("t5_layers", t5["num_layers"]),
        ("t5_max_length", 512),
        ("t5_relative_buckets", t5["relative_attention_num_buckets"]),
        ("t5_relative_max_distance", t5["relative_attention_max_distance"]),
        ("t5_pad_token_id", t5["pad_token_id"]),
        ("vae_base_dim", vae["base_dim"]),
        ("vae_z_dim", vae["z_dim"]),
        ("vae_patch_size", vae["patch_size"]),
        ("vae_spatial_scale", vae["scale_factor_spatial"]),
        ("vae_temporal_scale", vae["scale_factor_temporal"]),
    ):
        writer.add_uint32(KV(key), int(value))
    writer.add_float32(KV("t5_norm_eps"), float(t5["layer_norm_epsilon"]))
    writer.add_array(KV("vae_dim_mult"), [int(v) for v in vae["dim_mult"]])
    writer.add_array(KV("vae_latents_mean"), [float(v) for v in vae["latents_mean"]])
    writer.add_array(KV("vae_latents_std"), [float(v) for v in vae["latents_std"]])


def _print_summary(
    transformer_state: Mapping[str, torch.Tensor],
    t5_names: Iterable[str],
    vae_names: Iterable[str],
) -> None:
    print(
        "validated components: "
        f"gwp={len(transformer_state)} tensors, "
        f"t5={len(tuple(t5_names))} tensors, "
        f"vae={len(tuple(vae_names))} tensors"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="GWP transformer_ema directory")
    parser.add_argument("--base-model", type=Path, required=True, help="Wan2.2 TI2V diffusers model directory")
    parser.add_argument("--norm-stats", type=Path, required=True, help="normalization-statistics JSON")
    parser.add_argument("--out", type=Path, required=True, help="output GGUF")
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--action-chunk", type=int, default=48)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument(
        "--weight-policy", choices=tuple(POLICIES), required=True,
        help="explicit component storage policy; never inferred from --out",
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="conversion manifest (default: <out>.manifest.json)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate inputs/schema without writing")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base_model = args.base_model.resolve()
    out = args.out.resolve()
    manifest = (args.manifest or Path(str(out) + ".manifest.json")).resolve()
    policy = _policy(args.weight_policy)
    if args.image_height <= 0 or args.image_width <= 0:
        raise SystemExit("image dimensions must be positive")
    if args.action_chunk <= 0 or args.inference_steps <= 0:
        raise SystemExit("action chunk and inference steps must be positive")

    transformer_cfg, transformer_state = _load_transformer(checkpoint)
    _validate_transformer(transformer_cfg, transformer_state)
    t5_dir = base_model / "text_encoder"
    vae_dir = base_model / "vae"
    t5_cfg = _load_json(t5_dir / "config.json")
    vae_cfg = _load_json(vae_dir / "config.json")
    _, t5_map = _safetensor_index(t5_dir)
    _, vae_map = _safetensor_index(vae_dir)
    _validate_t5(t5_cfg, set(t5_map))
    _validate_vae(vae_cfg, set(vae_map))
    # Action-only inference encodes the reference image but never decodes video.
    vae_map = {
        name: path
        for name, path in vae_map.items()
        if (name.startswith("encoder.") or name.startswith("quant_conv."))
        and ".time_conv." not in name
    }
    stats = _load_stats(args.norm_stats.resolve(), int(transformer_cfg["in_action_channels"]))
    transformer_names = [name for name, _, _ in _transformer_descriptors(
        transformer_state, policy)]
    _validate_compact_names({"gwp": transformer_names, "t5": t5_map, "vae": vae_map})
    component_stats = _component_stats(
        transformer_state, t5_map, vae_map, stats, policy,
        transformer_layers=int(transformer_cfg["num_layers"]),
        t5_layers=int(t5_cfg["num_layers"]),
    )
    _print_summary(transformer_state, t5_map, vae_map)
    print(f"conversion policy: {policy['id']}")
    for component in ("gwp", "t5", "vae", "stats"):
        values = component_stats[component]
        print(
            f"  {component}: {values['tensor_count']} tensors, "
            f"{values['source_dtype']}->{values['stored_dtype']}, "
            f"{values['stored_bytes']} stored bytes"
        )

    if args.dry_run:
        print("dry-run complete; no GGUF was written")
        return 0

    if out.exists():
        raise SystemExit(f"refusing to overwrite existing GGUF: {out}")
    if manifest.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {manifest}")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.incomplete.{os.getpid()}")
    print(f"writing {policy['id']} GGUF: {out}")
    writer = gguf.GGUFWriter(str(temporary), arch=ARCH)
    _add_metadata(
        writer,
        transformer_cfg,
        t5_cfg,
        vae_cfg,
        image_height=args.image_height,
        image_width=args.image_width,
        action_chunk=args.action_chunk,
        inference_steps=args.inference_steps,
        policy=policy,
        component_stats=component_stats,
    )
    for name, tensor in _iter_transformer_tensors(transformer_state, policy):
        if name == "patch_embedding.weight":
            if tensor.ndim != 5 or tensor.shape[2] != 1:
                raise SystemExit(f"unsupported GWP patch kernel shape {tuple(tensor.shape)}")
            tensor = tensor.squeeze(2)
        _add_tensor(writer, _compact_name("gwp", name), _convert_tensor("gwp", tensor, policy))
    for name, tensor in _iter_safetensors(t5_map):
        _add_tensor(writer, _compact_name("t5", name), _convert_tensor("t5", tensor, policy))
    for name, tensor in _iter_safetensors(vae_map):
        if tensor.ndim == 5:
            # WanCausalConv3d left-pads time. For the first and only input frame,
            # only the final temporal kernel plane multiplies non-zero data.
            tensor = tensor[:, :, -1, :, :]
        _add_tensor(writer, _compact_name("vae", name), _convert_tensor("vae", tensor, policy))
    for name, values in sorted(stats.items()):
        writer.add_tensor(name, values, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    try:
        publish_no_overwrite(temporary, out)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(str(exc)) from exc
    result = {
        "format": "gwp05-conversion-manifest-v1",
        "converter_revision": CONVERTER_REVISION,
        "conversion_policy": policy["id"],
        "inputs": {
            "checkpoint": str(checkpoint),
            "base_model": str(base_model),
            "normalization": str(args.norm_stats.resolve()),
        },
        "components": component_stats,
        "output": {
            "path": str(out),
            "size_bytes": out.stat().st_size,
            "sha256": sha256_file(out),
        },
        "converter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "contract_helper": {
            "path": str(Path(gguf_contract.__file__).resolve()),
            "sha256": sha256_file(Path(gguf_contract.__file__).resolve()),
        },
    }
    try:
        write_json_no_overwrite(manifest, result)
    except FileExistsError as exc:
        raise SystemExit(f"refusing to overwrite existing manifest: {manifest}") from exc
    print(f"done: {out} ({out.stat().st_size / (1024 ** 3):.2f} GiB)")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("conversion interrupted", file=sys.stderr)
        raise SystemExit(130)
