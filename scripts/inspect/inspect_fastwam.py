#!/usr/bin/env python3
"""Inspect the FastWAM 0.5 action-only PolicySpec GGUF contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gguf

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.fastwam_gguf import (
    CONVERSION_POLICY,
    CONVERTER_REVISION,
    EXPECTED_COMPONENT_COUNTS,
)


class ContractError(ValueError):
    pass


def _field(reader: Any, name: str) -> Any:
    value = reader.get_field(name)
    if value is None:
        raise ContractError(f"missing metadata: {name}")
    return value.contents()


def inspect(path: Path) -> dict[str, Any]:
    reader = gguf.GGUFReader(path, "r")
    if _field(reader, "general.architecture") != "fastwam":
        raise ContractError("artifact architecture is not fastwam")
    if _field(reader, "fastwam.conversion_policy") != CONVERSION_POLICY or \
       _field(reader, "fastwam.converter_revision") != CONVERTER_REVISION:
        raise ContractError("unsupported FastWAM converter revision")
    if _field(reader, "fastwam.variant") != "uncond_action_only":
        raise ContractError("unsupported FastWAM variant")
    if int(_field(reader, "wam.artifact_schema_version")) != 2:
        raise ContractError("unsupported PolicySpec schema")

    geometry_keys = (
        "image_height", "image_width", "num_cameras", "action_dim",
        "proprio_dim", "action_horizon", "inference_steps",
        "latent_channels", "spatial_downsample", "temporal_downsample",
        "context_len", "text_dim", "video_hidden_dim", "action_hidden_dim",
        "num_layers", "num_heads", "attn_head_dim",
    )
    geometry = {
        key: int(_field(reader, f"fastwam.{key}"))
        for key in geometry_keys
    }
    if any(value <= 0 for value in geometry.values()):
        raise ContractError("FastWAM geometry contains a zero dimension")
    if geometry["video_hidden_dim"] != \
       geometry["num_heads"] * geometry["attn_head_dim"]:
        raise ContractError("video attention dimensions are inconsistent")
    norm_eps = float(_field(reader, "fastwam.norm_eps"))
    if norm_eps <= 0.0:
        raise ContractError("FastWAM model norm epsilon must be positive")

    components = {name: 0 for name in EXPECTED_COMPONENT_COUNTS}
    for tensor in reader.tensors:
        if tensor.name.startswith("wam.norm."):
            component = "stats"
            expected_dtype = "F32"
        elif tensor.name.startswith("fastwam."):
            parts = tensor.name.split(".")
            component = parts[1] if len(parts) >= 3 else ""
            expected_dtype = "BF16"
        else:
            raise ContractError(
                f"tensor is outside the FastWAM ABI: {tensor.name}")
        if component not in components:
            raise ContractError(f"unknown FastWAM component: {tensor.name}")
        dtype = tensor.tensor_type.name
        if dtype != expected_dtype:
            raise ContractError(
                f"{tensor.name}: dtype {dtype}, expected {expected_dtype}")
        components[component] += 1
    if components != EXPECTED_COMPONENT_COUNTS:
        raise ContractError(
            f"component counts differ: {components}")

    return {
        "architecture": "fastwam",
        "conversion_policy": CONVERSION_POLICY,
        "profile": _field(reader, "wam.policy.profile"),
        "geometry": geometry,
        "norm_eps": norm_eps,
        "components": components,
        "tensor_count": len(reader.tensors),
        "artifact_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    try:
        report = {"valid": True, **inspect(args.artifact.resolve())}
    except (ContractError, KeyError, OSError, TypeError, ValueError) as exc:
        report = {"valid": False, "error": str(exc)}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
