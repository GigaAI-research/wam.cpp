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
    EXPECTED_COMPONENT_COUNTS,
    SUPPORTED_CONVERTER_REVISIONS,
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
    revision = _field(reader, "fastwam.converter_revision")
    if _field(reader, "fastwam.conversion_policy") != CONVERSION_POLICY or \
       revision not in SUPPORTED_CONVERTER_REVISIONS:
        raise ContractError("unsupported FastWAM converter revision")
    if _field(reader, "fastwam.variant") != "uncond_action_only":
        raise ContractError("unsupported FastWAM variant")
    expected_schema = (
        2 if revision == "wam-0.5-fastwam-policy-spec-v2" else 3)
    if int(_field(reader, "wam.artifact_schema_version")) != expected_schema:
        raise ContractError("unsupported PolicySpec schema")

    geometry_keys = (
        "inference_steps",
        "latent_channels", "spatial_downsample", "temporal_downsample",
        "text_dim", "video_hidden_dim", "action_hidden_dim",
        "num_layers", "num_heads", "attn_head_dim",
    )
    geometry = {
        key: int(_field(reader, f"fastwam.{key}"))
        for key in geometry_keys
    }
    policy_geometry = {
        "image_height": int(_field(
            reader, "wam.input.image.composition.height")),
        "image_width": int(_field(
            reader, "wam.input.image.composition.width")),
        "num_cameras": len(_field(reader, "wam.input.image.roles")),
        "action_dim": int(_field(reader, "wam.output.action.model_dim")),
        "proprio_dim": int(_field(reader, "wam.input.state.model_dim")),
        "action_horizon": int(_field(reader, "wam.output.action.horizon")),
        "context_len": int(_field(reader, "wam.input.language.max_tokens")),
    }
    legacy_geometry = revision == "wam-0.5-fastwam-policy-spec-v2"
    for key, expected in policy_geometry.items():
        field = reader.get_field(f"fastwam.{key}")
        if field is None and legacy_geometry:
            raise ContractError(f"missing metadata: fastwam.{key}")
        if field is not None and int(field.contents()) != expected:
            raise ContractError(f"fastwam.{key} conflicts with PolicySpec")
        geometry[key] = expected
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
