#!/usr/bin/env python3
"""Validate a GWP-0.5 GGUF against the v0.1 runtime artifact contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import gguf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "missing Python package 'gguf'; install gguf or add llama.cpp/gguf-py to PYTHONPATH"
    ) from exc

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.gwp05_gguf import (  # noqa: E402
    CONVERTER_REVISION,
    DTYPE_BYTES,
    POLICIES_BY_ID,
    STATISTIC_NAMES,
    component_expected_counts,
    sha256_file,
    write_json_no_overwrite,
)


class ContractError(ValueError):
    pass


def _field(reader: gguf.GGUFReader, name: str) -> Any:
    field = reader.get_field(name)
    if field is None:
        raise ContractError(f"missing metadata: {name}")
    return field.contents()


def _optional(reader: gguf.GGUFReader, name: str, default: Any = None) -> Any:
    field = reader.get_field(name)
    return default if field is None else field.contents()


def _shape(tensors: dict[str, Any], name: str, expected: tuple[int, ...]) -> None:
    tensor = tensors.get(name)
    if tensor is None:
        raise ContractError(f"missing tensor: {name}")
    actual = tuple(int(value) for value in tensor.shape)
    canonical = list(expected)
    while len(canonical) > 1 and canonical[-1] == 1:
        canonical.pop()
    if actual != tuple(canonical):
        raise ContractError(f"{name}: shape {actual}, expected {tuple(canonical)}")


def _policy(reader: gguf.GGUFReader) -> tuple[str, dict[str, Any], bool]:
    conversion = _optional(reader, "gwp05.conversion_policy", "")
    if not conversion:
        legacy = _optional(reader, "gwp05.weight_policy", "source")
        if legacy != "source":
            raise ContractError(f"unsupported legacy weight policy: {legacy}")
        source = POLICIES_BY_ID["source-f32-v1"]
        return "legacy-source", source, False
    definition = POLICIES_BY_ID.get(conversion)
    if definition is None:
        raise ContractError(f"unsupported conversion policy: {conversion}")
    if _field(reader, "gwp05.weight_policy") != conversion:
        raise ContractError("gwp05.weight_policy differs from conversion_policy")
    if _field(reader, "gwp05.converter_revision") != CONVERTER_REVISION:
        raise ContractError("unsupported converter revision")
    return conversion, definition, True


def validate_reader(reader: gguf.GGUFReader, expected_policy: str | None = None) -> dict[str, Any]:
    if _field(reader, "general.architecture") != "gwp05" or \
       _field(reader, "gwp05.architecture") != "gwp05":
        raise ContractError("artifact architecture is not gwp05")

    keys = (
        "hidden", "n_layers", "n_heads", "head_dim", "ffn_dim",
        "action_hidden", "action_ffn_dim", "action_dim", "real_state_dim",
        "real_action_dim", "num_embodiments", "embodiment_id", "image_height",
        "image_width", "num_views", "action_chunk", "inference_steps",
        "t5_vocab_size", "t5_hidden", "t5_ffn_dim", "t5_heads", "t5_head_dim",
        "t5_layers", "vae_z_dim",
    )
    geometry = {key: int(_field(reader, f"gwp05.{key}")) for key in keys}
    flow_shift = float(_field(reader, "gwp05.flow_shift"))
    norm_eps = float(_field(reader, "gwp05.norm_eps"))
    if any(value <= 0 for key, value in geometry.items() if key != "embodiment_id"):
        raise ContractError("geometry contains a zero dimension")
    if geometry["hidden"] != geometry["n_heads"] * geometry["head_dim"] or \
       geometry["t5_hidden"] != geometry["t5_heads"] * geometry["t5_head_dim"]:
        raise ContractError("attention dimensions are inconsistent")
    if geometry["real_state_dim"] > geometry["action_dim"] or \
       geometry["real_action_dim"] > geometry["action_dim"]:
        raise ContractError("real state/action dimensions exceed action_dim")
    if geometry["num_views"] != 3:
        raise ContractError("GWP-0.5 requires exactly three views")
    if geometry["embodiment_id"] >= geometry["num_embodiments"]:
        raise ContractError("embodiment_id is out of range")
    if geometry["image_height"] % 32 or geometry["image_width"] % 32:
        raise ContractError("image dimensions must be divisible by 32")
    if not (flow_shift > 0.0 and norm_eps > 0.0):
        raise ContractError("flow_shift and norm_eps must be positive")

    policy_id, definition, explicit = _policy(reader)
    if expected_policy is not None and policy_id != expected_policy:
        raise ContractError(f"policy {policy_id}, expected {expected_policy}")
    packed = bool(definition.get("pack_self_qkv"))

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ContractError("duplicate tensor names")
    _shape(tensors, "gwp.patch.weight",
           (2, 2, geometry["vae_z_dim"], geometry["hidden"]))
    _shape(tensors, "gwp.state.in.weight",
           (128, geometry["action_dim"], geometry["num_embodiments"]))
    _shape(tensors, "gwp.decode.out.weight",
           (geometry["action_dim"], 128, geometry["num_embodiments"]))
    _shape(tensors, "t5.token_embd.weight",
           (geometry["t5_hidden"], geometry["t5_vocab_size"]))
    _shape(tensors, "vae.enc.in.weight", (3, 3, 12, 160))
    _shape(tensors, "state_q01", (geometry["action_dim"],))
    _shape(tensors, "action_q99", (geometry["action_dim"],))
    projection = "qkv" if packed else "q"
    output = 3 * geometry["hidden"] if packed else geometry["hidden"]
    for layer in range(geometry["n_layers"]):
        base = f"gwp.blk.{layer}"
        _shape(tensors, f"{base}.a.sa.{projection}.weight",
               (geometry["action_hidden"], output))
        _shape(tensors, f"{base}.v.sa.{projection}.weight",
               (geometry["hidden"], output))

    components: dict[str, dict[str, Any]] = {
        name: {"source_dtype": definition["source"][name],
               "stored_dtype": definition["stored"][name],
               "tensor_count": 0, "elements": 0, "stored_bytes": 0}
        for name in ("gwp", "t5", "vae", "stats")
    }
    statistics = set(STATISTIC_NAMES)
    for tensor in reader.tensors:
        if tensor.name.startswith("gwp."):
            component = "gwp"
        elif tensor.name.startswith("t5."):
            component = "t5"
        elif tensor.name.startswith("vae."):
            component = "vae"
        elif tensor.name in statistics:
            component = "stats"
        else:
            raise ContractError(f"tensor is outside known components: {tensor.name}")
        dtype = tensor.tensor_type.name
        expected_dtype = components[component]["stored_dtype"]
        if dtype != expected_dtype:
            raise ContractError(f"{tensor.name}: dtype {dtype}, expected {expected_dtype}")
        values = components[component]
        values["tensor_count"] += 1
        values["elements"] += int(tensor.n_elements)
        values["stored_bytes"] += int(tensor.n_bytes)

    expected_counts = component_expected_counts(
        geometry["n_layers"], geometry["t5_layers"], packed)
    for name, values in components.items():
        values["source_bytes"] = values["elements"] * DTYPE_BYTES[values["source_dtype"]]
        if values["tensor_count"] != expected_counts[name]:
            raise ContractError(
                f"{name}: {values['tensor_count']} tensors, expected {expected_counts[name]}")
        if explicit:
            base = f"gwp05.{name}_"
            expected_metadata = {
                "source_dtype": values["source_dtype"],
                "stored_dtype": values["stored_dtype"],
                "tensor_count": values["tensor_count"],
                "elements": values["elements"],
                "source_bytes": values["source_bytes"],
                "stored_bytes": values["stored_bytes"],
            }
            for field, expected in expected_metadata.items():
                actual = _field(reader, base + field)
                if actual != expected:
                    raise ContractError(
                        f"{base + field}: metadata {actual!r}, computed {expected!r}")

    return {
        "architecture": "gwp05",
        "policy": policy_id,
        "explicit_policy": explicit,
        "geometry": geometry,
        "components": components,
        "tensor_count": len(reader.tensors),
    }


def inspect(path: Path, expected_policy: str | None = None,
            expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ContractError(f"artifact does not exist: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ContractError(f"SHA-256 {digest}, expected {expected_sha256.lower()}")
    reader = gguf.GGUFReader(path, "r")
    result = validate_reader(reader, expected_policy)
    result.update({"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expect-policy", choices=("legacy-source", *POLICIES_BY_ID))
    parser.add_argument("--expect-sha256")
    parser.add_argument("--report", type=Path, help="write JSON report without overwriting")
    args = parser.parse_args()
    try:
        result = inspect(args.artifact, args.expect_policy, args.expect_sha256)
        report = {"format": "gwp05-inspection-v1", "valid": True, **result}
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        report = {
            "format": "gwp05-inspection-v1", "valid": False,
            "path": str(args.artifact.resolve()), "error": str(exc),
        }
    if args.report:
        try:
            write_json_no_overwrite(args.report.resolve(), report)
        except FileExistsError:
            print(f"refusing to overwrite report: {args.report}", file=sys.stderr)
            return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
