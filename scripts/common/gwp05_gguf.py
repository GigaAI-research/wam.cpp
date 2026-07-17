"""Shared GWP-0.5 GGUF policy and artifact I/O helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ARCH = "gwp05"
CONVERTER_REVISION = "gwp05-native-bf16-v1"
DTYPE_BYTES = {"F32": 4, "BF16": 2}
STATISTIC_NAMES = (
    "state_mean", "state_std", "state_q01", "state_q99",
    "action_mean", "action_std", "action_q01", "action_q99",
)
POLICIES: dict[str, dict[str, Any]] = {
    "source-f32": {
        "id": "source-f32-v1",
        "source": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
        "stored": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
    },
    "mot-bf16": {
        "id": "mot-bf16-v1",
        "source": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
        "stored": {"gwp": "BF16", "t5": "BF16", "vae": "F32", "stats": "F32"},
    },
    "mot-vae-bf16": {
        "id": "mot-vae-bf16-v1",
        "source": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
        "stored": {"gwp": "BF16", "t5": "BF16", "vae": "BF16", "stats": "F32"},
    },
    "mot-vae-bf16-qkv": {
        "id": "mot-vae-bf16-qkv-v1",
        "source": {"gwp": "F32", "t5": "BF16", "vae": "F32", "stats": "F32"},
        "stored": {"gwp": "BF16", "t5": "BF16", "vae": "BF16", "stats": "F32"},
        "pack_self_qkv": True,
    },
}
POLICIES_BY_ID = {value["id"]: value for value in POLICIES.values()}


def kv(name: str) -> str:
    return f"{ARCH}.{name}"


def policy(name: str) -> dict[str, Any]:
    try:
        return POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown conversion policy {name!r}") from exc


def component_expected_counts(layers: int, t5_layers: int, packed: bool) -> dict[str, int]:
    return {
        "gwp": 44 + (46 if packed else 54) * layers,
        "t5": 2 + 10 * t5_layers,
        "vae": 82,
        "stats": 8,
    }


def add_policy_metadata(writer: Any, definition: Mapping[str, Any],
                        component_stats: Mapping[str, Mapping[str, Any]] | None) -> None:
    """Write the explicit policy and component telemetry shared with inspection."""
    writer.add_string(kv("weight_policy"), definition["id"])
    writer.add_string(kv("conversion_policy"), definition["id"])
    writer.add_string(kv("converter_revision"), CONVERTER_REVISION)
    if component_stats is None:
        return
    for component in ("gwp", "t5", "vae", "stats"):
        values = component_stats[component]
        writer.add_string(kv(f"{component}_source_dtype"), str(values["source_dtype"]))
        writer.add_string(kv(f"{component}_stored_dtype"), str(values["stored_dtype"]))
        for field in ("tensor_count", "elements", "source_bytes", "stored_bytes"):
            writer.add_uint64(kv(f"{component}_{field}"), int(values[field]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_no_overwrite(temporary: Path, destination: Path) -> None:
    """Atomically publish a completed file while refusing replacement."""
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing file: {destination}") from exc
    temporary.unlink()


def write_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
