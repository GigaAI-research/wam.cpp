from __future__ import annotations

import json
from pathlib import Path


def canonical_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) +
            "\n").encode()


def load_manifest(path, *, expected_format=None):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load evaluation manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("evaluation manifest must be a JSON object")
    if expected_format is not None and value.get("format") != expected_format:
        raise ValueError(f"manifest format must be {expected_format}")
    return value


def write_manifest(path, value):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json(value))
