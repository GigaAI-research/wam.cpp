from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .manifest import canonical_json


def load_jsonl(path):
    source = Path(path)
    if not source.exists():
        return []
    with source.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_write(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path, value):
    _atomic_write(path, canonical_json(value))


def write_jsonl(path, values):
    payload = b"".join(
        (json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n").encode()
        for value in values)
    _atomic_write(path, payload)


class ResultWriter:
    def __init__(self, path):
        self.path = Path(path)

    def append(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def load(self):
        return load_jsonl(self.path)
