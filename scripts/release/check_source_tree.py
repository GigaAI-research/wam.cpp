#!/usr/bin/env python3
"""Reject generated outputs, private paths, and large model files in a source tree."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BUILD_RE = re.compile(r"^(build|build-|cmake-build-)")
PRIVATE_MARKERS = (
    "/" + "testessfs",
    "/home/" + "yejun",
    "/home/" + "shareuser",
    "C:" + "\\Users\\",
)
MODEL_SUFFIXES = {
    ".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".bin", ".npz", ".npy"
}
GENERATED_SUFFIXES = {
    ".pyc", ".nsys-rep", ".qdrep", ".prof", ".dump", ".sqlite"
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    failures = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        # Build directories are repository-root outputs.  Matching every path
        # component also rejects legitimate files such as build-phase2.md.
        if relative.parts and BUILD_RE.match(relative.parts[0]):
            failures.append(f"generated build path: {relative}")
            continue
        if "__pycache__" in relative.parts:
            failures.append(f"Python cache: {relative}")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in MODEL_SUFFIXES | GENERATED_SUFFIXES:
            failures.append(f"generated/model file: {relative}")
        if path.stat().st_size <= 8 * 1024 * 1024:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in PRIVATE_MARKERS:
                if marker in text:
                    failures.append(f"private path marker {marker!r}: {relative}")
                    break
    if failures:
        print("source-tree audit failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("source-tree audit: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
