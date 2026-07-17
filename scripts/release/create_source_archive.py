#!/usr/bin/env python3
"""Create a clean wam.cpp runtime source archive."""

from __future__ import annotations

import argparse
import gzip
import os
import subprocess
import tarfile
from pathlib import Path


ROOT_EXCLUDES = ("build", "build-", "cmake-build-")
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
def excluded(relative: Path, output: Path, source: Path) -> bool:
    if (source / relative).resolve() == output:
        return True
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return True
    if relative.parts and relative.parts[0].startswith(ROOT_EXCLUDES):
        return True
    return False


def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="wam.cpp-0.1.0")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "CMakeLists.txt").is_file():
        parser.error(f"not a wam.cpp source tree: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite source archive: {output}")

    subprocess.run(
        [str(source / "scripts/release/check_source_tree.py"), str(source)],
        check=True,
    )

    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(source.rglob("*")):
                    relative = path.relative_to(source)
                    if excluded(relative, output, source):
                        continue
                    if not (path.is_file() or path.is_symlink()):
                        continue
                    archive.add(
                        path,
                        arcname=str(Path(args.prefix) / relative),
                        recursive=False,
                        filter=normalized,
                    )

    os.chmod(output, 0o644)
    print(f"source archive: {output}")
    print("archive profile: runtime source only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
