#!/usr/bin/env python3
"""Forward directly to the adapter installed by wam.cpp."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


ADAPTER_RELATIVE = Path("share/wam/integrations/giga-world-policy/inference_openloop_wam_cpp.py")


def candidates() -> list[Path]:
    result = []
    explicit = os.environ.get("WAM_GWP_INSTALLED_ADAPTER")
    if explicit:
        result.append(Path(explicit))
    prefix = os.environ.get("WAM_INSTALL_PREFIX")
    if prefix:
        result.append(Path(prefix) / ADAPTER_RELATIVE)
    runner = shutil.which("run_inference_openloop_wam-cpp.sh")
    if runner:
        result.append(Path(runner).resolve().parent.parent / ADAPTER_RELATIVE)
    result.append(Path(sys.prefix) / ADAPTER_RELATIVE)
    return result


def main() -> int:
    current = Path(__file__).resolve()
    adapter = next(
        (candidate.resolve() for candidate in candidates()
         if candidate.is_file() and candidate.resolve() != current),
        None,
    )
    if adapter is None:
        raise SystemExit(
            "installed wam.cpp adapter not found; set WAM_INSTALL_PREFIX or "
            "WAM_GWP_INSTALLED_ADAPTER"
        )

    arguments = sys.argv[1:]
    has_upstream_root = any(
        value == "--upstream-root" or value.startswith("--upstream-root=")
        for value in arguments
    )
    if not has_upstream_root:
        arguments = ["--upstream-root", str(current.parents[1]), *arguments]
    os.execv(sys.executable, [sys.executable, str(adapter), *arguments])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
