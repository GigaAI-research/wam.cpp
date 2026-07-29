"""Run wam.cpp with the frozen LIBERO end-effector environment contract."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.server import serve_main
from wam.adapters.libero import CONTRACT


if __name__ == "__main__":
    serve_main(CONTRACT)
