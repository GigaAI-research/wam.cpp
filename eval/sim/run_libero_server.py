"""Run wam.cpp with the frozen LIBERO end-effector environment contract."""

from wam.adapters.libero import CONTRACT
from wam.serving.cli import serve_main


if __name__ == "__main__":
    serve_main(CONTRACT)
