"""Run wam.cpp with the frozen LIBERO-X end-effector contract."""

from wam.adapters.liberox import CONTRACT
from wam.serving.cli import serve_main


if __name__ == "__main__":
    serve_main(CONTRACT)
