"""Run wam.cpp with the frozen RoboTwin dual-arm environment contract."""

from wam.adapters.robotwin import CONTRACT
from wam.serving.cli import serve_main


if __name__ == "__main__":
    serve_main(CONTRACT)
