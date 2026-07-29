"""Compatibility facade for the formal transport-neutral serving stack."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from wam.adapters import EnvironmentContract, check_environment  # noqa: E402,F401
from wam.serving import ServiceCore, WebSocketTransport  # noqa: E402
from wam.serving.cli import serve_main  # noqa: E402,F401


class WamServer(WebSocketTransport):
    def __init__(self, model, language_provider, descriptor, contract, host,
                 port, random_seed=0, max_message_bytes=64 << 20):
        service = ServiceCore(
            model, language_provider, descriptor, contract, random_seed)
        super().__init__(service, host, port, max_message_bytes)
        self.types = service.types


def main(argv=None):
    serve_main(argv=argv)
