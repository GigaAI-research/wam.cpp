"""Compatibility exports for the formal wam remote client."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from wam.remote import (  # noqa: E402,F401
    Client,
    RpcClient,
    decode_tensor,
    encode_tensor,
    load_types,
)
