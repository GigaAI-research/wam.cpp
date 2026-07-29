"""Compatibility exports for the formal wam language providers."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from wam.language import (  # noqa: E402,F401
    PreparedLanguage,
    TokenLanguageProvider,
    WanUmt5EmbeddingProvider,
    _apply_wan_prompt_padding,
    create_language_provider,
)
