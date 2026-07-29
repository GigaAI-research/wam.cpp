"""Compatibility facade over the formal wam Python SDK."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from wam import Model, RuntimeConfig, SessionConfig, WamError  # noqa: E402

NativeError = WamError


class NativeModel(Model):
    def __init__(self, library, artifact, backend="cuda", precision="bf16",
                 device=0, prompt_cache_capacity=4,
                 language_mode="automatic"):
        super().__init__(
            artifact, library=library,
            config=RuntimeConfig(
                backend=backend, precision=precision, device=device,
                prompt_cache_capacity=prompt_cache_capacity,
                language_mode=language_mode))

    def create_session(self, random_seed=0, enable_prefix_cache=True):
        session = super().create_session(SessionConfig(
            enable_prefix_cache=enable_prefix_cache,
            random_seed=random_seed))
        return NativeSession(session, self.metadata)


class NativeSession:
    def __init__(self, session, metadata):
        self._session = session
        self._metadata = metadata

    def predict(self, images, state, language_values, attention_mask,
                action_noise=None):
        mode = self._metadata["language_mode"]
        if mode == "tokens":
            result = self._session.predict(
                images, state, token_ids=language_values,
                attention_mask=attention_mask, action_noise=action_noise)
        elif mode == "external_embedding":
            result = self._session.predict(
                images, state, embedding=language_values,
                embedding_attention_mask=attention_mask,
                action_noise=action_noise)
        else:
            raise ValueError(f"unsupported native language mode: {mode}")
        return result.action, result.stats

    def reset(self):
        self._session.reset()

    def close(self):
        self._session.close()
