#!/usr/bin/env python3
"""Fast tests for server-owned language provider contracts."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import threading
from types import MethodType

import numpy as np

from common.language import (
    PreparedLanguage, WanUmt5EmbeddingProvider, create_language_provider)


def metadata():
    return {
        "language_mode": "external_embedding",
        "policy_spec": {"language": {
            "input_mode": "embedding",
            "prompt_template": "execute: {task}",
            "tokenizer_family": "umt5",
            "max_tokens": 128,
        }},
    }


def fake_provider():
    provider = object.__new__(WanUmt5EmbeddingProvider)
    provider.spec = metadata()["policy_spec"]["language"]
    provider.cache_capacity = 2
    provider.cache = OrderedDict()
    provider.lock = threading.Lock()
    provider.calls = []

    def encode(self, prompt):
        self.calls.append(prompt)
        marker = len(self.calls)
        return PreparedLanguage(
            np.full((128, 4096), marker, dtype=np.uint16),
            np.ones(128, dtype=np.int32),
            preprocess_milliseconds=0.5,
            model_milliseconds=2.0)

    provider._encode = MethodType(encode, provider)
    return provider


def run():
    provider = fake_provider()
    first = provider.prepare("pick up bowl")
    repeated = provider.prepare("pick up bowl")
    assert provider.calls == ["execute: pick up bowl"]
    assert first.values.shape == (128, 4096)
    assert first.values.dtype == np.uint16
    assert first.attention_mask.shape == (128,)
    assert not first.cache_hit and repeated.cache_hit
    assert repeated.model_milliseconds == 0.0
    assert np.array_equal(first.values, repeated.values)

    provider.prepare("task two")
    provider.prepare("task three")
    assert list(provider.cache) == ["execute: task two", "execute: task three"]
    provider.prepare("pick up bowl")
    assert provider.calls[-1] == "execute: pick up bowl"

    try:
        create_language_provider(metadata())
    except ValueError as error:
        message = str(error)
        assert "--tokenizer" in message
        assert "--text-encoder" in message
        assert "--language-python-root" in message
    else:
        raise AssertionError("external provider must reject missing deployment paths")


def main():
    argparse.ArgumentParser().parse_args()
    run()
    print("wam server-owned language provider: PASS")


if __name__ == "__main__":
    main()
