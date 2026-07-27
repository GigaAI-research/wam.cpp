"""Server-owned language providers for raw-instruction RPC serving."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import numpy as np


@dataclass(frozen=True)
class PreparedLanguage:
    values: np.ndarray
    attention_mask: np.ndarray
    preprocess_milliseconds: float = 0.0
    model_milliseconds: float = 0.0
    cache_hit: bool = False


def _language_spec(metadata):
    spec = metadata["policy_spec"]["language"]
    if int(spec["max_tokens"]) <= 0:
        raise ValueError("PolicySpec language.max_tokens must be positive")
    return spec


def _prompt(spec, instruction):
    return spec["prompt_template"].replace("{task}", instruction)


def _apply_wan_prompt_padding(values, attention_mask):
    """Match FastWAM's prompt-embedding contract after UMT5 encoding."""
    values = np.asarray(values)
    attention_mask = np.asarray(attention_mask)
    if values.ndim != 2 or attention_mask.ndim != 1:
        raise ValueError("Wan prompt values/mask must be rank 2/rank 1")
    if values.shape[0] != attention_mask.shape[0]:
        raise ValueError("Wan prompt values and mask sequence lengths differ")
    if np.any((attention_mask != 0) & (attention_mask != 1)):
        raise ValueError("Wan prompt attention mask must be binary")
    if np.any(attention_mask[1:] > attention_mask[:-1]):
        raise ValueError("Wan prompt attention mask must use right padding")

    valid_tokens = int(np.sum(attention_mask, dtype=np.int64))
    padded_values = np.ascontiguousarray(values.copy())
    padded_values[valid_tokens:] = 0
    model_mask = np.ones(attention_mask.shape, dtype=np.int32)
    return padded_values, model_mask


class TokenLanguageProvider:
    def __init__(self, metadata, tokenizer_path):
        if metadata["language_mode"] != "tokens":
            raise ValueError("token provider requires token language mode")
        from transformers import AutoTokenizer
        self.spec = _language_spec(metadata)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False)
        family = self.spec["tokenizer_family"].lower()
        if family == "umt5" and "t5tokenizer" not in type(self.tokenizer).__name__.lower():
            raise ValueError(
                f"profile requires an UMT5/T5 tokenizer, got {type(self.tokenizer).__name__}")

    def prepare(self, instruction):
        start = time.perf_counter()
        encoded = self.tokenizer(
            _prompt(self.spec, instruction), padding="max_length",
            truncation=True, max_length=int(self.spec["max_tokens"]),
            return_attention_mask=True, add_special_tokens=True)
        elapsed = (time.perf_counter() - start) * 1000.0
        return PreparedLanguage(
            np.asarray(encoded["input_ids"], dtype=np.int32),
            np.asarray(encoded["attention_mask"], dtype=np.int32),
            preprocess_milliseconds=elapsed)


class WanUmt5EmbeddingProvider:
    """Exact Wan UMT5 provider used to build FastWAM cached embeddings."""

    def __init__(self, metadata, tokenizer_path, encoder_path, python_root,
                 device="cuda:0", cache_capacity=32):
        if metadata["language_mode"] != "external_embedding":
            raise ValueError("Wan UMT5 provider requires external_embedding mode")
        self.spec = _language_spec(metadata)
        if self.spec["input_mode"] != "embedding":
            raise ValueError("PolicySpec language.input_mode is not embedding")
        if self.spec["tokenizer_family"].lower() != "umt5":
            raise ValueError("Wan provider currently supports tokenizer_family=umt5 only")
        if int(self.spec["max_tokens"]) != 128:
            raise ValueError("FastWAM Wan UMT5 provider requires max_tokens=128")
        root = str(Path(python_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        import torch
        from safetensors.torch import load_file
        from fastwam.models.wan22.wan_video_text_encoder import (
            HuggingfaceTokenizer, WanTextEncoder)

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = HuggingfaceTokenizer(
            str(Path(tokenizer_path).resolve()), seq_len=128,
            clean="whitespace", local_files_only=True)
        with torch.device("meta"):
            encoder = WanTextEncoder()
        state = load_file(str(Path(encoder_path).resolve()), device="cpu")
        encoder.load_state_dict(state, strict=True, assign=True)
        self.encoder = encoder.to(device=self.device, dtype=torch.bfloat16).eval()
        self.cache_capacity = max(0, int(cache_capacity))
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _encode(self, prompt):
        torch = self.torch
        tokenize_start = time.perf_counter()
        ids, mask = self.tokenizer(
            prompt, return_mask=True, add_special_tokens=True)
        preprocess_ms = (time.perf_counter() - tokenize_start) * 1000.0
        ids = ids.to(self.device)
        device_mask = mask.to(device=self.device, dtype=torch.bool)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_start = time.perf_counter()
        with torch.inference_mode():
            context = self.encoder(ids, device_mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        model_ms = (time.perf_counter() - model_start) * 1000.0
        values = (context[0].detach().to(device="cpu", dtype=torch.bfloat16)
                  .contiguous().view(torch.uint16).numpy().copy())
        attention_mask = mask[0].to(dtype=torch.int32).contiguous().numpy().copy()
        values, attention_mask = _apply_wan_prompt_padding(values, attention_mask)
        return PreparedLanguage(values, attention_mask, preprocess_ms, model_ms)

    def prepare(self, instruction):
        prompt = _prompt(self.spec, instruction)
        with self.lock:
            cached = self.cache.get(prompt)
            if cached is not None:
                self.cache.move_to_end(prompt)
                return PreparedLanguage(cached[0], cached[1], cache_hit=True)
            result = self._encode(prompt)
            if self.cache_capacity:
                self.cache[prompt] = (result.values, result.attention_mask)
                self.cache.move_to_end(prompt)
                while len(self.cache) > self.cache_capacity:
                    self.cache.popitem(last=False)
            return result


def create_language_provider(metadata, *, tokenizer_path=None,
                             encoder_path=None, python_root=None,
                             device="cuda:0", cache_capacity=32):
    mode = metadata["language_mode"]
    if mode == "tokens":
        if tokenizer_path is None:
            raise ValueError("--tokenizer is required for token language mode")
        return TokenLanguageProvider(metadata, tokenizer_path)
    if mode == "external_embedding":
        missing = [name for name, value in (
            ("--tokenizer", tokenizer_path), ("--text-encoder", encoder_path),
            ("--language-python-root", python_root)) if value is None]
        if missing:
            raise ValueError("external embedding mode requires " + ", ".join(missing))
        return WanUmt5EmbeddingProvider(
            metadata, tokenizer_path, encoder_path, python_root,
            device=device, cache_capacity=cache_capacity)
    raise ValueError(f"raw-instruction serving does not support language mode {mode!r}")
