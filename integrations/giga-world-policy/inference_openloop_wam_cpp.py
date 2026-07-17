#!/usr/bin/env python3
"""GWP-0.5 observation adapter for the in-process wam.cpp RPC bridge."""

from __future__ import annotations

import hashlib
import argparse
import importlib
import json
import ipaddress
import os
import math
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from wam_proto import load_types
from wam_rpc import (
    Backend,
    LanguageEncoderPolicy,
    ModelOptions,
    Precision,
    WamRpcError,
    WamRpcService,
)


CAMERAS = (
    ("observation.images.cam_high", "camera_high"),
    ("observation.images.cam_left_wrist", "camera_left_wrist"),
    ("observation.images.cam_right_wrist", "camera_right_wrist"),
)
ACTION_SHAPE = (48, 14)
NOISE_SHAPE = (48, 32)
EMBEDDING_SHAPE = (64, 4096)
STATE_SHAPES = ((14,), (1, 14))
INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1
UPSTREAM_COMMIT = "5d55073a6508de7354c83679d9028f4010ff6cb2"
UPSTREAM_SOCKET_HASHES = {
    "giga_models/sockets/server.py":
        "d2a2915f0cb7466a26e62e433543e1b2fc61723dbc0026327ecffcc9fc29f801",
    "giga_models/sockets/client.py":
        "4812b0aee9072099e152b20a80de7ebebe7a7d787b4cf4882fd6847590b95b53",
}


@dataclass(frozen=True)
class LoadedEmbedding:
    values: np.ndarray
    attention_mask: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class AdapterTiming:
    mapping_milliseconds: float
    rpc_milliseconds: float
    action_parse_milliseconds: float
    total_milliseconds: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value, field: str) -> tuple[np.ndarray, bool]:
    if isinstance(value, np.ndarray):
        return value, False
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy(), True
    raise TypeError(f"{field} must be a NumPy array or torch.Tensor")


def _normalize_embedding(value, field: str, *, truncate: bool = False) \
        -> tuple[np.ndarray, np.ndarray]:
    array, _ = _numpy(value, field)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"{field} batch dimension must be 1, got {array.shape}")
        array = array[0]
    if array.ndim != 2 or array.shape[1] != EMBEDDING_SHAPE[1]:
        raise ValueError(f"{field} must have shape [T,4096] or [1,T,4096]")
    if array.shape[0] == 0:
        raise ValueError(f"{field} must contain at least one token")
    if array.shape[0] > EMBEDDING_SHAPE[0] and not truncate:
        raise ValueError(f"{field} must contain at most 64 tokens")
    tokens = min(array.shape[0], EMBEDDING_SHAPE[0])
    source = np.asarray(array[:tokens], dtype=np.float32)
    if tokens == EMBEDDING_SHAPE[0]:
        result = np.ascontiguousarray(source)
    else:
        result = np.zeros(EMBEDDING_SHAPE, dtype=np.float32)
        result[:tokens] = source
    if not np.isfinite(result).all():
        raise ValueError(f"{field} contains NaN or Inf")
    mask = np.zeros(EMBEDDING_SHAPE[0], dtype=np.int32)
    mask[:tokens] = 1
    return np.ascontiguousarray(result), mask


def _extract_embedding(value):
    if not isinstance(value, Mapping):
        return value
    if "t5_embedding" in value:
        return value["t5_embedding"]
    if "prompt_embeds" in value:
        return value["prompt_embeds"]
    condition = value.get("condition_dict")
    if isinstance(condition, Mapping) and "prompt_embeds" in condition:
        return condition["prompt_embeds"]
    if not value:
        raise ValueError("fixed T5 file contains an empty mapping")
    return next(iter(value.values()))


def load_fixed_prompt_embedding(path: Path, *, trusted: bool) -> LoadedEmbedding:
    """Load an explicitly trusted upstream .pt embedding on CPU."""
    path = Path(path)
    if not trusted:
        raise ValueError(
            "loading a .pt embedding requires trusted=True because torch.load uses pickle"
        )
    if not path.is_file():
        raise FileNotFoundError(f"fixed T5 embedding not found: {path}")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("loading a .pt embedding requires torch") from error
    value = torch.load(path, map_location="cpu", weights_only=False)
    embedding, mask = _normalize_embedding(
        _extract_embedding(value), "fixed prompt embedding", truncate=True
    )
    return LoadedEmbedding(embedding, mask, _sha256(path))


def load_fixed_tokens(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    root = json.loads(Path(path).read_text())
    if not isinstance(root, dict):
        raise ValueError("fixed token JSON must be an object")
    return _tokens(root.get("token_ids"), root.get("attention_mask"), "fixed tokens")


def load_fixed_noise(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"fixed noise not found: {path}")
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = np.fromfile(path, dtype="<f4")
    return _noise(value, "fixed noise", allow_flat=True)


def _tokens(token_ids, attention_mask, field: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if token_ids is None:
        raise ValueError(f"{field} requires token_ids")
    ids = _integer_array(token_ids, f"{field}.token_ids")
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if ids.ndim != 1 or ids.size == 0 or ids.size > EMBEDDING_SHAPE[0]:
        raise ValueError(f"{field} token_ids must have shape [1..64]")
    if attention_mask is None:
        mask = np.ones(ids.size, dtype=np.int64)
    else:
        mask = _integer_array(attention_mask, f"{field}.attention_mask")
        if mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask[0]
    if mask.ndim != 1 or mask.size != ids.size:
        raise ValueError(f"{field} attention_mask must match token_ids")
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(mask.dtype, np.integer):
        raise TypeError(f"{field} token_ids and attention_mask must be integers")
    ids_values = tuple(int(item) for item in ids)
    mask_values = tuple(int(item) for item in mask)
    if any(item < INT32_MIN or item > INT32_MAX for item in ids_values):
        raise ValueError(f"{field} token_ids exceed int32")
    if any(item not in (0, 1) for item in mask_values):
        raise ValueError(f"{field} attention_mask values must be 0 or 1")
    if not any(mask_values):
        raise ValueError(f"{field} attention_mask cannot be all padding")
    if any(left == 0 and right == 1 for left, right in zip(mask_values, mask_values[1:])):
        raise ValueError(f"{field} attention_mask must be one contiguous valid prefix")
    return ids_values, mask_values


def _integer_array(value, field: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as error:
        raise TypeError(f"{field} must be an integer sequence") from error


def _loaded_embedding(value: LoadedEmbedding) -> LoadedEmbedding:
    embedding = np.ascontiguousarray(value.values, dtype=np.float32)
    mask = _integer_array(value.attention_mask, "fixed embedding attention_mask")
    if embedding.shape != EMBEDDING_SHAPE or not np.isfinite(embedding).all():
        raise ValueError("fixed embedding must be finite F32 [64,4096]")
    if mask.shape != (64,):
        raise ValueError("fixed embedding attention_mask must have shape [64]")
    _, normalized_mask = _tokens(
        np.zeros(64, dtype=np.int32), mask, "fixed embedding"
    )
    return LoadedEmbedding(
        embedding,
        np.ascontiguousarray(normalized_mask, dtype=np.int32),
        value.source_sha256,
    )


def _image(value, field: str) -> np.ndarray:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"{field} torch tensor must use rank-3 CHW RGB layout")
        if tensor.shape[1] <= 0 or tensor.shape[2] <= 0:
            raise ValueError(f"{field} height and width must be positive")
        if tensor.is_floating_point():
            minimum, maximum = torch.aminmax(tensor)
            minimum = float(minimum)
            maximum = float(maximum)
            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ValueError(f"{field} contains NaN or Inf")
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(f"{field} float values must be in [0,1]")
            tensor = tensor.mul(255.0)
            tensor.round_()
            tensor = tensor.to(torch.uint8)
        elif tensor.dtype != torch.uint8:
            raise TypeError(f"{field} must use float [0,1] or uint8 [0,255]")
        return tensor.permute(1, 2, 0).contiguous().numpy()

    array, torch_input = _numpy(value, field)
    if array.ndim != 3:
        raise ValueError(f"{field} must have rank 3, got {array.shape}")
    if torch_input:
        if array.shape[0] != 3:
            raise ValueError(f"{field} torch tensor must use CHW RGB layout")
        array = np.transpose(array, (1, 2, 0))
    elif array.shape[-1] == 3 and array.shape[0] != 3:
        pass
    elif array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    elif array.shape[0] == 3 and array.shape[-1] == 3:
        raise ValueError(f"{field} NumPy layout is ambiguous; use unambiguous CHW or HWC")
    else:
        raise ValueError(f"{field} must have exactly three RGB channels")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{field} height and width must be positive")
    if np.issubdtype(array.dtype, np.floating):
        minimum = array.min()
        maximum = array.max()
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(f"{field} contains NaN or Inf")
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(f"{field} float values must be in [0,1]")
        scaled = np.multiply(array, 255.0)
        np.rint(scaled, out=scaled)
        array = scaled.astype(np.uint8)
    elif array.dtype != np.uint8:
        raise TypeError(f"{field} must use float [0,1] or uint8 [0,255]")
    return np.ascontiguousarray(array)


def _state(value) -> np.ndarray:
    array, _ = _numpy(value, "observation.state")
    if tuple(array.shape) not in STATE_SHAPES:
        raise ValueError(f"observation.state must have shape [14] or [1,14], got {array.shape}")
    result = np.ascontiguousarray(array.reshape(14), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("observation.state contains NaN or Inf")
    return result


def _noise(value, field: str, *, allow_flat: bool = False) -> np.ndarray:
    array, _ = _numpy(value, field)
    if tuple(array.shape) != NOISE_SHAPE:
        if allow_flat and array.ndim == 1 and array.size == np.prod(NOISE_SHAPE):
            array = array.reshape(NOISE_SHAPE)
        else:
            raise ValueError(f"{field} must have shape [48,32], got {array.shape}")
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{field} contains NaN or Inf")
    return result


def _set_tensor(message, value: np.ndarray, dtype: int, layout: str) -> None:
    little = value.astype("<f4" if dtype == 3 else "<i4", copy=False)
    message.dtype = dtype
    message.shape.extend(value.shape)
    message.layout = layout
    message.byte_order = 1
    message.data = little.tobytes(order="C")


class WamCppPolicy:
    """Map frozen GWP-0.5 observations to one serialized WAM RPC session."""

    def __init__(
        self,
        service,
        message_types,
        *,
        language_policy: str = "external_embedding",
        fixed_embedding: Optional[LoadedEmbedding] = None,
        fixed_tokens: Optional[tuple[Sequence[int], Sequence[int]]] = None,
        tokenizer: Optional[Callable[[str], Mapping]] = None,
        noise_policy: str = "session",
        fixed_noise: Optional[np.ndarray] = None,
        random_seed: int = 0,
        prefix_cache: bool = True,
        expected_backend: Optional[int] = None,
        expected_precision: Optional[int] = None,
    ):
        self._service = service
        try:
            if language_policy not in ("resident", "fixed", "external_embedding"):
                raise ValueError(f"unknown language policy: {language_policy}")
            if noise_policy not in ("session", "fixed", "adapter-random"):
                raise ValueError(f"unknown noise policy: {noise_policy}")
            if noise_policy == "fixed" and fixed_noise is None:
                raise ValueError("fixed noise policy requires fixed_noise")
            if language_policy == "fixed" and fixed_tokens is None:
                raise ValueError("fixed language policy requires fixed_tokens")
            if fixed_noise is not None:
                fixed_noise = _noise(fixed_noise, "fixed noise")
            if fixed_embedding is not None:
                fixed_embedding = _loaded_embedding(fixed_embedding)
            if fixed_tokens is not None:
                fixed_tokens = _tokens(fixed_tokens[0], fixed_tokens[1], "fixed tokens")
        except Exception:
            self._service.close()
            raise
        self._types = message_types
        self.language_policy = language_policy
        self.noise_policy = noise_policy
        self.fixed_embedding = fixed_embedding
        self.fixed_embedding_sha256 = (
            fixed_embedding.source_sha256 if fixed_embedding is not None else None
        )
        self.fixed_tokens = fixed_tokens
        self.fixed_noise = fixed_noise
        self.tokenizer = tokenizer
        self._expected_backend = expected_backend
        self._expected_precision = expected_precision
        self._random_seed = int(random_seed)
        self._random = np.random.default_rng(self._random_seed)
        self._lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        self.last_stats = None
        self.last_request_id = None
        self.last_adapter_milliseconds = None
        self.last_timing = None
        try:
            self.model_info = self._service.get_model_info().model
            self._validate_model_info(self.model_info)
            self._session_id = self._service.create_session(
                prefix_cache=prefix_cache
            ).session_id
            if not self._session_id:
                raise RuntimeError("WAM CreateSession returned an empty session_id")
        except Exception:
            self._service.close()
            raise

    @classmethod
    def create(
        cls,
        library: Path,
        model: Path,
        descriptor: Path,
        *,
        backend: int = Backend.CUDA,
        precision: int = Precision.BF16_LATENCY,
        device_index: int = 0,
        prompt_cache_capacity: int = 4,
        language_policy: str = "external_embedding",
        fixed_t5_path: Optional[Path] = None,
        trust_fixed_t5: bool = False,
        fixed_token_json: Optional[Path] = None,
        tokenizer: Optional[Callable[[str], Mapping]] = None,
        noise_policy: str = "session",
        fixed_noise_path: Optional[Path] = None,
        random_seed: int = 0,
        prefix_cache: bool = True,
        torch_cpu_threads: Optional[int] = 16,
    ):
        if torch_cpu_threads is not None:
            if int(torch_cpu_threads) < 1:
                raise ValueError("torch_cpu_threads must be positive")
            try:
                import torch
            except ModuleNotFoundError:
                pass
            else:
                torch.set_num_threads(int(torch_cpu_threads))
        types = load_types(descriptor)
        embedding = load_fixed_prompt_embedding(
            fixed_t5_path, trusted=trust_fixed_t5
        ) if fixed_t5_path is not None else None
        tokens = load_fixed_tokens(fixed_token_json) \
            if fixed_token_json is not None else None
        noise = load_fixed_noise(fixed_noise_path) \
            if fixed_noise_path is not None else None
        if language_policy == "fixed" and tokens is None:
            raise ValueError("fixed language policy requires fixed_token_json")
        if language_policy != "fixed" and tokens is not None:
            raise ValueError("fixed_token_json is valid only with fixed language policy")
        if noise_policy == "fixed" and noise is None:
            raise ValueError("fixed noise policy requires fixed_noise_path")
        if noise_policy != "fixed" and noise is not None:
            raise ValueError("fixed_noise_path is valid only with fixed noise policy")
        policy_value = {
            "resident": LanguageEncoderPolicy.RESIDENT,
            "fixed": LanguageEncoderPolicy.FIXED,
            "external_embedding": LanguageEncoderPolicy.EXTERNAL_EMBEDDING,
        }.get(language_policy)
        if policy_value is None:
            raise ValueError(f"unknown language policy: {language_policy}")
        options = ModelOptions(
            artifact_path=model,
            backend=backend,
            precision=precision,
            device_index=device_index,
            prompt_cache_capacity=prompt_cache_capacity,
            language_encoder_policy=policy_value,
            fixed_token_ids=tokens[0] if tokens else (),
            fixed_attention_mask=tokens[1] if tokens else (),
        )
        service = WamRpcService(library, options, types)
        return cls(
            service,
            types,
            language_policy=language_policy,
            fixed_embedding=embedding,
            fixed_tokens=tokens,
            tokenizer=tokenizer,
            noise_policy=noise_policy,
            fixed_noise=noise,
            random_seed=random_seed,
            prefix_cache=prefix_cache,
            expected_backend=None if int(backend) == int(Backend.AUTO) else backend,
            expected_precision=precision,
        )

    def _validate_model_info(self, info) -> None:
        if info.architecture != "gwp05":
            raise RuntimeError(f"expected gwp05 model, got {info.architecture!r}")
        capabilities = info.capabilities
        if info.backend not in capabilities.backends:
            raise RuntimeError("loaded model reports a backend outside its capabilities")
        if self._expected_backend is not None and (
            info.backend != int(self._expected_backend) or
            int(self._expected_backend) not in capabilities.backends
        ):
            raise RuntimeError("loaded model does not support the requested backend")
        if self._expected_precision is not None and (
            info.precision != int(self._expected_precision) or
            int(self._expected_precision) not in capabilities.precisions
        ):
            raise RuntimeError("loaded model does not support the requested precision")
        required = {
            "action": capabilities.action,
            "raw_images": capabilities.raw_images,
            "precomputed_embedding": capabilities.precomputed_embedding,
            "explicit_noise": capabilities.explicit_noise,
        }
        if self.language_policy != "external_embedding":
            required["token_input"] = capabilities.token_input
        if self.language_policy == "resident":
            required["arbitrary_token_input"] = capabilities.arbitrary_token_input
        if self.language_policy == "fixed":
            required["fixed_token_input"] = capabilities.fixed_token_input
        missing = [name for name, available in required.items() if not available]
        if missing:
            raise RuntimeError(f"model lacks required capabilities: {', '.join(missing)}")

    def _language(self, observation, request) -> None:
        value = observation.get("prompt_embedding")
        loaded = None
        if value is not None:
            embedding, mask = _normalize_embedding(value, "prompt_embedding")
        elif self.fixed_embedding is not None:
            loaded = self.fixed_embedding
            embedding, mask = loaded.values, loaded.attention_mask
        else:
            embedding = mask = None
        if embedding is not None:
            target = request.inputs.language.precomputed_embedding
            _set_tensor(target.embedding, embedding, 3, "T,D")
            _set_tensor(target.attention_mask, mask, 2, "T")
            return

        if observation.get("lang_tokens") is not None:
            tokens = _tokens(
                observation.get("lang_tokens"),
                observation.get("attention_mask"),
                "request tokens",
            )
        elif self.fixed_tokens is not None:
            tokens = _tokens(self.fixed_tokens[0], self.fixed_tokens[1], "fixed tokens")
        elif observation.get("prompt") is not None:
            if self.tokenizer is None:
                raise ValueError("request prompt requires an explicitly configured tokenizer")
            prompt = observation["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("request prompt must be a nonempty string")
            encoded = self.tokenizer(prompt)
            tokens = _tokens(
                encoded.get("input_ids"), encoded.get("attention_mask"), "tokenizer output"
            )
        else:
            raise ValueError("request has no supported language input")
        if self.language_policy == "external_embedding":
            raise ValueError("external_embedding policy requires prompt_embedding")
        request.inputs.language.tokens.token_ids.extend(tokens[0])
        request.inputs.language.tokens.attention_mask.extend(tokens[1])

    def _build_request(self, observation, request_id: int):
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        request = self._types["PredictRequest"](
            session_id=self._session_id, request_id=request_id
        )
        image_values = []
        for source_name, logical_name in CAMERAS:
            if source_name not in observation:
                raise ValueError(f"required camera view is missing: {source_name}")
            image_values.append((source_name, logical_name, observation[source_name]))
        converted = [
            _image(value, source_name)
            for source_name, _logical_name, value in image_values
        ]
        for (_source_name, logical_name, _input), value in zip(image_values, converted):
            image = request.inputs.images.add()
            image.name = logical_name
            image.encoding = 1
            image.height, image.width, image.channels = value.shape
            image.row_stride_bytes = value.shape[1] * value.shape[2]
            image.data = value.tobytes(order="C")
        if "observation.state" not in observation:
            raise ValueError("required field is missing: observation.state")
        _set_tensor(request.inputs.state, _state(observation["observation.state"]), 3, "D")
        self._language(observation, request)

        if observation.get("noise") is not None:
            noise = _noise(observation["noise"], "request noise")
        elif self.noise_policy == "fixed":
            noise = self.fixed_noise
        elif self.noise_policy == "adapter-random":
            noise = self._random.standard_normal(NOISE_SHAPE, dtype=np.float32)
        else:
            noise = None
        if noise is not None:
            _set_tensor(request.inputs.noise, noise, 3, "T,A")
        return request

    def _action(self, response, request_id: int) -> np.ndarray:
        if response.request_id != request_id:
            raise RuntimeError(
                f"Predict response request_id is {response.request_id}, expected {request_id}"
            )
        tensor = response.prediction.action
        if tensor.dtype != 3 or tuple(tensor.shape) != ACTION_SHAPE or \
                tensor.layout != "T,A" or tensor.byte_order != 1:
            raise RuntimeError(
                "Predict action must be little-endian F32 [48,14] with layout T,A"
            )
        if len(tensor.data) != np.prod(ACTION_SHAPE) * 4:
            raise RuntimeError("Predict action payload has the wrong byte size")
        action = np.frombuffer(tensor.data, dtype="<f4").reshape(ACTION_SHAPE).copy()
        if not np.isfinite(action).all():
            raise RuntimeError("Predict action contains NaN or Inf")
        return np.ascontiguousarray(action, dtype=np.float32)

    def inference(self, observation) -> np.ndarray:
        begin = time.perf_counter()
        with self._lock:
            if self._closed:
                raise RuntimeError("WamCppPolicy is closed")
            self._request_id += 1
            request_id = self._request_id
            request = self._build_request(observation, request_id)
            mapped = time.perf_counter()
            response = self._service.predict(request)
            predicted = time.perf_counter()
            action = self._action(response, request_id)
            finished = time.perf_counter()
            self.last_stats = response.prediction.stats
            self.last_request_id = request_id
            self.last_adapter_milliseconds = (finished - begin) * 1000.0
            self.last_timing = AdapterTiming(
                mapping_milliseconds=(mapped - begin) * 1000.0,
                rpc_milliseconds=(predicted - mapped) * 1000.0,
                action_parse_milliseconds=(finished - predicted) * 1000.0,
                total_milliseconds=self.last_adapter_milliseconds,
            )
            return action

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("WamCppPolicy is closed")
            self._service.reset_session(self._session_id)
            self._random = np.random.default_rng(self._random_seed)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._service.close_session(self._session_id)
            finally:
                self._service.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _upstream_package(root: Path) -> Path:
    root = Path(root).resolve()
    candidates = (
        root / "third_party/giga-models",
        root / "giga-models",
    )
    for package in candidates:
        if (package / "giga_models/sockets/server.py").is_file():
            return package
    pythonpath = os.environ.get("PYTHONPATH", "")
    raise RuntimeError(
        "cannot find the frozen GigaWorld-Policy sockets package; expected "
        f"GIGA_WORLD_POLICY_ROOT={root} with third_party/giga-models; "
        f"current PYTHONPATH={pythonpath!r}; compatible upstream commit="
        f"{UPSTREAM_COMMIT}; clone https://github.com/open-gigaai/giga-world-policy "
        "and initialize its third-party dependencies"
    )


def load_robot_inference_server(upstream_root: Path):
    """Verify and import the unchanged upstream RobotInferenceServer."""
    root = Path(upstream_root).resolve()
    package = _upstream_package(root)
    for relative, expected in UPSTREAM_SOCKET_HASHES.items():
        path = package / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen upstream hash mismatch for {path}: {actual}, expected {expected}"
            )
    git = root / ".git"
    if git.exists():
        import subprocess
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != UPSTREAM_COMMIT:
            raise RuntimeError(
                f"GigaWorld-Policy revision is {revision}, expected {UPSTREAM_COMMIT}"
            )
    sys.path.insert(0, str(package))
    try:
        module = importlib.import_module("giga_models.sockets.server")
    except Exception as error:
        pythonpath = os.environ.get("PYTHONPATH", "")
        raise RuntimeError(
            "cannot import frozen giga_models.sockets; "
            f"GIGA_WORLD_POLICY_ROOT={root}; PYTHONPATH={pythonpath!r}; "
            f"compatible upstream commit={UPSTREAM_COMMIT}; install the upstream "
            f"runtime dependencies in the active Python environment; cause: {error}"
        ) from error
    imported = Path(module.__file__).resolve()
    expected_root = package.resolve()
    if expected_root not in imported.parents:
        raise RuntimeError(
            f"giga_models.sockets resolved to {imported}, outside {expected_root}"
        )
    return module.RobotInferenceServer


def is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    if normalized in ("", "*", "0.0.0.0", "::", "[::]"):
        return False
    try:
        return ipaddress.ip_address(normalized.strip("[]")).is_loopback
    except ValueError:
        try:
            addresses = {
                item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            }
        except socket.gaierror:
            return False
        return bool(addresses) and all(
            ipaddress.ip_address(address).is_loopback for address in addresses
        )


def validate_bind_host(host: str, allow_unsafe_remote: bool, *, stream=None) -> None:
    if is_loopback_host(host):
        return
    if not allow_unsafe_remote:
        raise ValueError(
            f"refusing non-loopback ZeroMQ bind {host!r}; pass --allow-unsafe-remote "
            "only on a trusted network"
        )
    stream = stream or sys.stderr
    print(
        "WARNING: GigaWorld-Policy ZeroMQ uses torch.load(weights_only=False); "
        f"binding {host!r} permits unsafe pickle deserialization from the network",
        file=stream,
        flush=True,
    )


def _error_record(error: Exception) -> dict:
    record = {
        "format": "wam-gwp-server-error-v1",
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if isinstance(error, WamRpcError):
        record.update({
            "wam_error_code": error.error_code,
            "wam_error_name": error.error_name,
            "grpc_status_code": error.grpc_status_code,
            "details": [
                {"field": detail.field, "reason": detail.reason}
                for detail in error.details
            ],
        })
    return record


class ReportingPolicy:
    """Preserve upstream ERROR behavior while adding structured stderr evidence."""

    def __init__(self, policy, *, stream=None):
        self.policy = policy
        self.stream = stream or sys.stderr

    def inference(self, observation):
        try:
            return self.policy.inference(observation)
        except Exception as error:
            print(
                "wam-gwp-error " + json.dumps(_error_record(error), sort_keys=True),
                file=self.stream,
                flush=True,
            )
            raise


class _TerminationSignal(BaseException):
    def __init__(self, signum: int):
        self.signum = int(signum)


def _install_termination_handlers():
    if threading.current_thread() is not threading.main_thread():
        return {}
    previous = {}

    def terminate(signum, _frame):
        raise _TerminationSignal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, terminate)
    return previous


def _restore_signal_handlers(previous) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _close_upstream_server(server) -> None:
    socket_value = getattr(server, "socket", None)
    context = getattr(server, "context", None)
    try:
        if socket_value is not None:
            socket_value.close(linger=0)
    finally:
        if context is not None:
            context.term()


def run_compatible_server(
    policy,
    upstream_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 11444,
    allow_unsafe_remote: bool = False,
    robot_server_class=None,
    bind_validated: bool = False,
) -> int:
    if not bind_validated:
        validate_bind_host(host, allow_unsafe_remote)
    server_type = robot_server_class or load_robot_inference_server(upstream_root)
    server = None
    exit_code = 0
    previous_handlers = _install_termination_handlers()
    try:
        server = server_type(ReportingPolicy(policy), host=host, port=port)
        try:
            server.run()
        except _TerminationSignal as termination:
            exit_code = 128 + termination.signum
    finally:
        try:
            policy.close()
        finally:
            try:
                if server is not None:
                    _close_upstream_server(server)
            finally:
                _restore_signal_handlers(previous_handlers)
    return exit_code


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11444)
    parser.add_argument("--allow-unsafe-remote", action="store_true")
    parser.add_argument("--backend", choices=("cuda", "auto"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt-cache-capacity", type=int, default=4)
    parser.add_argument(
        "--language-policy",
        choices=("resident", "fixed", "external_embedding"),
        default="external_embedding",
    )
    parser.add_argument("--fixed-t5", type=Path)
    parser.add_argument("--trust-fixed-t5", action="store_true")
    parser.add_argument("--fixed-token-json", type=Path)
    parser.add_argument(
        "--noise-policy", choices=("session", "fixed", "adapter-random"),
        default="session",
    )
    parser.add_argument("--fixed-noise", type=Path)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--torch-cpu-threads", type=int, default=16)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1,65535]")
    if args.device < 0 or args.prompt_cache_capacity < 0 or args.torch_cpu_threads < 1:
        parser.error(
            "--device/--prompt-cache-capacity must be non-negative and "
            "--torch-cpu-threads must be positive"
        )
    if args.fixed_t5 is not None and not args.trust_fixed_t5:
        parser.error("--fixed-t5 requires --trust-fixed-t5")
    return args


def main() -> int:
    args = parse_args()
    validate_bind_host(args.host, args.allow_unsafe_remote)
    server_type = load_robot_inference_server(args.upstream_root)
    policy = WamCppPolicy.create(
        args.library,
        args.model,
        args.descriptor,
        backend={"cuda": Backend.CUDA, "auto": Backend.AUTO}[args.backend],
        precision={"bf16": Precision.BF16_LATENCY,
                   "f32": Precision.F32_REFERENCE}[args.precision],
        device_index=args.device,
        prompt_cache_capacity=args.prompt_cache_capacity,
        language_policy=args.language_policy,
        fixed_t5_path=args.fixed_t5,
        trust_fixed_t5=args.trust_fixed_t5,
        fixed_token_json=args.fixed_token_json,
        noise_policy=args.noise_policy,
        fixed_noise_path=args.fixed_noise,
        random_seed=args.random_seed,
        prefix_cache=not args.disable_prefix_cache,
        torch_cpu_threads=args.torch_cpu_threads,
    )
    return run_compatible_server(
        policy,
        args.upstream_root,
        host=args.host,
        port=args.port,
        allow_unsafe_remote=args.allow_unsafe_remote,
        robot_server_class=server_type,
        bind_validated=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
