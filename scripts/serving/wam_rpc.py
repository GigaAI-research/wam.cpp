#!/usr/bin/env python3
"""In-process Python bridge for the public WAM serialized RPC C ABI."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Sequence, Type


ABI_VERSION = 2


class Backend(IntEnum):
    AUTO = 1
    CUDA = 2
    CPU_METADATA = 3


class Precision(IntEnum):
    F32_REFERENCE = 1
    BF16_LATENCY = 2


class LanguageEncoderPolicy(IntEnum):
    RESIDENT = 0
    FIXED = 1
    EXTERNAL_EMBEDDING = 2


class PrefixCachePolicy(IntEnum):
    UNSPECIFIED = 0
    ENABLED = 1
    DISABLED = 2


class RpcMethod(IntEnum):
    GET_MODEL_INFO = 1
    CREATE_SESSION = 2
    PREDICT = 3
    RESET_SESSION = 4
    CLOSE_SESSION = 5


ERROR_NAMES = {
    0: "OK",
    1: "INVALID_ARGUMENT",
    2: "NOT_FOUND",
    3: "UNSUPPORTED",
    4: "INCOMPATIBLE_ARTIFACT",
    5: "RESOURCE_EXHAUSTED",
    6: "FAILED_PRECONDITION",
    7: "INFERENCE_FAILED",
    8: "INTERNAL",
}


class WamRpcModelOptionsV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("artifact_path", ctypes.c_char_p),
        ("backend", ctypes.c_uint32),
        ("precision", ctypes.c_uint32),
        ("device_index", ctypes.c_int32),
        ("prompt_cache_capacity", ctypes.c_size_t),
        ("language_encoder_policy", ctypes.c_uint32),
        ("fixed_token_ids", ctypes.POINTER(ctypes.c_int32)),
        ("fixed_token_count", ctypes.c_size_t),
        ("fixed_attention_mask", ctypes.POINTER(ctypes.c_int32)),
        ("fixed_attention_mask_count", ctypes.c_size_t),
    ]


class WamRpcResult(ctypes.Structure):
    _fields_ = [
        ("error_code", ctypes.c_uint32),
        ("grpc_status_code", ctypes.c_uint32),
        ("payload", ctypes.POINTER(ctypes.c_uint8)),
        ("payload_size", ctypes.c_size_t),
        ("message", ctypes.c_char_p),
    ]


@dataclass(frozen=True)
class ModelOptions:
    artifact_path: Path
    backend: int = Backend.AUTO
    precision: int = Precision.F32_REFERENCE
    device_index: int = 0
    prompt_cache_capacity: int = 4
    language_encoder_policy: int = LanguageEncoderPolicy.RESIDENT
    fixed_token_ids: Sequence[int] = ()
    fixed_attention_mask: Sequence[int] = ()


@dataclass(frozen=True)
class ErrorDetail:
    field: str
    reason: str


class WamAbiError(RuntimeError):
    """The loaded shared library does not implement the expected C ABI."""


class WamRpcError(RuntimeError):
    def __init__(self, error_code: int, grpc_status_code: int, message: str,
                 details: Sequence[ErrorDetail] = (), payload: bytes = b""):
        self.error_code = int(error_code)
        self.error_name = ERROR_NAMES.get(self.error_code, f"UNKNOWN_{self.error_code}")
        self.grpc_status_code = int(grpc_status_code)
        self.message = message
        self.details = tuple(details)
        self.payload = payload
        suffix = ""
        if self.details:
            fields = ", ".join(
                f"{detail.field}: {detail.reason}" for detail in self.details
            )
            suffix = f" [{fields}]"
        super().__init__(f"{self.error_name}: {message}{suffix}")


class _OwnedResult:
    def __init__(self, library):
        self._library = library
        self.value = WamRpcResult()

    def __enter__(self) -> WamRpcResult:
        return self.value

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._library.wam_rpc_result_free(ctypes.byref(self.value))


class WamRpcService:
    """Own one native service and expose serialized and dynamic-Proto RPC calls."""

    def __init__(self, library: Path, options: ModelOptions,
                 message_types: Optional[Dict[str, Type]] = None,
                 expected_abi: int = ABI_VERSION):
        self._service = ctypes.c_void_p()
        self._options_keepalive = None
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        self._bind_functions()
        actual_abi = int(self._library.wam_rpc_c_abi_version())
        if actual_abi != expected_abi:
            raise WamAbiError(
                f"WAM C ABI mismatch: expected {expected_abi}, got {actual_abi}"
            )
        self.abi_version = actual_abi
        self._types = message_types
        native, keepalive = self._native_options(options)
        self._options_keepalive = keepalive
        with _OwnedResult(self._library) as result:
            status = self._library.wam_rpc_service_create_v2(
                ctypes.byref(native), ctypes.byref(self._service), ctypes.byref(result)
            )
            if status != 0 or result.error_code != 0:
                raise self._error(result, "service creation failed")
        if not self._service.value:
            raise RuntimeError("WAM service creation returned a null service")

    def _bind_functions(self) -> None:
        self._library.wam_rpc_c_abi_version.argtypes = []
        self._library.wam_rpc_c_abi_version.restype = ctypes.c_uint32
        self._library.wam_rpc_service_create_v2.argtypes = [
            ctypes.POINTER(WamRpcModelOptionsV2),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(WamRpcResult),
        ]
        self._library.wam_rpc_service_create_v2.restype = ctypes.c_int
        self._library.wam_rpc_service_call.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(WamRpcResult),
        ]
        self._library.wam_rpc_service_call.restype = ctypes.c_int
        self._library.wam_rpc_result_free.argtypes = [ctypes.POINTER(WamRpcResult)]
        self._library.wam_rpc_result_free.restype = None
        self._library.wam_rpc_service_free.argtypes = [ctypes.c_void_p]
        self._library.wam_rpc_service_free.restype = None

    @staticmethod
    def _native_options(options: ModelOptions):
        if options.prompt_cache_capacity < 0:
            raise ValueError("prompt_cache_capacity must be non-negative")
        artifact = str(Path(options.artifact_path).resolve()).encode()
        tokens = tuple(int(value) for value in options.fixed_token_ids)
        mask = tuple(int(value) for value in options.fixed_attention_mask)
        token_storage = (ctypes.c_int32 * len(tokens))(*tokens) if tokens else None
        mask_storage = (ctypes.c_int32 * len(mask))(*mask) if mask else None
        native = WamRpcModelOptionsV2(
            ctypes.sizeof(WamRpcModelOptionsV2),
            artifact,
            int(options.backend),
            int(options.precision),
            int(options.device_index),
            int(options.prompt_cache_capacity),
            int(options.language_encoder_policy),
            token_storage,
            len(tokens),
            mask_storage,
            len(mask),
        )
        return native, (artifact, token_storage, mask_storage)

    @staticmethod
    def _bytes(pointer, size: int) -> bytes:
        return ctypes.string_at(pointer, size) if size else b""

    def _error(self, result: WamRpcResult, fallback: str) -> WamRpcError:
        payload = self._bytes(result.payload, result.payload_size)
        message = result.message.decode("utf-8", errors="replace") \
            if result.message else fallback
        error_code = int(result.error_code)
        grpc_status = int(result.grpc_status_code)
        details = []
        if payload and self._types is not None:
            try:
                error = self._types["Error"].FromString(payload)
                error_code = int(error.code) or error_code
                message = error.message or message
                details = [ErrorDetail(item.field, item.reason) for item in error.details]
            except Exception:
                pass
        return WamRpcError(error_code, grpc_status, message, details, payload)

    def _require_open(self) -> None:
        if not self._service or not self._service.value:
            raise RuntimeError("WAM RPC service is closed")

    def call_bytes(self, method: int, request: bytes = b"") -> bytes:
        self._require_open()
        request = bytes(request)
        storage = (ctypes.c_uint8 * len(request)).from_buffer_copy(request) \
            if request else None
        with _OwnedResult(self._library) as result:
            status = self._library.wam_rpc_service_call(
                self._service, int(method), storage, len(request), ctypes.byref(result)
            )
            if status != 0 or result.error_code != 0:
                raise self._error(result, "WAM RPC call failed")
            return self._bytes(result.payload, result.payload_size)

    def _message(self, method: RpcMethod, request, response_name: str):
        if self._types is None:
            raise RuntimeError("dynamic Proto message types are required")
        payload = self.call_bytes(
            method, request.SerializeToString(deterministic=True)
        )
        return self._types[response_name].FromString(payload)

    def get_model_info(self):
        return self._message(
            RpcMethod.GET_MODEL_INFO,
            self._new_message("GetModelInfoRequest"),
            "GetModelInfoResponse",
        )

    def create_session(self, prefix_cache: Optional[bool] = None):
        request = self._new_message("CreateSessionRequest")
        if prefix_cache is not None:
            request.prefix_cache = int(
                PrefixCachePolicy.ENABLED if prefix_cache else PrefixCachePolicy.DISABLED
            )
        return self._message(
            RpcMethod.CREATE_SESSION, request, "CreateSessionResponse"
        )

    def predict(self, request):
        return self._message(RpcMethod.PREDICT, request, "PredictResponse")

    def reset_session(self, session_id: str):
        request = self._new_message("ResetSessionRequest", session_id=session_id)
        return self._message(
            RpcMethod.RESET_SESSION, request, "ResetSessionResponse"
        )

    def close_session(self, session_id: str):
        request = self._new_message("CloseSessionRequest", session_id=session_id)
        return self._message(
            RpcMethod.CLOSE_SESSION, request, "CloseSessionResponse"
        )

    def _new_message(self, name: str, **values):
        if self._types is None:
            raise RuntimeError("dynamic Proto message types are required")
        return self._types[name](**values)

    def close(self) -> None:
        if self._service and self._service.value:
            self._library.wam_rpc_service_free(self._service)
            self._service = ctypes.c_void_p()
        self._options_keepalive = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
