"""Small ctypes bridge to the wam.cpp C ABI."""

from __future__ import annotations

import ctypes as C
import json
from pathlib import Path

import numpy as np


class NativeError(RuntimeError):
    def __init__(self, code: int, message: str, details):
        super().__init__(message)
        self.code = code
        self.details = details


class _Error(C.Structure):
    _fields_ = [("code", C.c_uint32), ("message", C.c_void_p),
                ("details_json", C.c_void_p)]


class _ModelOptions(C.Structure):
    _fields_ = [("struct_size", C.c_size_t), ("artifact_path", C.c_char_p),
                ("backend", C.c_uint32), ("compute_precision", C.c_uint32),
                ("device_index", C.c_int32),
                ("prompt_cache_capacity", C.c_size_t),
                ("language_mode", C.c_uint32)]


class _SessionOptions(C.Structure):
    _fields_ = [("struct_size", C.c_size_t),
                ("enable_prefix_cache", C.c_uint8),
                ("random_seed", C.c_uint64)]


class _Image(C.Structure):
    _fields_ = [("name", C.c_char_p), ("encoding", C.c_uint32),
                ("data", C.POINTER(C.c_uint8)), ("byte_size", C.c_size_t),
                ("width", C.c_uint32), ("height", C.c_uint32),
                ("channels", C.c_uint32), ("row_stride_bytes", C.c_size_t)]


class _TensorView(C.Structure):
    _fields_ = [("data", C.c_void_p), ("byte_size", C.c_size_t),
                ("dtype", C.c_uint32), ("shape", C.POINTER(C.c_int64)),
                ("rank", C.c_size_t), ("layout", C.c_char_p),
                ("byte_order", C.c_uint32)]


class _PredictInputs(C.Structure):
    _fields_ = [("images", C.POINTER(_Image)), ("image_count", C.c_size_t),
                ("token_ids", C.POINTER(C.c_int32)),
                ("attention_mask", C.POINTER(C.c_int32)),
                ("token_count", C.c_size_t), ("state", _TensorView),
                ("action_noise", _TensorView),
                ("embedding", _TensorView),
                ("embedding_attention_mask", _TensorView)]


class _PhaseTiming(C.Structure):
    _fields_ = [("name", C.c_void_p), ("milliseconds", C.c_double)]


class _Stats(C.Structure):
    _fields_ = [(name, C.c_double) for name in (
        "preprocess_milliseconds", "model_milliseconds",
        "model_vision_milliseconds", "model_text_milliseconds",
        "model_prefill_milliseconds", "model_decode_milliseconds",
        "postprocess_milliseconds", "total_milliseconds")] + [
        ("peak_device_memory_bytes", C.c_uint64),
        ("model_timings", C.POINTER(_PhaseTiming)),
        ("model_timing_count", C.c_size_t)]


class _Prediction(C.Structure):
    _fields_ = [("action_data", C.POINTER(C.c_uint8)),
                ("action_byte_size", C.c_size_t),
                ("action_dtype", C.c_uint32),
                ("action_shape", C.POINTER(C.c_int64)),
                ("action_rank", C.c_size_t), ("stats", _Stats)]


BACKENDS = {"automatic": 1, "cuda": 2, "cpu-metadata": 3}
PRECISIONS = {"automatic": 1, "f32": 2, "f16": 3, "bf16": 4}
LANGUAGE_MODES = {"automatic": 0, "tokens": 1, "external_embedding": 2}


def _configure(lib):
    lib.wam_c_abi_version.restype = C.c_uint32
    lib.wam_c_model_create.argtypes = [C.POINTER(_ModelOptions),
                                        C.POINTER(C.c_void_p), C.POINTER(_Error)]
    lib.wam_c_model_metadata_json.argtypes = [C.c_void_p,
                                               C.POINTER(C.c_void_p), C.POINTER(_Error)]
    lib.wam_c_session_create.argtypes = [C.c_void_p, C.POINTER(_SessionOptions),
                                          C.POINTER(C.c_void_p), C.POINTER(_Error)]
    lib.wam_c_session_predict.argtypes = [C.c_void_p, C.POINTER(_PredictInputs),
                                           C.POINTER(_Prediction), C.POINTER(_Error)]
    lib.wam_c_session_reset.argtypes = [C.c_void_p, C.POINTER(_Error)]
    lib.wam_c_prediction_free.argtypes = [C.POINTER(_Prediction)]
    lib.wam_c_string_free.argtypes = [C.c_void_p]
    lib.wam_c_error_free.argtypes = [C.POINTER(_Error)]
    lib.wam_c_session_free.argtypes = [C.c_void_p]
    lib.wam_c_model_free.argtypes = [C.c_void_p]


def _check(lib, status, error):
    if status == 0:
        return
    message = C.string_at(error.message).decode() if error.message else "wam C ABI failure"
    details = json.loads(C.string_at(error.details_json).decode()) if error.details_json else []
    code = int(error.code)
    lib.wam_c_error_free(C.byref(error))
    raise NativeError(code, message, details)


def _tensor_view(array, dtype_code, layout=b""):
    if array is None:
        return _TensorView(None, 0, 0, None, 0, b"", 3), None
    array = np.ascontiguousarray(array)
    shape = (C.c_int64 * array.ndim)(*array.shape)
    view = _TensorView(array.ctypes.data, array.nbytes, dtype_code, shape,
                       array.ndim, layout, 1)
    return view, (array, shape)


class NativeModel:
    def __init__(self, library, artifact, backend="cuda", precision="bf16",
                 device=0, prompt_cache_capacity=4, language_mode="automatic"):
        self.lib = C.CDLL(str(Path(library).resolve()))
        _configure(self.lib)
        abi_version = self.lib.wam_c_abi_version()
        if abi_version != 3:
            raise RuntimeError(
                f"wam C ABI version mismatch: expected 3, got {abi_version}")
        self.handle = C.c_void_p()
        path = str(Path(artifact).resolve()).encode()
        options = _ModelOptions(C.sizeof(_ModelOptions), path, BACKENDS[backend],
                                PRECISIONS[precision], device,
                                prompt_cache_capacity,
                                LANGUAGE_MODES[language_mode])
        error = _Error()
        _check(self.lib, self.lib.wam_c_model_create(
            C.byref(options), C.byref(self.handle), C.byref(error)), error)
        try:
            value = C.c_void_p()
            error = _Error()
            _check(self.lib, self.lib.wam_c_model_metadata_json(
                self.handle, C.byref(value), C.byref(error)), error)
            try:
                self.metadata = json.loads(C.string_at(value).decode())
            finally:
                self.lib.wam_c_string_free(value)
        except Exception:
            self.close()
            raise

    def create_session(self, random_seed=0, enable_prefix_cache=True):
        return NativeSession(self, random_seed, enable_prefix_cache)

    def close(self):
        if self.handle:
            self.lib.wam_c_model_free(self.handle)
            self.handle = C.c_void_p()


class NativeSession:
    def __init__(self, model, random_seed, enable_prefix_cache):
        self.model = model
        self.handle = C.c_void_p()
        options = _SessionOptions(C.sizeof(_SessionOptions),
                                  bool(enable_prefix_cache), random_seed)
        error = _Error()
        _check(model.lib, model.lib.wam_c_session_create(
            model.handle, C.byref(options), C.byref(self.handle), C.byref(error)), error)

    def predict(self, images, state, language_values, attention_mask,
                action_noise=None):
        keepalive = []
        image_array = (_Image * len(images))()
        for index, item in enumerate(images):
            pixels = np.ascontiguousarray(item["data"], dtype=np.uint8)
            name = item["name"].encode()
            keepalive.extend((pixels, name))
            image_array[index] = _Image(
                name, 1, pixels.ctypes.data_as(C.POINTER(C.c_uint8)), pixels.nbytes,
                pixels.shape[1], pixels.shape[0], 3, pixels.strides[0])
        state_view, state_keep = _tensor_view(
            np.ascontiguousarray(state, dtype=np.float32), 3)
        noise_view, noise_keep = _tensor_view(
            None if action_noise is None else np.ascontiguousarray(action_noise, dtype=np.float32), 3)
        mode = self.model.metadata["language_mode"]
        empty_view, _ = _tensor_view(None, 0)
        token_ids = None
        embedding_keep = mask_keep = None
        embedding_view = mask_view = empty_view
        token_pointer = mask_pointer = None
        token_count = 0
        if mode == "tokens":
            token_ids = np.ascontiguousarray(language_values, dtype=np.int32)
            attention_mask = np.ascontiguousarray(attention_mask, dtype=np.int32)
            token_pointer = token_ids.ctypes.data_as(C.POINTER(C.c_int32))
            mask_pointer = attention_mask.ctypes.data_as(C.POINTER(C.c_int32))
            token_count = token_ids.size
        elif mode == "external_embedding":
            embedding = np.ascontiguousarray(language_values)
            if embedding.dtype != np.uint16:
                raise ValueError("external embedding must use BF16 uint16 storage")
            mask = np.ascontiguousarray(attention_mask, dtype=np.int32)
            embedding_view, embedding_keep = _tensor_view(embedding, 4, b"T,D")
            mask_view, mask_keep = _tensor_view(mask, 2, b"T")
        else:
            raise ValueError(f"unsupported native language mode: {mode}")
        keepalive.extend((image_array, state_keep, noise_keep, token_ids,
                          attention_mask, embedding_keep, mask_keep))
        inputs = _PredictInputs(
            image_array, len(images),
            token_pointer, mask_pointer, token_count, state_view, noise_view,
            embedding_view, mask_view)
        output = _Prediction()
        error = _Error()
        _check(self.model.lib, self.model.lib.wam_c_session_predict(
            self.handle, C.byref(inputs), C.byref(output), C.byref(error)), error)
        try:
            if output.action_dtype != 3:
                raise NativeError(8, f"native action dtype is {output.action_dtype}, expected F32", [])
            shape = tuple(output.action_shape[index] for index in range(output.action_rank))
            count = output.action_byte_size // np.dtype(np.float32).itemsize
            action = np.ctypeslib.as_array(
                C.cast(output.action_data, C.POINTER(C.c_float)), shape=(count,)).copy().reshape(shape)
            scalar_names = tuple(
                name for name, _ in _Stats._fields_
                if name not in ("model_timings", "model_timing_count"))
            stats = {name: getattr(output.stats, name)
                     for name in scalar_names}
            stats["model_timings"] = [
                {
                    "name": C.string_at(
                        output.stats.model_timings[index].name).decode(),
                    "milliseconds": output.stats.model_timings[
                        index].milliseconds,
                }
                for index in range(output.stats.model_timing_count)
            ]
            return action, stats
        finally:
            self.model.lib.wam_c_prediction_free(C.byref(output))

    def reset(self):
        error = _Error()
        _check(self.model.lib, self.model.lib.wam_c_session_reset(
            self.handle, C.byref(error)), error)

    def close(self):
        if self.handle:
            self.model.lib.wam_c_session_free(self.handle)
            self.handle = C.c_void_p()
