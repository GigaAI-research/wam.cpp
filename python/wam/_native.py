from __future__ import annotations

import ctypes as C
import ctypes.util
import json
import os
from pathlib import Path

import numpy as np

from .errors import WamError

ABI_VERSION = 4


class Error(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t),
                ("code", C.c_uint32), ("message", C.c_void_p),
                ("details_json", C.c_void_p)]


class ModelOptions(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t),
                ("artifact_path", C.c_char_p), ("backend", C.c_uint32),
                ("compute_precision", C.c_uint32), ("device_index", C.c_int32),
                ("prompt_cache_capacity", C.c_size_t),
                ("language_mode", C.c_uint32)]


class SessionOptions(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t),
                ("enable_prefix_cache", C.c_uint8), ("random_seed", C.c_uint64)]


class Image(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t),
                ("name", C.c_char_p), ("encoding", C.c_uint32),
                ("data", C.POINTER(C.c_uint8)), ("byte_size", C.c_size_t),
                ("width", C.c_uint32), ("height", C.c_uint32),
                ("channels", C.c_uint32), ("row_stride_bytes", C.c_size_t)]


class TensorView(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t),
                ("data", C.c_void_p), ("byte_size", C.c_size_t),
                ("dtype", C.c_uint32), ("shape", C.POINTER(C.c_int64)),
                ("rank", C.c_size_t), ("layout", C.c_char_p),
                ("byte_order", C.c_uint32)]


class PredictInputs(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t),
                ("images", C.POINTER(Image)), ("image_count", C.c_size_t),
                ("token_ids", C.POINTER(C.c_int32)),
                ("attention_mask", C.POINTER(C.c_int32)),
                ("token_count", C.c_size_t), ("state", TensorView),
                ("action_noise", TensorView), ("embedding", TensorView),
                ("embedding_attention_mask", TensorView)]


class PhaseTiming(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t),
                ("name", C.c_void_p), ("milliseconds", C.c_double)]


class Stats(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t)] + [
        (name, C.c_double) for name in (
            "preprocess_milliseconds", "model_milliseconds",
            "model_vision_milliseconds", "model_text_milliseconds",
            "model_prefill_milliseconds", "model_decode_milliseconds",
            "postprocess_milliseconds", "total_milliseconds")
    ] + [("peak_device_memory_bytes", C.c_uint64),
         ("model_timings", C.POINTER(PhaseTiming)),
         ("model_timing_count", C.c_size_t)]


class Prediction(C.Structure):
    _fields_ = [("struct_version", C.c_uint32), ("struct_size", C.c_size_t),
                ("action_data", C.POINTER(C.c_uint8)),
                ("action_byte_size", C.c_size_t), ("action_dtype", C.c_uint32),
                ("action_shape", C.POINTER(C.c_int64)),
                ("action_rank", C.c_size_t), ("stats", Stats)]


def load_library(path=None):
    candidate = (path or os.environ.get("WAM_C_API_LIBRARY") or
                 ctypes.util.find_library("wam_c_api"))
    if not candidate:
        raise FileNotFoundError(
            "wam_c_api library was not found; pass library= or set "
            "WAM_C_API_LIBRARY")
    resolved = (str(Path(candidate).expanduser().resolve())
                if path or os.path.sep in candidate else candidate)
    lib = C.CDLL(resolved)
    lib.wam_c_abi_version.restype = C.c_uint32
    actual_abi = lib.wam_c_abi_version()
    if actual_abi != ABI_VERSION:
        raise RuntimeError(
            f"wam C ABI mismatch: Python requires v{ABI_VERSION}, "
            f"library provides v{actual_abi}")
    for name, type_ in (("model_options", ModelOptions), ("session_options", SessionOptions),
                        ("image", Image), ("tensor_view", TensorView),
                        ("predict_inputs", PredictInputs)):
        initializer = getattr(lib, f"wam_c_{name}_init")
        initializer.argtypes = [C.POINTER(type_)]
        initializer.restype = None
    lib.wam_c_model_create.argtypes = [C.POINTER(ModelOptions), C.POINTER(C.c_void_p), C.POINTER(C.POINTER(Error))]
    lib.wam_c_model_metadata_json.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.POINTER(Error))]
    lib.wam_c_session_create.argtypes = [C.c_void_p, C.POINTER(SessionOptions), C.POINTER(C.c_void_p), C.POINTER(C.POINTER(Error))]
    lib.wam_c_session_predict.argtypes = [C.c_void_p, C.POINTER(PredictInputs), C.POINTER(C.POINTER(Prediction)), C.POINTER(C.POINTER(Error))]
    lib.wam_c_session_reset.argtypes = [C.c_void_p, C.POINTER(C.POINTER(Error))]
    lib.wam_c_prediction_free.argtypes = [C.POINTER(Prediction)]
    lib.wam_c_string_free.argtypes = [C.c_void_p]
    lib.wam_c_error_free.argtypes = [C.POINTER(Error)]
    lib.wam_c_session_free.argtypes = [C.c_void_p]
    lib.wam_c_model_free.argtypes = [C.c_void_p]
    for name in ("wam_c_model_create", "wam_c_model_metadata_json",
                 "wam_c_session_create", "wam_c_session_predict",
                 "wam_c_session_reset"):
        getattr(lib, name).restype = C.c_int
    for name in ("wam_c_prediction_free", "wam_c_string_free",
                 "wam_c_error_free", "wam_c_session_free",
                 "wam_c_model_free"):
        getattr(lib, name).restype = None
    return lib


def check(lib, status, error):
    if status == 0:
        return
    if not error:
        raise WamError(8, "wam C ABI failed without an error object")
    try:
        value = error.contents
        message = (C.string_at(value.message).decode()
                   if value.message else "wam C ABI failure")
        details = (json.loads(C.string_at(value.details_json).decode())
                   if value.details_json else [])
        code = int(value.code)
    finally:
        lib.wam_c_error_free(error)
    raise WamError(code, message, details)


def tensor_view(lib, value, dtype, layout=b""):
    view = TensorView()
    lib.wam_c_tensor_view_init(C.byref(view))
    if value is None:
        return view, None
    array = np.ascontiguousarray(value, dtype=dtype)
    shape = (C.c_int64 * array.ndim)(*array.shape)
    view.data, view.byte_size = array.ctypes.data, array.nbytes
    view.dtype = {np.dtype(np.uint8): 1, np.dtype(np.int32): 2,
                  np.dtype(np.float32): 3, np.dtype(np.uint16): 4}[array.dtype]
    view.shape, view.rank, view.layout, view.byte_order = shape, array.ndim, layout, 1
    return view, (array, shape)
