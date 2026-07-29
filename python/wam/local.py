from __future__ import annotations

import ctypes as C
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _native as N
from .errors import WamError
from .resources import LanguageResourceError, LanguageResources

BACKENDS = {"automatic": 1, "cuda": 2, "cpu-metadata": 3, "cpu": 4}
PRECISIONS = {"automatic": 1, "f32": 2, "f16": 3, "bf16": 4}
LANGUAGE_MODES = {"automatic": 0, "tokens": 1, "external_embedding": 2}


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str = "automatic"
    precision: str = "automatic"
    device: int = 0
    prompt_cache_capacity: int = 0
    language_mode: str = "automatic"


@dataclass(frozen=True)
class SessionConfig:
    enable_prefix_cache: bool = True
    random_seed: int = 0


@dataclass(frozen=True)
class Prediction:
    action: np.ndarray
    stats: dict


class Model:
    def __init__(self, artifact, *, config=None, library=None):
        self._lib = N.load_library(library)
        self._handle = C.c_void_p()
        config = config or RuntimeConfig()
        options = N.ModelOptions()
        self._lib.wam_c_model_options_init(C.byref(options))
        self._artifact_bytes = str(Path(artifact).expanduser().resolve()).encode()
        options.artifact_path = self._artifact_bytes
        try:
            options.backend = BACKENDS[config.backend]
            options.compute_precision = PRECISIONS[config.precision]
            options.language_mode = LANGUAGE_MODES[config.language_mode]
        except KeyError as error:
            raise ValueError(f"unsupported runtime option {error.args[0]!r}") from error
        options.device_index = config.device
        options.prompt_cache_capacity = config.prompt_cache_capacity
        error = C.POINTER(N.Error)()
        N.check(self._lib, self._lib.wam_c_model_create(
            C.byref(options), C.byref(self._handle), C.byref(error)), error)
        try:
            value = C.c_void_p()
            error = C.POINTER(N.Error)()
            N.check(self._lib, self._lib.wam_c_model_metadata_json(
                self._handle, C.byref(value), C.byref(error)), error)
            try:
                self.metadata = json.loads(C.string_at(value).decode())
            finally:
                self._lib.wam_c_string_free(value)
        except Exception:
            self.close()
            raise

    @classmethod
    def load(cls, artifact, *, config=None, library=None):
        return cls(artifact, config=config, library=library)

    def create_session(self, config=None):
        self._require_open()
        return Session(self, config or SessionConfig())

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("Model is closed")

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.wam_c_model_free(self._handle)
            self._handle = C.c_void_p()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


class Session:
    def __init__(self, model, config):
        self._model = model
        self._handle = C.c_void_p()
        options = N.SessionOptions()
        model._lib.wam_c_session_options_init(C.byref(options))
        options.enable_prefix_cache = config.enable_prefix_cache
        options.random_seed = config.random_seed
        error = C.POINTER(N.Error)()
        N.check(model._lib, model._lib.wam_c_session_create(
            model._handle, C.byref(options), C.byref(self._handle), C.byref(error)), error)

    def predict(self, images, state, *, token_ids=None, attention_mask=None,
                embedding=None, embedding_attention_mask=None, action_noise=None):
        if not self._handle:
            raise RuntimeError("Session is closed")
        lib, keepalive = self._model._lib, []
        image_values = (N.Image * len(images))()
        for index, item in enumerate(images):
            lib.wam_c_image_init(C.byref(image_values[index]))
            pixels = np.ascontiguousarray(item["data"], dtype=np.uint8)
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError(f"image {item['name']!r} must be HWC RGB")
            name = str(item["name"]).encode()
            image_values[index].name, image_values[index].encoding = name, 1
            image_values[index].data = pixels.ctypes.data_as(C.POINTER(C.c_uint8))
            image_values[index].byte_size = pixels.nbytes
            image_values[index].width, image_values[index].height = pixels.shape[1], pixels.shape[0]
            image_values[index].channels, image_values[index].row_stride_bytes = 3, pixels.strides[0]
            keepalive.extend((pixels, name))
        inputs = N.PredictInputs()
        lib.wam_c_predict_inputs_init(C.byref(inputs))
        inputs.images, inputs.image_count = image_values, len(images)
        inputs.state, state_keep = N.tensor_view(lib, state, np.float32)
        inputs.action_noise, noise_keep = N.tensor_view(lib, action_noise, np.float32)
        keepalive.extend((image_values, state_keep, noise_keep))
        if token_ids is not None:
            if embedding is not None:
                raise ValueError("token_ids and embedding are mutually exclusive")
            tokens = np.ascontiguousarray(token_ids, dtype=np.int32)
            mask = np.ascontiguousarray(attention_mask, dtype=np.int32)
            if tokens.shape != mask.shape or tokens.ndim != 1:
                raise ValueError("token_ids and attention_mask must be equal rank-1 arrays")
            inputs.token_ids = tokens.ctypes.data_as(C.POINTER(C.c_int32))
            inputs.attention_mask = mask.ctypes.data_as(C.POINTER(C.c_int32))
            inputs.token_count = tokens.size
            keepalive.extend((tokens, mask))
        elif embedding is not None:
            inputs.embedding, embedding_keep = N.tensor_view(lib, embedding, np.uint16, b"T,D")
            inputs.embedding_attention_mask, mask_keep = N.tensor_view(
                lib, embedding_attention_mask, np.int32, b"T")
            keepalive.extend((embedding_keep, mask_keep))
        output, error = C.POINTER(N.Prediction)(), C.POINTER(N.Error)()
        N.check(lib, lib.wam_c_session_predict(
            self._handle, C.byref(inputs), C.byref(output), C.byref(error)), error)
        try:
            value = output.contents
            if value.action_dtype != 3:
                raise WamError(8, f"native action dtype {value.action_dtype} is not F32")
            shape = tuple(value.action_shape[i] for i in range(value.action_rank))
            count = value.action_byte_size // 4
            action = np.ctypeslib.as_array(C.cast(value.action_data, C.POINTER(C.c_float)), shape=(count,)).copy().reshape(shape)
            names = [name for name, _ in N.Stats._fields_[2:10]]
            stats = {name: getattr(value.stats, name) for name in names}
            stats["peak_device_memory_bytes"] = value.stats.peak_device_memory_bytes
            stats["model_timings"] = [{"name": C.string_at(value.stats.model_timings[i].name).decode(),
                                        "milliseconds": value.stats.model_timings[i].milliseconds}
                                       for i in range(value.stats.model_timing_count)]
            return Prediction(action, stats)
        finally:
            lib.wam_c_prediction_free(output)

    def reset(self):
        if not self._handle:
            raise RuntimeError("Session is closed")
        error = C.POINTER(N.Error)()
        N.check(self._model._lib, self._model._lib.wam_c_session_reset(
            self._handle, C.byref(error)), error)

    def close(self):
        if getattr(self, "_handle", None):
            self._model._lib.wam_c_session_free(self._handle)
            self._handle = C.c_void_p()

    def __enter__(self):
        if not self._handle:
            raise RuntimeError("Session is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


class Pipeline:
    def __init__(self, model, session, resources, language_options):
        self.model, self.session = model, session
        self.language_resources = resources
        self._language_options = language_options
        self._language_provider = None

    @classmethod
    def load(cls, artifact, *, runtime_config=None, session_config=None,
             library=None, tokenizer=None, language_encoder=None,
             language_python_root=None, language_device="cuda:0",
             language_cache_capacity=32):
        resources = LanguageResources.discover(
            artifact, tokenizer=tokenizer, encoder=language_encoder)
        model = Model.load(artifact, config=runtime_config, library=library)
        try:
            options = dict(python_root=language_python_root,
                           device=language_device,
                           cache_capacity=language_cache_capacity)
            return cls(model, model.create_session(session_config), resources,
                       options)
        except Exception:
            model.close()
            raise

    def predict(self, images, state, *, instruction=None, **kwargs):
        if instruction is not None:
            if any(kwargs.get(name) is not None for name in (
                    "token_ids", "embedding", "attention_mask",
                    "embedding_attention_mask")):
                raise ValueError("instruction and prepared language inputs are mutually exclusive")
            if self._language_provider is None:
                from .language import create_language_provider
                mode = self.model.metadata["language_mode"]
                required = ("tokenizer",) if mode == "tokens" else (
                    "tokenizer", "encoder")
                try:
                    self.language_resources.require(*required)
                    self._language_provider = create_language_provider(
                        self.model.metadata,
                        tokenizer_path=self.language_resources.tokenizer,
                        encoder_path=self.language_resources.encoder,
                        **self._language_options)
                except ValueError as error:
                    raise LanguageResourceError(str(error)) from error
            language = self._language_provider.prepare(instruction)
            if self.model.metadata["language_mode"] == "tokens":
                kwargs.update(token_ids=language.values,
                              attention_mask=language.attention_mask)
            else:
                kwargs.update(embedding=language.values,
                              embedding_attention_mask=language.attention_mask)
            prediction = self.session.predict(images, state, **kwargs)
            prediction.stats["preprocess_milliseconds"] += language.preprocess_milliseconds
            prediction.stats["model_text_milliseconds"] += language.model_milliseconds
            prediction.stats["model_milliseconds"] += language.model_milliseconds
            prediction.stats["total_milliseconds"] += (
                language.preprocess_milliseconds + language.model_milliseconds)
            return prediction
        return self.session.predict(images, state, **kwargs)

    def reset(self):
        self.session.reset()

    def close(self):
        self.session.close()
        self.model.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
