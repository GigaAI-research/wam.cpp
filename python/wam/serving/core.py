from __future__ import annotations

import asyncio
import io
import logging

import numpy as np
from google.protobuf import json_format

from ..errors import ErrorCode, WamError
from ..local import SessionConfig
from ..remote import decode_tensor, encode_tensor, load_types

LOG = logging.getLogger("wam.service")
PROTOCOL_MAJOR = 0
PROTOCOL_MINOR = 6


class ServiceCore:
    """Transport-neutral RPC state and inference service."""

    def __init__(self, model, language_provider, descriptor, contract,
                 random_seed=0):
        self.model = model
        self.language_provider = language_provider
        self.types = load_types(descriptor)
        self.contract = contract
        self.random_seed = int(random_seed)
        concurrent = model.metadata["capabilities"].get(
            "concurrent_sessions", False)
        self.inference_lock = None if concurrent else asyncio.Lock()
        self.model_info = self.types["ModelInfo"]()
        json_format.ParseDict(model.metadata, self.model_info)

    def connection(self):
        return ServiceConnection(self)

    async def execute(self, function, *args):
        if self.inference_lock is None:
            return await asyncio.to_thread(function, *args)
        async with self.inference_lock:
            return await asyncio.to_thread(function, *args)


class ServiceConnection:
    def __init__(self, service):
        self.service = service
        self.session = None
        self.expected_request_id = 0
        self.closed = False

    def response(self, request_id):
        value = self.service.types["ServerEnvelope"]()
        value.request_id = request_id
        return value

    def error(self, request_id, code, message, *, field, reason=None,
              fatal=False):
        response = self.response(request_id)
        response.error.code = code
        response.error.message = message
        response.error.fatal = fatal
        detail = response.error.details.add()
        detail.field = field
        detail.reason = reason or message
        if fatal:
            self.closed = True
        return response

    def _native_error(self, request_id, error):
        mapping = {
            ErrorCode.INVALID_ARGUMENT: (3, False),
            ErrorCode.UNSUPPORTED: (4, False),
            ErrorCode.INCOMPATIBLE_ARTIFACT: (2, True),
            ErrorCode.FAILED_PRECONDITION: (2, True),
        }
        code, fatal = mapping.get(error.code, (5, True))
        response = self.response(request_id)
        response.error.code = code
        response.error.message = str(error)
        response.error.fatal = fatal
        for item in error.details or [{"field": "service", "reason": str(error)}]:
            detail = response.error.details.add()
            detail.field = str(item.get("field", "service"))
            detail.reason = str(item.get("reason", str(error)))
        if fatal:
            self.closed = True
        return response

    @staticmethod
    def _images(observation):
        result, names = [], set()
        for wire in observation.images:
            if not wire.name or wire.name in names:
                raise ValueError(f"duplicate or empty image role: {wire.name!r}")
            names.add(wire.name)
            if wire.encoding == 1:
                expected = int(wire.width) * int(wire.height) * 3
                if len(wire.data) != expected:
                    raise ValueError(
                        f"image {wire.name} payload has {len(wire.data)} "
                        f"bytes, expected {expected}")
                pixels = np.frombuffer(wire.data, dtype=np.uint8).reshape(
                    wire.height, wire.width, 3).copy()
            elif wire.encoding in (2, 3):
                from PIL import Image as PilImage
                pixels = np.asarray(
                    PilImage.open(io.BytesIO(wire.data)).convert("RGB"))
                if pixels.shape[:2] != (wire.height, wire.width):
                    raise ValueError(
                        f"decoded image {wire.name} shape {pixels.shape[:2]} "
                        f"!= {(wire.height, wire.width)}")
            else:
                raise ValueError(
                    f"image {wire.name} has unsupported encoding {wire.encoding}")
            result.append({"name": wire.name, "data": pixels})
        return result

    def _predict(self, request):
        observation = request.observation
        state = decode_tensor(observation.state)
        if state.dtype != np.float32 or state.ndim != 1:
            raise ValueError(
                f"state must be rank-1 F32, got {state.dtype} {state.shape}")
        language = self.service.language_provider.prepare(
            observation.instruction)
        noise = (decode_tensor(request.action_noise)
                 if request.HasField("action_noise") else None)
        kwargs = {"action_noise": noise}
        if self.service.model.metadata["language_mode"] == "tokens":
            kwargs.update(token_ids=language.values,
                          attention_mask=language.attention_mask)
        else:
            kwargs.update(embedding=language.values,
                          embedding_attention_mask=language.attention_mask)
        prediction = self.session.predict(
            self._images(observation), state, **kwargs)
        stats = prediction.stats
        stats["preprocess_milliseconds"] += language.preprocess_milliseconds
        stats["model_text_milliseconds"] += language.model_milliseconds
        stats["model_milliseconds"] += language.model_milliseconds
        stats["total_milliseconds"] += (
            language.preprocess_milliseconds + language.model_milliseconds)
        if language.model_milliseconds:
            stats["model_timings"].insert(0, {
                "name": "language_encoder",
                "milliseconds": language.model_milliseconds})
        return prediction.action, stats

    async def handle(self, request):
        request_id = int(request.request_id)
        if self.closed:
            return self.error(request_id, 1, "connection is closed",
                              field="connection", fatal=True)
        if request_id != self.expected_request_id:
            return self.error(
                request_id, 1,
                f"expected request id {self.expected_request_id}, got {request_id}",
                field="request_id", fatal=True)
        kind = request.WhichOneof("payload")
        if self.session is None:
            if kind != "hello":
                return self.error(request_id, 1,
                                  "the first message must be HelloRequest",
                                  field="hello", fatal=True)
            hello = request.hello
            if (hello.protocol_major, hello.protocol_minor) != (
                    PROTOCOL_MAJOR, PROTOCOL_MINOR):
                return self.error(
                    request_id, 2,
                    f"server protocol is {PROTOCOL_MAJOR}.{PROTOCOL_MINOR}",
                    field="protocol_version", fatal=True)
            if hello.environment_id != self.service.contract.environment_id:
                return self.error(
                    request_id, 2,
                    f"server environment is {self.service.contract.environment_id}",
                    field="environment_id", fatal=True)
            try:
                self.session = self.service.model.create_session(SessionConfig(
                    random_seed=self.service.random_seed))
            except WamError as error:
                return self._native_error(request_id, error)
            response = self.response(request_id)
            response.hello.model_info.CopyFrom(self.service.model_info)
            self.expected_request_id += 1
            return response

        if kind == "predict":
            try:
                action, stats = await self.service.execute(
                    self._predict, request.predict)
                response = self.response(request_id)
                encode_tensor(response.predict.prediction.action,
                              action, np.float32)
                for name, value in stats.items():
                    if name != "model_timings":
                        setattr(response.predict.prediction.stats, name, value)
                for item in stats.get("model_timings", ()):
                    timing = response.predict.prediction.stats.model_timings.add()
                    timing.name = item["name"]
                    timing.milliseconds = item["milliseconds"]
            except ValueError as error:
                response = self.error(request_id, 3, str(error),
                                      field="predict")
            except WamError as error:
                response = self._native_error(request_id, error)
        elif kind == "reset":
            try:
                await self.service.execute(self.session.reset)
                response = self.response(request_id)
                response.reset.SetInParent()
            except WamError as error:
                response = self._native_error(request_id, error)
        elif kind == "close":
            response = self.response(request_id)
            response.close.SetInParent()
            self.closed = True
        else:
            response = self.error(
                request_id, 1, f"payload {kind!r} is invalid after Hello",
                field="envelope", fatal=True)
        self.expected_request_id += 1
        return response

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self.closed = True
