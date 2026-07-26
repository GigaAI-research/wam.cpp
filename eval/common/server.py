"""Environment-independent wam.cpp WebSocket/Protobuf server."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import io
import logging
from pathlib import Path

import numpy as np
from google.protobuf import json_format

from common.native import NativeError, NativeModel
from common.language import create_language_provider
from common.rpc import decode_tensor, encode_tensor, load_types


LOG = logging.getLogger("wam.server")
PROTOCOL_MAJOR = 0
PROTOCOL_MINOR = 5


@dataclass(frozen=True)
class EnvironmentContract:
    environment_id: str
    image_roles: tuple[str, ...]
    state_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    state_dim: int
    action_dim: int
    action_representation: str
    action_frame: str
    gripper_encoding: str


def check_environment(metadata, contract):
    spec = metadata["policy_spec"]
    roles = tuple(view["role"] for view in spec["images"]["views"])
    failures = []
    if set(roles) != set(contract.image_roles):
        failures.append(f"image roles {roles} != {contract.image_roles}")
    if spec["state"]["real_dim"] != contract.state_dim:
        failures.append(f"state dim {spec['state']['real_dim']} != {contract.state_dim}")
    if tuple(spec["state"]["fields"]) != contract.state_fields:
        failures.append("state field order differs from environment contract")
    action = spec["action"]
    if action["real_dim"] != contract.action_dim:
        failures.append(f"action dim {action['real_dim']} != {contract.action_dim}")
    if action["representation"] != contract.action_representation:
        failures.append(f"action representation {action['representation']} != "
                        f"{contract.action_representation}")
    if action["frame"] != contract.action_frame:
        failures.append(f"action frame {action['frame']} != {contract.action_frame}")
    if tuple(action["fields"]) != contract.action_fields:
        failures.append("action field order differs from environment contract")
    if action["gripper"] != contract.gripper_encoding:
        failures.append(f"gripper encoding {action['gripper']} != "
                        f"{contract.gripper_encoding}")
    if failures:
        raise ValueError(f"{contract.environment_id} contract incompatible: " + "; ".join(failures))


class WamServer:
    def __init__(self, model, language_provider, descriptor, contract, host, port,
                 random_seed=0, max_message_bytes=64 << 20):
        self.model = model
        self.language_provider = language_provider
        self.types = load_types(descriptor)
        self.contract = contract
        self.host = host
        self.port = port
        self.random_seed = random_seed
        self.max_message_bytes = max_message_bytes
        self.inference_lock = (None if model.metadata["capabilities"].get(
            "concurrent_sessions", False) else asyncio.Lock())
        self.model_info = self.types["ModelInfo"]()
        json_format.ParseDict(model.metadata, self.model_info)

    def _response(self, request_id):
        response = self.types["ServerEnvelope"]()
        response.request_id = request_id
        return response

    def _error(self, request_id, code, field, message, fatal):
        response = self._response(request_id)
        response.error.code = code
        response.error.field = field
        response.error.message = message
        response.error.fatal = fatal
        return response

    @staticmethod
    def _native_error(request_id, error):
        field = error.details[0]["field"] if error.details else "predict"
        if error.code == 1:
            return request_id, 3, field, str(error), False
        if error.code == 3:
            return request_id, 4, field, str(error), False
        if error.code in (4, 6):
            return request_id, 2, field, str(error), True
        return request_id, 5, field, str(error), True

    @staticmethod
    def _decode_images(observation):
        from PIL import Image as PilImage
        result = []
        names = set()
        for wire in observation.images:
            if not wire.name or wire.name in names:
                raise ValueError(f"duplicate or empty image role: {wire.name!r}")
            names.add(wire.name)
            if wire.encoding == 1:
                expected = int(wire.width) * int(wire.height) * 3
                if len(wire.data) != expected:
                    raise ValueError(f"image {wire.name} payload has {len(wire.data)} bytes, expected {expected}")
                pixels = np.frombuffer(wire.data, dtype=np.uint8).reshape(
                    wire.height, wire.width, 3).copy()
            elif wire.encoding in (2, 3):
                pixels = np.asarray(PilImage.open(io.BytesIO(wire.data)).convert("RGB"))
                if pixels.shape[:2] != (wire.height, wire.width):
                    raise ValueError(f"decoded image {wire.name} shape {pixels.shape[:2]} "
                                     f"!= {(wire.height, wire.width)}")
            else:
                raise ValueError(f"image {wire.name} has unsupported encoding {wire.encoding}")
            result.append({"name": wire.name, "data": pixels})
        return result

    def _predict_blocking(self, session, request):
        observation = request.observation
        images = self._decode_images(observation)
        state = decode_tensor(observation.state)
        if state.dtype != np.float32 or state.ndim != 1:
            raise ValueError(f"state must be rank-1 F32, got {state.dtype} {state.shape}")
        language = self.language_provider.prepare(observation.instruction)
        noise = decode_tensor(request.action_noise) if request.HasField("action_noise") else None
        action, stats = session.predict(
            images, state, language.values, language.attention_mask, noise)
        stats["preprocess_milliseconds"] += language.preprocess_milliseconds
        stats["model_text_milliseconds"] += language.model_milliseconds
        stats["model_milliseconds"] += language.model_milliseconds
        stats["total_milliseconds"] += (
            language.preprocess_milliseconds + language.model_milliseconds)
        if language.model_milliseconds:
            stats["model_timings"].insert(0, {
                "name": "language_encoder",
                "milliseconds": language.model_milliseconds,
            })
        return action, stats

    async def _predict(self, session, request):
        if self.inference_lock is None:
            return await asyncio.to_thread(self._predict_blocking, session, request)
        async with self.inference_lock:
            return await asyncio.to_thread(self._predict_blocking, session, request)

    async def _reset(self, session):
        if self.inference_lock is None:
            return await asyncio.to_thread(session.reset)
        async with self.inference_lock:
            return await asyncio.to_thread(session.reset)

    async def handler(self, websocket):
        session = None
        expected_id = 0
        fatal = False
        try:
            while True:
                payload = await websocket.recv()
                if isinstance(payload, str):
                    response = self._error(expected_id, 1, "frame",
                                           "text WebSocket frames are not accepted", True)
                    await websocket.send(response.SerializeToString())
                    fatal = True
                    break
                if len(payload) > self.max_message_bytes:
                    response = self._error(expected_id, 1, "frame",
                                           "WebSocket message exceeds configured limit", True)
                    await websocket.send(response.SerializeToString())
                    fatal = True
                    break
                request = self.types["ClientEnvelope"]()
                try:
                    request.ParseFromString(payload)
                except Exception as error:
                    response = self._error(expected_id, 1, "envelope",
                                           f"invalid protobuf: {error}", True)
                    await websocket.send(response.SerializeToString())
                    fatal = True
                    break
                kind = request.WhichOneof("payload")
                if request.request_id != expected_id:
                    response = self._error(request.request_id, 1, "request_id",
                                           f"expected {expected_id}, got {request.request_id}", True)
                    await websocket.send(response.SerializeToString())
                    fatal = True
                    break
                if session is None:
                    if kind != "hello" or expected_id != 0:
                        response = self._error(request.request_id, 1, "hello",
                                               "the first message must be HelloRequest", True)
                        await websocket.send(response.SerializeToString())
                        fatal = True
                        break
                    hello = request.hello
                    if hello.protocol_major != PROTOCOL_MAJOR:
                        response = self._error(0, 2, "protocol_major",
                                               f"server protocol major is {PROTOCOL_MAJOR}", True)
                        await websocket.send(response.SerializeToString())
                        fatal = True
                        break
                    if hello.environment_id != self.contract.environment_id:
                        response = self._error(0, 2, "environment_id",
                                               f"server environment is {self.contract.environment_id}", True)
                        await websocket.send(response.SerializeToString())
                        fatal = True
                        break
                    session = self.model.create_session(self.random_seed)
                    response = self._response(0)
                    response.hello.model_info.CopyFrom(self.model_info)
                    await websocket.send(response.SerializeToString())
                    expected_id = 1
                    continue
                if kind == "predict":
                    try:
                        action, stats = await self._predict(session, request.predict)
                        response = self._response(request.request_id)
                        encode_tensor(response.predict.prediction.action, action, np.float32)
                        timings = stats.get("model_timings", ())
                        for name, value in stats.items():
                            if name != "model_timings":
                                setattr(response.predict.prediction.stats,
                                        name, value)
                        for item in timings:
                            timing = response.predict.prediction.stats.model_timings.add()
                            timing.name = item["name"]
                            timing.milliseconds = item["milliseconds"]
                        LOG.info("request=%d action_shape=%s total_ms=%.3f",
                                 request.request_id, action.shape,
                                 stats["total_milliseconds"])
                    except ValueError as error:
                        response = self._error(request.request_id, 3, "predict",
                                               str(error), False)
                    except NativeError as error:
                        response = self._error(*self._native_error(
                            request.request_id, error))
                        fatal = response.error.fatal
                    await websocket.send(response.SerializeToString())
                elif kind == "reset":
                    try:
                        await self._reset(session)
                        response = self._response(request.request_id)
                        response.reset.SetInParent()
                    except NativeError as error:
                        response = self._error(request.request_id, 5, "session", str(error), True)
                        fatal = True
                    await websocket.send(response.SerializeToString())
                elif kind == "close":
                    response = self._response(request.request_id)
                    response.close.SetInParent()
                    await websocket.send(response.SerializeToString())
                    break
                else:
                    response = self._error(request.request_id, 1, "envelope",
                                           f"payload {kind!r} is not valid after Hello", True)
                    await websocket.send(response.SerializeToString())
                    fatal = True
                expected_id += 1
                if fatal:
                    break
        except Exception as error:
            if "ConnectionClosed" not in type(error).__name__:
                LOG.exception("connection failed")
        finally:
            if session is not None:
                session.close()
            if fatal:
                await websocket.close(code=1002)

    async def serve(self):
        from websockets.asyncio.server import serve
        async with serve(self.handler, self.host, self.port, compression=None,
                         max_size=self.max_message_bytes,
                         ping_interval=120, ping_timeout=600) as server:
            LOG.info("ready ws://%s:%d environment=%s architecture=%s profile=%s",
                     self.host, self.port, self.contract.environment_id,
                     self.model.metadata["architecture"],
                     self.model.metadata["policy_spec"]["profile"])
            await server.serve_forever()


def serve_main(contract, argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--language-python-root", type=Path)
    parser.add_argument("--language-device")
    parser.add_argument("--language-cache-capacity", type=int, default=32)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18060)
    parser.add_argument("--backend", choices=("automatic", "cuda", "cpu-metadata"), default="cuda")
    parser.add_argument("--precision", choices=("automatic", "f32", "f16", "bf16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--prompt-cache-capacity", type=int, default=4)
    parser.add_argument("--language-mode",
                        choices=("automatic", "tokens", "external_embedding"),
                        default="automatic")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    model = NativeModel(args.library, args.model, args.backend, args.precision,
                        args.device, args.prompt_cache_capacity,
                        args.language_mode)
    try:
        check_environment(model.metadata, contract)
        language_device = args.language_device or f"cuda:{args.device}"
        provider = create_language_provider(
            model.metadata, tokenizer_path=args.tokenizer,
            encoder_path=args.text_encoder, python_root=args.language_python_root,
            device=language_device, cache_capacity=args.language_cache_capacity)
        asyncio.run(WamServer(model, provider, args.descriptor, contract,
                              args.host, args.port, args.random_seed).serve())
    finally:
        model.close()


def main(argv=None):
    raise SystemExit("use an environment-specific run_*_server.py entry point")
