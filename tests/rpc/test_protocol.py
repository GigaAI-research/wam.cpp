#!/usr/bin/env python3
"""wam.rpc.v06 service-core and WebSocket transport contract tests."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import time

import numpy as np

from wam import ErrorCode, Prediction, SessionConfig, WamError
from wam.adapters import EnvironmentContract, check_environment
from wam.language import PreparedLanguage
from wam.remote import (Client as RpcClient, RemoteError, decode_tensor,
                        encode_tensor)
from wam.serving import ServiceCore, WebSocketTransport


def metadata(concurrent=True):
    roles = ("camera_high", "camera_left_wrist", "camera_right_wrist")
    normalization = {"kind": "none", "clip": False, "epsilon": 0}
    return {
        "runtime_version": {"major": 0, "minor": 6, "patch": 0},
        "protocol_version": {"major": 0, "minor": 6},
        "architecture": "fake", "artifact_policy": "robotwin-test",
        "artifact_bytes": 1,
        "backend": "cuda", "compute_precision": "bf16",
        "language_mode": "tokens",
        "capabilities": {"action": True, "raw_images": True,
                         "token_input": True, "explicit_action_noise": True,
                         "concurrent_sessions": concurrent},
        "policy_spec": {
            "artifact_schema_version": 2, "profile": "robotwin-test",
            "checkpoint_revision": "test", "training_dataset": "test",
            "embodiment": "test",
            "images": {"views": [
                {"role": role, "target_height": 2, "target_width": 2,
                 "resize": "stretch", "interpolation": "bilinear",
                 "antialias": False} for role in roles],
                "composition": {"kind": "canvas", "height": 2, "width": 2,
                                "placements": []},
                "color_space": "rgb", "pixel_range": "minus_one_to_one",
                "tensor_layout": "chw", "resample_boundary": "truncate"},
            "state": {"fields": [str(i) for i in range(14)], "real_dim": 14,
                      "model_dim": 14, "pad_value": 0,
                      "normalization": normalization},
            "language": {"input_mode": "tokens", "prompt_template": "{task}",
                         "tokenizer_family": "fake", "tokenizer_revision": "fake",
                         "max_tokens": 4, "text_encoder_in_artifact": True,
                         "padding_side": "right", "truncation_side": "right",
                         "attention_mask_required": True,
                         "special_token_ids": []},
            "action": {"horizon": 3, "real_dim": 14, "model_dim": 14,
                       "fields": [str(i) for i in range(14)],
                       "representation": "joint_position", "frame": "controller",
                       "gripper": "continuous", "normalization": normalization,
                       "recovery": {"kind": "identity",
                                    "reference_state_indices": []}}
        }
    }


class FakeLanguageProvider:
    def __init__(self, external=False): self.external = external

    def prepare(self, _instruction):
        if _instruction == "invalid":
            raise ValueError("instruction fixture is invalid")
        if self.external:
            return PreparedLanguage(
                np.zeros((4, 8), np.uint16), np.ones(4, np.int32),
                preprocess_milliseconds=0.5, model_milliseconds=2.0)
        return PreparedLanguage(np.ones(4, np.int32), np.ones(4, np.int32),
                                preprocess_milliseconds=0.5)


class FakeSession:
    def __init__(self, owner, identifier):
        self.owner, self.identifier, self.resets = owner, identifier, 0

    def predict(self, images, state, *, token_ids=None, attention_mask=None,
                embedding=None, embedding_attention_mask=None,
                action_noise=None):
        assert len(images) == 3 and state.shape == (14,)
        if state[0] == 9:
            raise WamError(
                ErrorCode.INVALID_ARGUMENT, "state fixture is invalid",
                [{"field": "observation.state", "reason": "fixture"}])
        if self.owner.metadata["language_mode"] == "external_embedding":
            assert embedding.shape == (4, 8) and embedding.dtype == np.uint16
            assert embedding_attention_mask.shape == (4,)
        else:
            assert token_ids.shape == attention_mask.shape == (4,)
        assert action_noise is None
        self.owner.active += 1
        self.owner.max_active = max(self.owner.max_active, self.owner.active)
        time.sleep(0.01)
        self.owner.active -= 1
        action = np.full((3, 14), self.resets, np.float32)
        names = ("preprocess_milliseconds", "model_milliseconds",
                 "model_vision_milliseconds", "model_text_milliseconds",
                 "model_prefill_milliseconds", "model_decode_milliseconds",
                 "postprocess_milliseconds", "total_milliseconds",
                 "peak_device_memory_bytes")
        stats = {name: 0 for name in names}
        stats["model_timings"] = [
            {"name": "decode.fake", "milliseconds": 1.25}]
        return Prediction(action, stats)

    def reset(self): self.resets += 1
    def close(self): self.owner.closed.append(self.identifier)


class FakeModel:
    def __init__(self, external=False, concurrent=True):
        self.metadata = metadata(concurrent)
        self.created, self.closed = 0, []
        self.active = self.max_active = 0
        if external:
            self.metadata["language_mode"] = "external_embedding"
            self.metadata["policy_spec"]["language"]["input_mode"] = "embedding"

    def create_session(self, config):
        assert isinstance(config, SessionConfig)
        self.created += 1
        return FakeSession(self, self.created)


def contract():
    fields = tuple(str(i) for i in range(14))
    return EnvironmentContract("robotwin",
        ("camera_high", "camera_left_wrist", "camera_right_wrist"),
        fields, fields, 14, 14, "joint_position", "controller", "continuous")


@asynccontextmanager
async def running_server(descriptor, external=False, concurrent=True):
    from websockets.asyncio.server import serve
    model = FakeModel(external, concurrent)
    service = ServiceCore(model, FakeLanguageProvider(external), descriptor,
                          contract())
    app = WebSocketTransport(service, "127.0.0.1", 0)
    async with serve(app.handler, "127.0.0.1", 0, compression=None) as server:
        yield (model,
               f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}",
               service.types, service)


async def exchange(socket, message, response_type):
    await socket.send(message.SerializeToString())
    payload = await socket.recv()
    assert isinstance(payload, bytes)
    response = response_type(); response.ParseFromString(payload)
    return response


def hello(types):
    value = types["ClientEnvelope"]()
    value.hello.protocol_major, value.hello.protocol_minor = 0, 6
    value.hello.environment_id = "robotwin"
    return value


def predict(types, request_id, *, instruction="test", state_first=0):
    value = types["ClientEnvelope"](); value.request_id = request_id
    for role in ("camera_high", "camera_left_wrist", "camera_right_wrist"):
        image = value.predict.observation.images.add()
        image.name, image.encoding, image.width, image.height = role, 1, 2, 2
        image.data = bytes(12)
    state = np.zeros(14, np.float32); state[0] = state_first
    encode_tensor(value.predict.observation.state, state, np.float32)
    value.predict.observation.instruction = instruction
    return value


async def run(descriptor):
    from websockets.asyncio.client import connect
    try:
        RpcClient("127.0.0.1", 1, descriptor, "")
    except ValueError:
        pass
    else:
        raise AssertionError("RPC environment id must be explicit")
    check_environment(metadata(), contract())
    incompatible = metadata()
    incompatible["policy_spec"]["state"]["fields"][0] = "wrong"
    try:
        check_environment(incompatible, contract())
    except ValueError:
        pass
    else:
        raise AssertionError("field-order mismatch must fail compatibility")
    async with running_server(descriptor) as (model, url, types, service):
        response_type = types["ServerEnvelope"]
        core_connection = service.connection()
        assert (await core_connection.handle(hello(types))).WhichOneof(
            "payload") == "hello"
        core_prediction = await core_connection.handle(predict(types, 1))
        assert np.array_equal(
            decode_tensor(core_prediction.predict.prediction.action),
            np.zeros((3, 14), np.float32))
        core_connection.close()
        async with connect(url) as first:
            assert (await exchange(first, hello(types), response_type)).WhichOneof("payload") == "hello"
            response = await exchange(first, predict(types, 1), response_type)
            assert np.all(decode_tensor(response.predict.prediction.action) == 0)
            assert len(response.predict.prediction.stats.model_timings) == 1
            timing = response.predict.prediction.stats.model_timings[0]
            assert timing.name == "decode.fake" and timing.milliseconds == 1.25
            invalid = await exchange(
                first, predict(types, 2, instruction="invalid"), response_type)
            assert not invalid.error.fatal
            assert invalid.error.details[0].field == "predict"
            reset = types["ClientEnvelope"](); reset.request_id = 3; reset.reset.SetInParent()
            assert (await exchange(first, reset, response_type)).WhichOneof("payload") == "reset"
            response = await exchange(first, predict(types, 4), response_type)
            assert np.all(decode_tensor(response.predict.prediction.action) == 1)
            close = types["ClientEnvelope"](); close.request_id = 5
            close.close.SetInParent()
            assert (await exchange(first, close, response_type)).WhichOneof("payload") == "close"
        await asyncio.sleep(0.01)
        assert model.closed == [1, 2]

        async with connect(url) as second, connect(url) as third:
            await exchange(second, hello(types), response_type)
            await exchange(third, hello(types), response_type)
            one = await exchange(second, predict(types, 1), response_type)
            two = await exchange(third, predict(types, 1), response_type)
            assert np.all(decode_tensor(one.predict.prediction.action) == 0)
            assert np.all(decode_tensor(two.predict.prediction.action) == 0)

        port = int(url.rsplit(":", 1)[1])

        def client_prediction():
            with RpcClient("127.0.0.1", port, descriptor, "robotwin",
                           connect_timeout=2) as client:
                images = [{"name": role, "data": np.zeros((2, 2, 3), np.uint8)}
                          for role in ("camera_high", "camera_left_wrist",
                                       "camera_right_wrist")]
                try:
                    client.predict(images, np.zeros(14, np.float32), "invalid")
                except RemoteError as error:
                    assert not error.fatal
                    assert error.details[0]["field"] == "predict"
                else:
                    raise AssertionError("remote Client did not map predict error")
                return client.predict(images, np.zeros(14, np.float32), "test")[0]

        assert np.array_equal(
            await asyncio.to_thread(client_prediction),
            np.zeros((3, 14), np.float32))

        def incompatible_client():
            with RpcClient("127.0.0.1", port, descriptor, "libero",
                           connect_timeout=2):
                pass

        try:
            await asyncio.to_thread(incompatible_client)
        except RemoteError as error:
            assert error.fatal and error.details[0]["field"] == "environment_id"
        else:
            raise AssertionError("remote Client accepted an incompatible environment")

        async with connect(url) as bad:
            response = await exchange(bad, predict(types, 0), response_type)
            assert response.error.fatal and response.error.details[0].field == "hello"
        async with connect(url) as bad:
            await exchange(bad, hello(types), response_type)
            response = await exchange(bad, predict(types, 2), response_type)
            assert response.error.fatal and response.error.details[0].field == "request_id"
        async with connect(url) as bad:
            await bad.send("text")
            response = response_type(); response.ParseFromString(await bad.recv())
            assert response.error.fatal and response.error.details[0].field == "frame"
        async with connect(url) as bad:
            await bad.send(b"\x80")
            response = response_type(); response.ParseFromString(await bad.recv())
            assert response.error.fatal
            assert response.error.details[0].field == "envelope"
        async with connect(url) as recoverable:
            await exchange(recoverable, hello(types), response_type)
            response = await exchange(
                recoverable, predict(types, 1, state_first=9), response_type)
            assert not response.error.fatal and response.error.code == 3
            assert response.error.details[0].field == "observation.state"
            response = await exchange(
                recoverable, predict(types, 2), response_type)
            assert response.WhichOneof("payload") == "predict"
        await asyncio.sleep(0.01)
    assert sorted(model.closed) == list(range(1, model.created + 1))

    async with running_server(
            descriptor, external=True) as (_, url, types, _service):
        async with connect(url) as socket:
            await exchange(socket, hello(types), types["ServerEnvelope"])
            response = await exchange(
                socket, predict(types, 1), types["ServerEnvelope"])
            stats = response.predict.prediction.stats
            assert stats.preprocess_milliseconds == 0.5
            assert stats.model_text_milliseconds == 2.0
            assert stats.model_milliseconds == 2.0
            assert stats.total_milliseconds == 2.5
            assert stats.model_timings[0].name == "language_encoder"

    async with running_server(
            descriptor, concurrent=False) as (model, url, types, _service):
        async with connect(url) as first, connect(url) as second:
            await exchange(first, hello(types), types["ServerEnvelope"])
            await exchange(second, hello(types), types["ServerEnvelope"])
            await asyncio.gather(
                exchange(first, predict(types, 1), types["ServerEnvelope"]),
                exchange(second, predict(types, 1), types["ServerEnvelope"]))
        assert model.max_active == 1


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--descriptor", required=True)
    args = parser.parse_args(); asyncio.run(run(args.descriptor))
    print("wam 0.6 WebSocket/Protobuf protocol: PASS")


if __name__ == "__main__": main()
