#!/usr/bin/env python3
"""Slice 7 WebSocket/Protobuf state-machine contract tests."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager

import numpy as np

from common.rpc import RpcClient, decode_tensor, encode_tensor
from common.server import EnvironmentContract, WamServer, check_environment


def metadata():
    roles = ("camera_high", "camera_left_wrist", "camera_right_wrist")
    normalization = {"kind": "none", "clip": False, "epsilon": 0}
    return {
        "runtime_version": {"major": 0, "minor": 5, "patch": 0},
        "protocol_version": {"major": 0, "minor": 5},
        "architecture": "fake", "artifact_policy": "robotwin-test",
        "artifact_sha256": "", "artifact_bytes": 1,
        "backend": "cuda", "compute_precision": "bf16",
        "language_mode": "tokens",
        "capabilities": {"action": True, "raw_images": True,
                         "token_input": True, "explicit_action_noise": True,
                         "concurrent_sessions": True},
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


class FakeTokenizer:
    def __call__(self, _text, **kwargs):
        size = kwargs["max_length"]
        return {"input_ids": [1] * size, "attention_mask": [1] * size}


class FakeSession:
    def __init__(self, owner, identifier):
        self.owner, self.identifier, self.resets = owner, identifier, 0

    def predict(self, images, state, token_ids, attention_mask, noise):
        assert len(images) == 3 and state.shape == (14,)
        assert token_ids.shape == attention_mask.shape == (4,)
        assert noise is None
        action = np.full((3, 14), self.identifier + self.resets, np.float32)
        names = ("preprocess_milliseconds", "model_milliseconds",
                 "model_vision_milliseconds", "model_text_milliseconds",
                 "model_prefill_milliseconds", "model_decode_milliseconds",
                 "postprocess_milliseconds", "total_milliseconds",
                 "peak_device_memory_bytes")
        stats = {name: 0 for name in names}
        stats["model_timings"] = [
            {"name": "decode.fake", "milliseconds": 1.25}]
        return action, stats

    def reset(self): self.resets += 1
    def close(self): self.owner.closed.append(self.identifier)


class FakeModel:
    def __init__(self):
        self.metadata, self.created, self.closed = metadata(), 0, []

    def create_session(self, _seed):
        self.created += 1
        return FakeSession(self, self.created)


def contract():
    fields = tuple(str(i) for i in range(14))
    return EnvironmentContract("robotwin",
        ("camera_high", "camera_left_wrist", "camera_right_wrist"),
        fields, fields, 14, 14, "joint_position", "controller", "continuous")


@asynccontextmanager
async def running_server(descriptor):
    from websockets.asyncio.server import serve
    model = FakeModel()
    app = WamServer(model, FakeTokenizer(), descriptor, contract(), "127.0.0.1", 0)
    async with serve(app.handler, "127.0.0.1", 0, compression=None) as server:
        yield model, f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}", app.types


async def exchange(socket, message, response_type):
    await socket.send(message.SerializeToString())
    payload = await socket.recv()
    assert isinstance(payload, bytes)
    response = response_type(); response.ParseFromString(payload)
    return response


def hello(types):
    value = types["ClientEnvelope"]()
    value.hello.protocol_major, value.hello.protocol_minor = 0, 5
    value.hello.environment_id = "robotwin"
    return value


def predict(types, request_id):
    value = types["ClientEnvelope"](); value.request_id = request_id
    for role in ("camera_high", "camera_left_wrist", "camera_right_wrist"):
        image = value.predict.observation.images.add()
        image.name, image.encoding, image.width, image.height = role, 1, 2, 2
        image.data = bytes(12)
    encode_tensor(value.predict.observation.state, np.zeros(14, np.float32), np.float32)
    value.predict.observation.instruction = "test"
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
    async with running_server(descriptor) as (model, url, types):
        response_type = types["ServerEnvelope"]
        async with connect(url) as first:
            assert (await exchange(first, hello(types), response_type)).WhichOneof("payload") == "hello"
            response = await exchange(first, predict(types, 1), response_type)
            assert np.all(decode_tensor(response.predict.prediction.action) == 1)
            assert len(response.predict.prediction.stats.model_timings) == 1
            timing = response.predict.prediction.stats.model_timings[0]
            assert timing.name == "decode.fake" and timing.milliseconds == 1.25
            reset = types["ClientEnvelope"](); reset.request_id = 2; reset.reset.SetInParent()
            assert (await exchange(first, reset, response_type)).WhichOneof("payload") == "reset"
            response = await exchange(first, predict(types, 3), response_type)
            assert np.all(decode_tensor(response.predict.prediction.action) == 2)
        await asyncio.sleep(0.01)
        assert model.closed == [1]

        async with connect(url) as second, connect(url) as third:
            await exchange(second, hello(types), response_type)
            await exchange(third, hello(types), response_type)
            one = await exchange(second, predict(types, 1), response_type)
            two = await exchange(third, predict(types, 1), response_type)
            assert np.all(decode_tensor(one.predict.prediction.action) == 2)
            assert np.all(decode_tensor(two.predict.prediction.action) == 3)

        async with connect(url) as bad:
            response = await exchange(bad, predict(types, 0), response_type)
            assert response.error.fatal and response.error.field == "hello"
        async with connect(url) as bad:
            await exchange(bad, hello(types), response_type)
            response = await exchange(bad, predict(types, 2), response_type)
            assert response.error.fatal and response.error.field == "request_id"
        async with connect(url) as bad:
            await bad.send("text")
            response = response_type(); response.ParseFromString(await bad.recv())
            assert response.error.fatal and response.error.field == "frame"
        await asyncio.sleep(0.01)
    assert sorted(model.closed) == list(range(1, model.created + 1))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--descriptor", required=True)
    args = parser.parse_args(); asyncio.run(run(args.descriptor))
    print("wam 0.5 WebSocket/Protobuf protocol: PASS")


if __name__ == "__main__": main()
