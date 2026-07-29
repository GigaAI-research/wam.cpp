"""WebSocket client for the wam.rpc.v06 protocol."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np


PACKAGE = "wam.rpc.v06"
MESSAGE_NAMES = (
    "ClientEnvelope", "ServerEnvelope", "ModelInfo", "PolicySpec",
    "Tensor", "Image", "Prediction")


class RemoteError(RuntimeError):
    def __init__(self, code, message, details, fatal=False):
        super().__init__(message)
        self.code = int(code)
        self.details = details
        self.fatal = bool(fatal)


def load_types(descriptor):
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.ParseFromString(Path(descriptor).read_bytes())
    if not descriptor_set.file:
        raise ValueError(f"empty protobuf descriptor: {descriptor}")
    pool = descriptor_pool.DescriptorPool()
    pending = list(descriptor_set.file)
    while pending:
        deferred = []
        for item in pending:
            try:
                pool.Add(item)
            except TypeError:
                deferred.append(item)
        if len(deferred) == len(pending):
            raise ValueError("protobuf descriptor dependencies cannot be resolved")
        pending = deferred
    descriptors = {
        name: pool.FindMessageTypeByName(f"{PACKAGE}.{name}")
        for name in MESSAGE_NAMES}
    if hasattr(message_factory, "GetMessageClass"):
        return {name: message_factory.GetMessageClass(value)
                for name, value in descriptors.items()}
    factory = message_factory.MessageFactory(pool)
    return {name: factory.GetPrototype(value)
            for name, value in descriptors.items()}


def encode_tensor(message, value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    message.dtype = {np.dtype(np.float32): 1, np.dtype(np.int32): 2,
                     np.dtype(np.uint8): 3}[array.dtype]
    message.shape.extend(array.shape)
    message.data = array.tobytes()
    return message


def decode_tensor(message):
    dtype = {1: np.dtype("<f4"), 2: np.dtype("<i4"),
             3: np.dtype("u1")}.get(message.dtype)
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype enum {message.dtype}")
    shape = tuple(int(value) for value in message.shape)
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(message.data) != expected:
        raise ValueError(f"tensor payload has {len(message.data)} bytes, expected {expected}")
    return np.frombuffer(message.data, dtype=dtype).reshape(shape).copy()


class Client:
    def __init__(self, host, port, descriptor, environment_id,
                 connect_timeout=600):
        if not environment_id:
            raise ValueError("environment_id must not be empty")
        self.url = f"ws://{host}:{port}"
        self.types = load_types(descriptor)
        self.environment_id = environment_id
        self.connect_timeout = connect_timeout
        self.socket = None
        self.request_id = 0
        self.model_info = None

    def connect(self):
        if self.socket is not None:
            return self.model_info
        from websockets.sync.client import connect
        deadline = time.monotonic() + self.connect_timeout
        while True:
            try:
                self.socket = connect(self.url, compression=None,
                                      max_size=64 << 20)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"server did not become ready: {self.url}")
                time.sleep(2)
        request = self.types["ClientEnvelope"]()
        request.request_id = 0
        request.hello.protocol_major = 0
        request.hello.protocol_minor = 6
        request.hello.environment_id = self.environment_id
        response = self._exchange(request)
        if response.WhichOneof("payload") != "hello":
            self._raise_response(response)
        self.model_info = response.hello.model_info
        self.request_id = 1
        return self.model_info

    def _exchange(self, request):
        if self.socket is None:
            raise RuntimeError("RPC client is not connected")
        self.socket.send(request.SerializeToString())
        payload = self.socket.recv()
        if isinstance(payload, str):
            raise RuntimeError("server returned a text frame")
        response = self.types["ServerEnvelope"]()
        response.ParseFromString(payload)
        if response.request_id != request.request_id:
            raise RuntimeError(f"response id {response.request_id} != {request.request_id}")
        return response

    def _raise_response(self, response):
        if response.WhichOneof("payload") == "error":
            error = RemoteError(
                response.error.code, response.error.message,
                [{"field": item.field, "reason": item.reason}
                 for item in response.error.details],
                response.error.fatal)
            if error.fatal and self.socket is not None:
                self.socket.close()
                self.socket = None
            raise error
        raise RuntimeError(f"unexpected RPC response: {response.WhichOneof('payload')}")

    def predict(self, images, state, instruction, action_noise=None):
        if self.socket is None:
            self.connect()
        request = self.types["ClientEnvelope"]()
        request.request_id = self.request_id
        for item in images:
            image = request.predict.observation.images.add()
            image.name = item["name"]
            pixels = np.ascontiguousarray(item["data"], dtype=np.uint8)
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError(f"{image.name} must be HWC RGB, got {pixels.shape}")
            image.encoding = 1
            image.height, image.width = pixels.shape[:2]
            image.data = pixels.tobytes()
        encode_tensor(request.predict.observation.state, state, np.float32)
        request.predict.observation.instruction = instruction
        if action_noise is not None:
            encode_tensor(request.predict.action_noise, action_noise, np.float32)
        response = self._exchange(request)
        self.request_id += 1
        if response.WhichOneof("payload") != "predict":
            self._raise_response(response)
        return decode_tensor(response.predict.prediction.action), response.predict.prediction.stats

    def reset(self):
        request = self.types["ClientEnvelope"]()
        request.request_id = self.request_id
        request.reset.SetInParent()
        response = self._exchange(request)
        self.request_id += 1
        if response.WhichOneof("payload") != "reset":
            self._raise_response(response)

    def close(self):
        if self.socket is None:
            return
        try:
            request = self.types["ClientEnvelope"]()
            request.request_id = self.request_id
            request.close.SetInParent()
            response = self._exchange(request)
            if response.WhichOneof("payload") != "close":
                self._raise_response(response)
        finally:
            self.socket.close()
            self.socket = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()


RpcClient = Client
