#!/usr/bin/env python3
"""Exercise WamService with dynamic classes loaded from the WAM descriptor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import grpc

from wam_proto import load_types


def discover_descriptor(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit.resolve()
    candidates = []
    environment = os.environ.get("WAM_PROTO_DESCRIPTOR")
    if environment:
        candidates.append(Path(environment).expanduser())
    script = Path(__file__).resolve()
    candidates.extend([
        script.parent / "wam.desc",
        script.parent.parent / "share" / "wam" / "proto" / "wam.desc",
    ])
    return next((path.resolve() for path in candidates if path.is_file()), None)


def compile_fallback(proto: Path, protoc: Path, descriptor: Path) -> None:
    if not proto.is_file():
        raise FileNotFoundError(f"Proto source not found: {proto}")
    if not protoc.is_file():
        raise FileNotFoundError(f"protoc not found: {protoc}")
    subprocess.run([
        str(protoc),
        f"--proto_path={proto.parent}",
        f"--descriptor_set_out={descriptor}",
        proto.name,
    ], cwd=proto.parent, check=True)


def resolve_types(args, parser):
    descriptor = discover_descriptor(args.descriptor)
    if descriptor is not None and descriptor.is_file():
        return load_types(descriptor)
    if args.descriptor is not None:
        parser.error(f"Proto descriptor not found: {args.descriptor}")
    if (args.proto is None) != (args.protoc is None):
        parser.error("legacy fallback requires both --proto and --protoc")
    if args.proto is not None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "wam.desc"
            compile_fallback(args.proto.resolve(), args.protoc.resolve(), generated)
            return load_types(generated)
    parser.error(
        "no WAM Proto descriptor found; pass --descriptor, set "
        "WAM_PROTO_DESCRIPTOR, or use legacy --proto/--protoc"
    )


def rpc(channel, types, name, request, timeout=300):
    response_type = types[name + "Response"]
    call = channel.unary_unary(
        f"/wam.rpc.v1.WamService/{name}",
        request_serializer=lambda value: value.SerializeToString(deterministic=True),
        response_deserializer=response_type.FromString)
    try:
        return call(request, timeout=timeout)
    except grpc.RpcError as error:
        detail = None
        for key, value in error.trailing_metadata() or ():
            if key == "wam-error-bin":
                detail = types["Error"].FromString(value)
        if detail is not None:
            fields = ", ".join(f"{item.field}: {item.reason}" for item in detail.details)
            raise RuntimeError(f"{error.code().name}: {detail.message} [{fields}]") from error
        raise


def tensor(message, value, base: Path, default_shape, default_layout):
    message.dtype = {"u8": 1, "i32": 2, "f32": 3, "float32": 3,
                     "bf16": 4}[value.get("dtype", "f32")]
    message.shape.extend(value.get("shape", default_shape))
    message.layout = value.get("layout", default_layout)
    message.byte_order = 1 if message.dtype != 1 else 3
    if "path" in value:
        message.data = (base / value["path"]).read_bytes()
    else:
        numbers = value["values"]
        if message.dtype == 3:
            message.data = b"".join(struct.pack("<f", item) for item in numbers)
        elif message.dtype == 2:
            message.data = b"".join(struct.pack("<i", item) for item in numbers)
        else:
            raise ValueError("inline values support F32/I32 only")


def build_predict(types, path: Path, session_id: str, request_id: int):
    root = json.loads(path.read_text())
    base = path.parent
    request = types["PredictRequest"](session_id=session_id, request_id=request_id)
    for value in root["images"]:
        image = request.inputs.images.add()
        image.name = value.get("logical_name", value.get("name", ""))
        encoding = value.get("encoding", Path(value["path"]).suffix.lstrip(".")).lower()
        image.encoding = {"rgb_u8": 1, "png": 2, "jpeg": 3, "jpg": 3}[encoding]
        if "shape" in value:
            image.height, image.width, image.channels = value["shape"]
        image.data = (base / value["path"]).read_bytes()
    language = root["language"]
    if "precomputed_embedding" in language:
        tensor(request.inputs.language.precomputed_embedding.embedding,
               language["precomputed_embedding"], base, [64, 4096], "T,D")
    else:
        tokens = language["tokens"]
        request.inputs.language.tokens.token_ids.extend(tokens["token_ids"])
        request.inputs.language.tokens.attention_mask.extend(
            tokens.get("attention_mask", [1] * len(tokens["token_ids"])))
    tensor(request.inputs.state, root["state"], base, [14], "D")
    if "noise" in root:
        tensor(request.inputs.noise, root["noise"], base, [48, 32], "T,A")
    return request


def lifecycle(channel, types, request_path=None, action_path=None, expected_path=None):
    info = rpc(channel, types, "GetModelInfo", types["GetModelInfoRequest"]())
    created = rpc(channel, types, "CreateSession", types["CreateSessionRequest"]())
    result = {"architecture": info.model.architecture, "session_id": created.session_id}
    if request_path:
        responses = [rpc(channel, types, "Predict",
                         build_predict(types, request_path, created.session_id, request_id))
                     for request_id in (1, 2)]
        actions = [bytes(response.prediction.action.data) for response in responses]
        if actions[0] != actions[1] or [value.request_id for value in responses] != [1, 2]:
            raise RuntimeError("repeated prepared request changed action or request_id")
        response = responses[-1]
        action = actions[-1]
        if action_path:
            action_path.write_bytes(action)
        result.update({
            "request_ids": [value.request_id for value in responses],
            "action_shape": list(response.prediction.action.shape),
            "action_sha256": hashlib.sha256(action).hexdigest(),
            "total_milliseconds": [value.prediction.stats.total_milliseconds
                                   for value in responses],
            "projected_prompt_cache_hit": [
                value.prediction.stats.projected_prompt_cache_hit for value in responses],
        })
        if expected_path:
            expected = expected_path.read_bytes()
            if len(expected) != len(action) or len(action) % 4:
                raise RuntimeError("expected action byte size differs from RPC action")
            actual_values = struct.unpack(f"<{len(action) // 4}f", action)
            expected_values = struct.unpack(f"<{len(expected) // 4}f", expected)
            errors = [abs(a - b) for a, b in zip(actual_values, expected_values)]
            result["mean_abs"] = sum(errors) / len(errors)
            result["max_abs"] = max(errors)
    rpc(channel, types, "ResetSession",
        types["ResetSessionRequest"](session_id=created.session_id))
    rpc(channel, types, "CloseSession",
        types["CloseSessionRequest"](session_id=created.session_id))
    try:
        rpc(channel, types, "CloseSession",
            types["CloseSessionRequest"](session_id=created.session_id))
        raise RuntimeError("second CloseSession unexpectedly succeeded")
    except RuntimeError as error:
        if "NOT_FOUND" not in str(error) or "session_id" not in str(error):
            raise
        result["structured_error"] = "NOT_FOUND"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--proto", type=Path,
                        help="legacy fallback: Proto source (requires --protoc)")
    parser.add_argument("--protoc", type=Path,
                        help="legacy fallback: protoc executable (requires --proto)")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--action", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    types = resolve_types(args, parser)
    with grpc.insecure_channel(args.endpoint) as channel:
        result = lifecycle(channel, types, args.request, args.action, args.expected)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
