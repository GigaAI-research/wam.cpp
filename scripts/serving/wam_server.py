#!/usr/bin/env python3
"""Run the v0.1 WamService gRPC adapter over libwam_c_api."""

from __future__ import annotations

import argparse
import json
import signal
from concurrent import futures
from pathlib import Path

try:
    import grpc
except ModuleNotFoundError as exc:
    raise SystemExit("wam-server requires the Python package grpcio") from exc

from wam_rpc import (
    Backend,
    LanguageEncoderPolicy,
    ModelOptions,
    Precision,
    WamRpcError,
    WamRpcService,
)

METHODS = {
    "GetModelInfo": 1,
    "CreateSession": 2,
    "Predict": 3,
    "ResetSession": 4,
    "CloseSession": 5,
}

GRPC_CODES = {
    3: grpc.StatusCode.INVALID_ARGUMENT,
    5: grpc.StatusCode.NOT_FOUND,
    8: grpc.StatusCode.RESOURCE_EXHAUSTED,
    9: grpc.StatusCode.FAILED_PRECONDITION,
    12: grpc.StatusCode.UNIMPLEMENTED,
    13: grpc.StatusCode.INTERNAL,
}


class Bridge:
    def __init__(self, library: Path, options: ModelOptions):
        self.service = WamRpcService(library, options)

    def close(self):
        self.service.close()

    def call(self, method: int, request: bytes, context) -> bytes:
        try:
            return self.service.call_bytes(method, request)
        except WamRpcError as error:
            context.set_trailing_metadata((("wam-error-bin", error.payload),))
            context.abort(
                GRPC_CODES.get(error.grpc_status_code, grpc.StatusCode.INTERNAL),
                error.message,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--listen", default="127.0.0.1:50051")
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu-metadata"), default="auto")
    parser.add_argument("--precision", choices=("f32", "bf16"), default="f32")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt-cache-capacity", type=int, default=4)
    parser.add_argument("--language-policy",
                        choices=("resident", "fixed", "external_embedding"),
                        default="resident")
    parser.add_argument("--fixed-prompt", type=Path,
                        help="JSON with token_ids and attention_mask for fixed policy")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.prompt_cache_capacity < 0 or args.workers <= 0:
        parser.error("cache capacity must be non-negative and workers must be positive")

    if args.language_policy == "fixed" and args.fixed_prompt is None:
        parser.error("fixed policy requires --fixed-prompt")
    if args.language_policy != "fixed" and args.fixed_prompt is not None:
        parser.error("--fixed-prompt is valid only with fixed policy")
    prompt = json.loads(args.fixed_prompt.read_text()) if args.fixed_prompt else {}
    options = ModelOptions(
        artifact_path=args.model,
        backend={
            "auto": Backend.AUTO,
            "cuda": Backend.CUDA,
            "cpu-metadata": Backend.CPU_METADATA,
        }[args.backend],
        precision={
            "f32": Precision.F32_REFERENCE,
            "bf16": Precision.BF16_LATENCY,
        }[args.precision],
        device_index=args.device,
        prompt_cache_capacity=args.prompt_cache_capacity,
        language_encoder_policy={
            "resident": LanguageEncoderPolicy.RESIDENT,
            "fixed": LanguageEncoderPolicy.FIXED,
            "external_embedding": LanguageEncoderPolicy.EXTERNAL_EMBEDDING,
        }[args.language_policy],
        fixed_token_ids=prompt.get("token_ids", []),
        fixed_attention_mask=prompt.get("attention_mask", []),
    )
    bridge = Bridge(args.library, options)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    handlers = {
        name: grpc.unary_unary_rpc_method_handler(
            lambda request, context, method=method: bridge.call(method, request, context),
            request_deserializer=lambda value: value,
            response_serializer=lambda value: value)
        for name, method in METHODS.items()
    }
    server.add_generic_rpc_handlers((
        grpc.method_handlers_generic_handler("wam.rpc.v1.WamService", handlers),))
    if server.add_insecure_port(args.listen) == 0:
        bridge.close()
        raise SystemExit(f"cannot bind {args.listen}")
    server.start()
    print(f"wam-server ready listen={args.listen}", flush=True)

    def stop(_signum, _frame):
        server.stop(3)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        server.wait_for_termination()
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
