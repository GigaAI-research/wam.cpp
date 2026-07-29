from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from ..adapters import create_adapter, check_environment
from ..language import create_language_provider
from ..local import Model, RuntimeConfig
from ..resources import LanguageResources
from .core import ServiceCore
from .websocket import WebSocketTransport


def serve_main(contract=None, argv=None):
    parser = argparse.ArgumentParser(prog="wam-serve")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--environment",
                        choices=("robotwin", "libero", "liberox"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--language-python-root", type=Path)
    parser.add_argument("--language-device")
    parser.add_argument("--language-cache-capacity", type=int, default=32)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18060)
    parser.add_argument("--backend",
                        choices=("automatic", "cuda", "cpu-metadata", "cpu"),
                        default="cuda")
    parser.add_argument("--precision",
                        choices=("automatic", "f32", "f16", "bf16"),
                        default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--prompt-cache-capacity", type=int, default=4)
    parser.add_argument("--language-mode",
                        choices=("automatic", "tokens", "external_embedding"),
                        default="automatic")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if contract is None:
        if args.environment is None:
            parser.error("--environment is required")
        contract = create_adapter(args.environment).contract
    elif args.environment and args.environment != contract.environment_id:
        parser.error("--environment differs from the fixed server contract")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = RuntimeConfig(
        backend=args.backend, precision=args.precision, device=args.device,
        prompt_cache_capacity=args.prompt_cache_capacity,
        language_mode=args.language_mode)
    with Model.load(args.model, library=args.library, config=config) as model:
        check_environment(model.metadata, contract)
        resources = LanguageResources.discover(
            args.model, tokenizer=args.tokenizer, encoder=args.text_encoder)
        mode = model.metadata["language_mode"]
        resources.require("tokenizer", *("encoder",)
                          if mode == "external_embedding" else ())
        provider = create_language_provider(
            model.metadata, tokenizer_path=resources.tokenizer,
            encoder_path=resources.encoder,
            python_root=args.language_python_root,
            device=args.language_device or f"cuda:{args.device}",
            cache_capacity=args.language_cache_capacity)
        service = ServiceCore(model, provider, args.descriptor, contract,
                              args.random_seed)
        asyncio.run(WebSocketTransport(
            service, args.host, args.port).serve())


def main(argv=None):
    serve_main(argv=argv)
