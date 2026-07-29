from __future__ import annotations

import argparse
import json

import numpy as np

from .local import Pipeline, RuntimeConfig, SessionConfig


def _inputs(path):
    values = np.load(path, allow_pickle=False)
    images = [{"name": key[6:], "data": values[key]}
              for key in values.files if key.startswith("image.")]
    kwargs = {}
    for key in ("token_ids", "attention_mask", "embedding",
                "embedding_attention_mask", "action_noise"):
        if key in values:
            kwargs[key] = values[key]
    if "state" not in values or not images:
        raise ValueError("input NPZ requires state and at least one image.<role>")
    return images, values["state"], kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wam-predict")
    parser.add_argument("model", help="GGUF file or artifact bundle directory")
    parser.add_argument("input", help="NPZ containing state, image.<role>, and language arrays")
    parser.add_argument("--library", help="path to libwam_c_api")
    parser.add_argument("--output", help="write the F32 action array as NPY")
    parser.add_argument("--instruction", help="raw instruction; language resources come from the bundle or explicit options")
    parser.add_argument("--tokenizer")
    parser.add_argument("--language-encoder")
    parser.add_argument("--language-python-root")
    parser.add_argument("--language-device", default="cuda:0")
    parser.add_argument("--backend", default="cuda", choices=("automatic", "cuda", "cpu-metadata", "cpu"))
    parser.add_argument("--precision", default="bf16", choices=("automatic", "f32", "f16", "bf16"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    images, state, kwargs = _inputs(args.input)
    runtime = RuntimeConfig(args.backend, args.precision, args.device)
    with Pipeline.load(
        args.model, library=args.library, runtime_config=runtime,
        session_config=SessionConfig(random_seed=args.seed),
        tokenizer=args.tokenizer, language_encoder=args.language_encoder,
        language_python_root=args.language_python_root,
        language_device=args.language_device,
    ) as pipeline:
        prediction = pipeline.predict(images, state,
                                      instruction=args.instruction, **kwargs)
    if args.output:
        np.save(args.output, prediction.action)
    else:
        print(json.dumps({"shape": list(prediction.action.shape),
                          "action": prediction.action.tolist(),
                          "stats": prediction.stats}))


if __name__ == "__main__":
    main()
