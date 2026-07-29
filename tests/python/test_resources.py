import json
import tempfile
from pathlib import Path

from wam import LanguageResourceError, LanguageResources
from wam import Pipeline, Prediction
import wam.language
import numpy as np


def main():
    with tempfile.TemporaryDirectory(prefix="wam-resources-") as directory:
        root = Path(directory)
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "encoder.gguf").write_bytes(b"fixture")
        (root / "manifest.json").write_text(json.dumps({
            "manifest_schema_version": 1, "model": "model.gguf",
            "tokenizer": "tokenizer.json", "language_encoder": "encoder.gguf",
        }), encoding="utf-8")
        resources = LanguageResources.discover(root).require("tokenizer", "encoder")
        assert resources.tokenizer == (root / "tokenizer.json").resolve()
        (root / "manifest.json").write_text(json.dumps({
            "manifest_schema_version": 1, "model": "model.gguf",
            "tokenizer": "../outside.json",
        }), encoding="utf-8")
        try:
            LanguageResources.discover(root)
        except LanguageResourceError as error:
            assert "escapes" in str(error)
        else:
            raise AssertionError("escaping bundle resource was accepted")

        class Provider:
            def prepare(self, instruction):
                assert instruction == "pick up cup"
                return wam.language.PreparedLanguage(
                    np.asarray([1, 2], dtype=np.int32),
                    np.asarray([1, 1], dtype=np.int32))

        class Model:
            metadata = {"language_mode": "tokens"}

        class Session:
            def predict(self, images, state, **kwargs):
                assert kwargs["token_ids"].tolist() == [1, 2]
                return Prediction(np.zeros((1, 1), np.float32), {
                    "preprocess_milliseconds": 0.0,
                    "model_text_milliseconds": 0.0,
                    "model_milliseconds": 0.0,
                    "total_milliseconds": 0.0,
                })

        original = wam.language.create_language_provider
        wam.language.create_language_provider = lambda *args, **kwargs: Provider()
        try:
            pipeline = Pipeline(Model(), Session(),
                                LanguageResources(root / "tokenizer.json"), {})
            assert pipeline.predict([], np.zeros(1, np.float32),
                                    instruction="pick up cup").action.shape == (1, 1)
        finally:
            wam.language.create_language_provider = original


if __name__ == "__main__":
    main()
