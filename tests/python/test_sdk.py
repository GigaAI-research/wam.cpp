import argparse
import subprocess
import tempfile
from pathlib import Path

import wam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--fixture-writer", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="wam-python-sdk-") as directory:
        artifact = Path(directory) / "model.gguf"
        subprocess.run([args.fixture_writer, str(artifact)], check=True)
        with wam.Model.load(
            artifact, library=args.library,
            config=wam.RuntimeConfig(backend="cpu-metadata"),
        ) as model:
            assert model.metadata["architecture"] == "gwp05"
            try:
                model.create_session()
            except wam.WamError as error:
                assert error.code == wam.ErrorCode.UNSUPPORTED
                assert error.details
            else:
                raise AssertionError("metadata-only model created a session")
        try:
            model.create_session()
        except RuntimeError as error:
            assert "closed" in str(error)
        else:
            raise AssertionError("closed model remained usable")


if __name__ == "__main__":
    main()
