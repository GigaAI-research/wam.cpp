import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="wam-wheel-") as directory:
        root = Path(directory)
        wheel_dir = root / "dist"
        source = root / "source"
        shutil.copytree(args.source, source, ignore=shutil.ignore_patterns(
            ".git", "build", "*.egg-info", "__pycache__"))
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(source),
                        "--no-deps", "--no-build-isolation", "--wheel-dir", wheel_dir],
                       check=True)
        wheel = next(wheel_dir.glob("wam-0.6.0-*.whl"))
        environment = root / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", environment], check=True)
        python = environment / "bin" / "python"
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], check=True)
        subprocess.run([str(python), "-c",
                        "import wam; assert wam.__version__ == '0.6.0'; "
                        "assert wam.RuntimeConfig().backend == 'automatic'"], check=True)
        subprocess.run([str(environment / "bin" / "wam-predict"), "--help"],
                       check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
