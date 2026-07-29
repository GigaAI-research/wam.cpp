from __future__ import annotations

import re
from pathlib import Path


def run():
    root = Path(__file__).resolve().parents[2]
    required = (
        "docs/adding-a-model.md",
        "docs/adding-an-environment.md",
        "docs/artifact-bundle.md",
        "docs/serving.md",
        "docs/evaluation.md",
        "docs/support-matrix.md",
        "examples/cpp/predict.cpp",
        "examples/python/predict.py",
        "examples/python/remote_predict.py",
        ".github/workflows/cpu.yml",
    )
    for relative in required:
        assert (root / relative).is_file(), f"missing release resource: {relative}"

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# wam.cpp 0.6\n")
    for command in ("wam-inspect", "wam-validate", "wam-predict", "wam-serve"):
        assert command in readme, f"README omits user path: {command}"
    assert "wam.cpp 0.5" not in readme
    assert "Slice " not in readme
    assert "PolicySpecDraft" not in readme
    assert "serving-evaluation.md" not in readme

    release_docs = [root / "README.md", root / "ARCHITECTURE.md"]
    release_docs.extend(root.glob("docs/*.md"))
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for document in release_docs:
        text = document.read_text(encoding="utf-8")
        assert "wam.cpp 0.5" not in text, f"stale product version: {document}"
        assert "PolicySpecDraft" not in text, f"stale type: {document}"
        for target in link_pattern.findall(text):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#", 1)[0]
            resolved = (document.parent / path).resolve()
            assert resolved.exists(), f"broken link in {document}: {target}"

    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    package = (root / "pyproject.toml").read_text(encoding="utf-8")
    version = (root / "include/wam/version.h").read_text(encoding="utf-8")
    assert "project(wam VERSION 0.6.0" in cmake
    assert 'version = "0.6.0"' in package
    assert '"Pillow>=10,<13"' in package
    assert '"websockets>=14,<16"' in package
    assert "WAM_VERSION_MAJOR 0" in version
    assert "WAM_VERSION_MINOR 6" in version

    assert not (root / "scripts/inspect/inspect_gguf.py").exists()
    assert not (root / "scripts/inspect/inspect_fastwam.py").exists()
    assert not (root / "plan.md").exists()
    assert not (root / "MIGRATION_INVENTORY.md").exists()


if __name__ == "__main__":
    run()
    print("wam release documentation contract: PASS")
