#!/usr/bin/env python3
"""Unit tests for frozen LIBERO-X manifests and staging."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


def load_module():
    path = (Path(__file__).resolve().parents[2] /
            "eval/sim/liberox_manifest.py")
    spec = importlib.util.spec_from_file_location("liberox_manifest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    module = load_module()
    repository_manifest = (Path(__file__).resolve().parents[2] /
                           "eval/manifests/fastwam_liberox_level1_task0_10_v1.json")
    frozen = json.loads(repository_manifest.read_text())
    module.validate_manifest(frozen)
    assert module.manifest_sha256(frozen) == \
        "458b441e34d2566333430db97848b537e4967a32cea7f2889486a80f7897e5c1"

    with tempfile.TemporaryDirectory(prefix="wam-liberox-manifest-") as temp:
        root = Path(temp)
        bddl = root / "source/bddl/LEVEL1"
        init = root / "source/init/LEVEL1"
        bddl.mkdir(parents=True)
        init.mkdir(parents=True)
        for name, count in (("SCENE2_task", 3), ("SCENE10_task", 4)):
            (bddl / f"{name}.bddl").write_text(f"problem {name}\n")
            (init / f"{name}.init").write_text(
                json.dumps(list(range(count))))

        def load(path):
            return json.loads(Path(path).read_text())

        def save(value, path):
            Path(path).write_text(json.dumps(value))

        execution = {
            "load_mode": "init", "num_steps_wait": 10,
            "max_steps": 1200, "resize_size": 224, "replan_steps": 10,
            "flip_images": True, "binarize_gripper": True,
            "action_noise_seed": 7,
            "action_noise_generator": module.ACTION_NOISE_GENERATOR,
        }
        manifest = module.build_manifest(
            root / "source/bddl", root / "source/init", "LEVEL1", [0],
            2, execution, load)
        module.validate_manifest(manifest)
        assert manifest["tasks"][0]["bddl_file"] == "SCENE2_task.bddl"
        assert manifest["tasks"][0]["init_count"] == 3
        assert manifest["tasks"][0]["episode_count"] == 2
        assert module.manifest_sha256(manifest) == \
            module.manifest_sha256(manifest)

        staged_bddl, staged_init, selected = module.stage_manifest(
            manifest, root / "source/bddl", root / "source/init",
            root / "stage", load, save)
        assert (staged_bddl / "LEVEL1/SCENE2_task.bddl").is_file()
        assert load(staged_init / "LEVEL1/SCENE2_task.init") == [0, 1]

        results = [
            {"task_file": str(staged_bddl / "LEVEL1/SCENE2_task.bddl"),
             "episode_index": index, "success": index == 0}
            for index in range(2)
        ]
        module.validate_results(results, selected)
        requests = [{field: float(index + 1)
                     for field in module.LATENCY_FIELDS}
                    for index in range(2)]
        summary = module.build_summary(
            manifest, {"profile": "test"}, results, requests, selected)
        assert summary["successes"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["latency"]["server_model_milliseconds"]["p50"] == 1.5

        (bddl / "SCENE2_task.bddl").write_text("changed\n")
        try:
            module.stage_manifest(
                manifest, root / "source/bddl", root / "source/init",
                root / "another-stage", load, save)
        except ValueError as error:
            assert "hash changed" in str(error)
        else:
            raise AssertionError("staging must reject changed source data")


if __name__ == "__main__":
    run()
    print("wam LIBERO-X fixed manifest: PASS")
