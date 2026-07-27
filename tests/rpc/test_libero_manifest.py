#!/usr/bin/env python3
"""Unit tests for deterministic LIBERO manifest and reporting helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


def load_client():
    path = (Path(__file__).resolve().parents[2] /
            "eval/sim/run_libero_client.py")
    spec = importlib.util.spec_from_file_location("wam_libero_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_merger():
    path = (Path(__file__).resolve().parents[2] /
            "eval/sim/merge_libero_results.py")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("wam_libero_merger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    client = load_client()
    assert client._parse_task_ids("0,2-4,3") == [0, 2, 3, 4]
    execution = {
        "replan_steps": 10, "num_steps_wait": 30, "max_steps": 400,
        "resolution": 256, "binarize_gripper": True,
        "explicit_action_noise": True}
    manifest = client.build_manifest(
        "libero_spatial", [0, 2], 2, 7, 1000, execution,
        {0: 50, 2: 50})
    client.validate_manifest(manifest)
    assert [item["seed"] for item in manifest["episodes"]] == [7, 8, 9, 10]
    assert [item["action_noise_seed"] for item in manifest["episodes"]] == [
        1000, 1001, 1002, 1003]
    assert client._manifest_hash(manifest) == client._manifest_hash(manifest)
    assert manifest["episodes"][2:4][0]["episode_id"] == \
        "libero_spatial-task02-episode000"

    invalid = dict(manifest)
    invalid["execution"] = dict(execution, explicit_action_noise=False)
    try:
        client.validate_manifest(invalid)
    except ValueError as error:
        assert "explicit action noise" in str(error)
    else:
        raise AssertionError("manifest must require explicit action noise")

    action = SimpleNamespace(
        fields=client.ACTION_FIELDS, real_dim=7,
        representation="eef_delta_pose", frame="robot_base")
    spec = SimpleNamespace(action=action)
    command = client.policy_action_to_command(
        np.asarray([0, 0, 0, 0, 0, 0, 0.75], np.float32), spec)
    assert command[-1] == -1.0

    episodes = [{"success": True}, {"success": False}]
    requests = [{
        "rpc_roundtrip_milliseconds": 10.0,
        "server_total_milliseconds": 8.0,
        "server_model_milliseconds": 7.0,
        "server_text_milliseconds": 2.0}]
    summary = client._summary(
        manifest, "abc", {"profile": "test"}, episodes, requests)
    assert summary["successes"] == 1 and summary["success_rate"] == 0.5
    assert summary["latency"]["server_total_milliseconds"]["mean"] == 8.0
    assert summary["tasks"] == []

    info = SimpleNamespace(
        policy_spec=SimpleNamespace(action=SimpleNamespace(horizon=16)),
        capabilities=SimpleNamespace(explicit_action_noise=True))
    client._validate_runtime(info, execution)
    try:
        client._validate_runtime(info, dict(execution, replan_steps=17))
    except ValueError as error:
        assert "replan_steps" in str(error)
    else:
        raise AssertionError("replan_steps must not exceed the action horizon")
    info.capabilities.explicit_action_noise = False
    try:
        client._validate_runtime(info, execution)
    except ValueError as error:
        assert "explicit action noise" in str(error)
    else:
        raise AssertionError("explicit action-noise capability must be required")

    merger = load_merger()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest_path = root / "manifest.json"
        manifest_path.write_bytes(client._canonical_json(manifest))
        for shard_index, (start, end) in enumerate(((0, 2), (2, 4))):
            shard = root / f"shard{shard_index}"
            shard.mkdir()
            (shard / "manifest.json").write_bytes(client._canonical_json(manifest))
            (shard / "selection.json").write_text(json.dumps(
                {"episode_start": start, "episode_end": end}))
            shard_episodes = []
            for entry in manifest["episodes"][start:end]:
                shard_episodes.append(dict(entry, status="completed",
                                           success=entry["episode_index"] == 0,
                                           request_count=0))
            client._write_jsonl_atomic(shard / "episodes.jsonl", shard_episodes)
            client._write_json_atomic(shard / "summary.json", client._summary(
                manifest, client._manifest_hash(manifest),
                {"profile": "fixture"}, shard_episodes, []))
        merged = merger.merge(
            manifest_path, [root / "shard0", root / "shard1"],
            root / "merged")
        assert merged["completed_episodes"] == 4
        assert merged["successes"] == 2
        assert [(item["task_id"], item["successes"])
                for item in merged["tasks"]] == [(0, 1), (2, 1)]
        merged_episodes = client._load_jsonl(root / "merged/episodes.jsonl")
        assert [item["episode_id"] for item in merged_episodes] == [
            item["episode_id"] for item in manifest["episodes"]]


def main():
    argparse.ArgumentParser().parse_args()
    run()
    print("wam LIBERO fixed manifest: PASS")


if __name__ == "__main__":
    main()
