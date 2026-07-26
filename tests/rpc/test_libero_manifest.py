#!/usr/bin/env python3
"""Unit tests for deterministic LIBERO manifest and reporting helpers."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_client():
    path = (Path(__file__).resolve().parents[2] /
            "eval/sim/run_libero_client.py")
    spec = importlib.util.spec_from_file_location("wam_libero_client", path)
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


def main():
    argparse.ArgumentParser().parse_args()
    run()
    print("wam LIBERO fixed manifest: PASS")


if __name__ == "__main__":
    main()
