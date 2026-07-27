#!/usr/bin/env python3
"""Unit tests for the LIBERO-X environment adapter."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_client():
    path = (Path(__file__).resolve().parents[2] /
            "eval/sim/run_liberox_client.py")
    spec = importlib.util.spec_from_file_location("wam_liberox_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    client = load_client()
    views = [SimpleNamespace(role="scene"), SimpleNamespace(role="wrist")]
    state = SimpleNamespace(fields=client.STATE_FIELDS)
    action = SimpleNamespace(
        fields=client.ACTION_FIELDS, real_dim=7,
        representation="eef_delta_pose", frame="robot_base")
    policy_spec = SimpleNamespace(
        images=SimpleNamespace(views=views), state=state, action=action)

    observation = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
    }
    images, state_values = client.observation_to_policy_observation(
        observation, policy_spec)
    assert [image["name"] for image in images] == ["scene", "wrist"]
    assert state_values.tolist() == list(range(8))

    chunk = np.zeros((2, 7), dtype=np.float32)
    chunk[:, -1] = [-0.25, 0.75]
    command = client.policy_action_to_command(chunk, policy_spec)
    assert command[:, -1].tolist() == [1.0, -1.0]
    assert chunk[:, -1].tolist() == [-0.25, 0.75]

    first_noise = client.donor_action_noise(7, 32, 7)
    repeated_noise = client.donor_action_noise(7, 32, 7)
    assert first_noise.dtype == np.float32
    assert np.array_equal(first_noise, repeated_noise)
    assert np.array_equal(first_noise.reshape(-1)[:8], np.asarray([
        -0.8203125, 0.396484375, 0.8984375, -1.390625,
        -0.1669921875, 0.28515625, -0.640625, -0.89453125,
    ], dtype=np.float32))


def main():
    argparse.ArgumentParser().parse_args()
    run()
    print("wam LIBERO-X adapter: PASS")


if __name__ == "__main__":
    main()
