from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wam.adapters.libero import ACTION_FIELDS, STATE_FIELDS, LiberoAdapter
from wam.adapters.liberox import LiberoXAdapter
from wam.adapters.robotwin import JOINT_FIELDS, RoboTwinAdapter
from wam.eval import (ActionChunkExecutor, Metrics, ResultWriter,
                      VideoWriter, load_manifest, write_manifest)


def policy_spec(roles, state_fields, action_fields, action_dim,
                representation, frame):
    return SimpleNamespace(
        images=SimpleNamespace(
            views=[SimpleNamespace(role=role) for role in roles]),
        state=SimpleNamespace(fields=state_fields, real_dim=len(state_fields)),
        action=SimpleNamespace(
            fields=action_fields, real_dim=action_dim, horizon=4,
            model_dim=action_dim, representation=representation,
            frame=frame))


def run():
    rng = np.random.default_rng(7)
    robotwin_spec = policy_spec(
        ("camera_high", "camera_left_wrist", "camera_right_wrist"),
        JOINT_FIELDS, JOINT_FIELDS, 14, "joint_position", "controller")
    robotwin = RoboTwinAdapter()
    robotwin_observation = {
        "observation.images.cam_high": np.zeros((2, 3, 4), np.float32),
        "observation.images.cam_left_wrist": np.ones((2, 3, 3), np.uint8),
        "observation.images.cam_right_wrist": np.ones((2, 3, 3), np.uint8),
        "observation.state": np.arange(14, dtype=np.float32),
    }
    images, state = robotwin.observation(robotwin_observation, robotwin_spec)
    assert [item["name"] for item in images] == [
        "camera_high", "camera_left_wrist", "camera_right_wrist"]
    assert images[0]["data"].shape == (2, 3, 3) and state.shape == (14,)
    robotwin_executor = ActionChunkExecutor(
        2, lambda value: robotwin.action(value, robotwin_spec))
    robotwin_executor.push(rng.standard_normal((4, 14), dtype=np.float32))
    assert robotwin_executor.next().shape == (14,)

    libero_spec = policy_spec(
        ("scene", "wrist"), STATE_FIELDS, ACTION_FIELDS, 7,
        "eef_delta_pose", "robot_base")
    libero = LiberoAdapter()
    libero_observation = {
        "agentview_image": np.zeros((2, 3, 3), np.uint8),
        "robot0_eye_in_hand_image": np.ones((2, 3, 3), np.uint8),
        "robot0_eef_pos": np.arange(3, dtype=np.float32),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1], np.float32),
        "robot0_gripper_qpos": np.asarray([0.1, 0.2], np.float32),
    }
    _, state = libero.observation(libero_observation, libero_spec)
    assert state.shape == (8,)
    command = libero.action(np.asarray([0, 0, 0, 0, 0, 0, 0.75]),
                            libero_spec)
    assert command[-1] == -1.0
    libero_executor = ActionChunkExecutor(
        2, lambda value: libero.action(value, libero_spec))
    _, executed = libero_executor.run(
        rng.random((4, 7), dtype=np.float32),
        lambda action: (action, 0.0, True, {}))
    assert executed == 1

    liberox = LiberoXAdapter()
    images, state = liberox.observation({
        "observation/image": np.zeros((2, 3, 3), np.uint8),
        "observation/wrist_image": np.ones((2, 3, 3), np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
    }, libero_spec)
    assert len(images) == 2 and state.shape == (8,)
    transformed = liberox.action(
        np.asarray([[0, 0, 0, 0, 0, 0, 0.5]]), libero_spec)
    assert transformed[0, -1] == -1.0
    liberox_executor = ActionChunkExecutor(
        2, lambda value: liberox.action(value, libero_spec))
    _, executed = liberox_executor.run(
        rng.standard_normal((4, 7), dtype=np.float32),
        lambda action: (action, 0.0, True, {}))
    assert executed == 1

    executor = ActionChunkExecutor(2, lambda value: value + 1)
    assert executor.push(np.zeros((4, 2), np.float32)) == 2
    assert executor.next().tolist() == [1, 1] and executor.pending == 1
    executor.reset()
    assert executor.pending == 0
    steps = []
    result, count = executor.run(
        np.zeros((4, 2), np.float32),
        lambda action: (steps.append(action.copy()), 0, len(steps) == 1, {}))
    assert count == 1 and result[2] and executor.pending == 0
    _, count = executor.run(
        np.zeros((4, 2), np.float32),
        lambda action: (action, 0.0, False, True, {}))
    assert count == 1 and executor.pending == 0
    try:
        ActionChunkExecutor(2, lambda value: value[0]).push(
            np.zeros((4, 2), np.float32))
    except ValueError as error:
        assert "preserve rank" in str(error)
    else:
        raise AssertionError("executor accepted a shape-changing transform")

    metrics = Metrics()
    metrics.record(latency=2.0)
    metrics.record(latency=4.0)
    assert metrics.summary(("latency",))["latency"]["mean"] == 3.0
    with tempfile.TemporaryDirectory(prefix="wam-phase8-") as directory:
        root = Path(directory)
        write_manifest(root / "manifest.json", {"format": "fixture", "seed": 7})
        assert load_manifest(root / "manifest.json", expected_format="fixture")["seed"] == 7
        writer = ResultWriter(root / "results.jsonl")
        writer.append({"episode": 0, "success": True})
        assert writer.load() == [{"episode": 0, "success": True}]
        with VideoWriter() as video:
            video.append(np.zeros((2, 2, 3), np.uint8))


if __name__ == "__main__":
    run()
    print("wam Phase 8 adapters/eval: PASS")
