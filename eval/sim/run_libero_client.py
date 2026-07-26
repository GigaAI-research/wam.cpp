"""Run one upstream LIBERO episode against a wam.cpp RPC server."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.rpc import RpcClient


STATE_FIELDS = (
    "eef.position.x", "eef.position.y", "eef.position.z",
    "eef.rotation.axis_angle.x", "eef.rotation.axis_angle.y",
    "eef.rotation.axis_angle.z", "gripper.left", "gripper.right")
ACTION_FIELDS = (
    "eef.delta.position.x", "eef.delta.position.y", "eef.delta.position.z",
    "eef.delta.rotation.axis_angle.x", "eef.delta.rotation.axis_angle.y",
    "eef.delta.rotation.axis_angle.z", "gripper.command")
MAX_STEPS = {"libero_spatial": 400, "libero_object": 400,
             "libero_goal": 400, "libero_10": 700, "libero_90": 700}


def check_observation_compatibility(policy_spec):
    roles = {view.role for view in policy_spec.images.views}
    if roles != {"scene", "wrist"}:
        raise RuntimeError(f"LIBERO requires scene/wrist image roles, got {sorted(roles)}")
    if tuple(policy_spec.state.fields) != STATE_FIELDS:
        raise RuntimeError("LIBERO state field order does not match the server PolicySpec")


def _quat2axisangle(value):
    quat = np.asarray(value, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(quat[:3] * (2.0 * math.acos(quat[3]) / denominator),
                      dtype=np.float32)


def observation_to_policy_observation(observation, policy_spec):
    check_observation_compatibility(policy_spec)
    images = [
        {"name": "scene", "data": np.ascontiguousarray(
            observation["agentview_image"][::-1, ::-1], dtype=np.uint8)},
        {"name": "wrist", "data": np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1, ::-1], dtype=np.uint8)},
    ]
    state = np.concatenate((
        observation["robot0_eef_pos"],
        _quat2axisangle(observation["robot0_eef_quat"]),
        observation["robot0_gripper_qpos"])).astype(np.float32)
    return images, state


def check_action_compatibility(policy_spec):
    action = policy_spec.action
    if tuple(action.fields) != ACTION_FIELDS or action.real_dim != 7:
        raise RuntimeError("LIBERO requires the frozen 7D delta-pose action contract")
    if action.representation != "eef_delta_pose" or action.frame != "robot_base":
        raise RuntimeError("LIBERO requires robot_base eef_delta_pose actions")


def policy_action_to_command(action, policy_spec, binarize_gripper=True):
    check_action_compatibility(policy_spec)
    command = np.asarray(action, dtype=np.float32).copy()
    if command.shape != (7,):
        raise ValueError(f"LIBERO command must be 7D, got {command.shape}")
    command[-1] = 1.0 - 2.0 * command[-1]
    if binarize_gripper:
        command[-1] = np.sign(command[-1])
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--suite", choices=tuple(MAX_STEPS), default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18060)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--video-output", type=Path)
    return parser.parse_args(argv)


def _configure_libero(root):
    if os.environ.get("LIBERO_CONFIG_PATH"):
        return None
    import yaml
    holder = tempfile.TemporaryDirectory(prefix="wam-libero-")
    package = root / "libero/libero"
    config = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(root / "libero/datasets"),
        "assets": str(package / "assets"),
    }
    Path(holder.name, "config.yaml").write_text(yaml.safe_dump(config))
    os.environ["LIBERO_CONFIG_PATH"] = holder.name
    return holder


def main(argv=None):
    args = parse_args(argv)
    root = args.libero_root.resolve()
    sys.path.insert(0, str(root))
    libero_config = _configure_libero(root)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task-id must be within [0,{suite.n_tasks - 1}]")
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.episode_index < len(init_states):
        raise ValueError(f"episode-index must be within [0,{len(init_states) - 1}]")
    bddl = (Path(get_libero_path("bddl_files")) / task.problem_folder /
            task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=args.resolution,
        camera_widths=args.resolution)
    rpc = RpcClient(args.host, args.port, args.descriptor, "libero")
    writer = None
    latencies = []
    try:
        info = rpc.connect()
        check_observation_compatibility(info.policy_spec)
        check_action_compatibility(info.policy_spec)
        if not 1 <= args.replan_steps <= info.policy_spec.action.horizon:
            raise ValueError(
                f"replan-steps must be within [1,{info.policy_spec.action.horizon}]")
        rpc.reset()
        env.seed(args.seed)
        env.reset()
        observation = env.set_init_state(init_states[args.episode_index])
        if args.video_output:
            import imageio.v2 as imageio
            args.video_output.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video_output), fps=20)
        done = False
        for _ in range(args.num_steps_wait):
            if writer:
                writer.append_data(observation["agentview_image"][::-1, ::-1])
            observation, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                break
        steps = requests = 0
        limit = args.max_steps or MAX_STEPS[args.suite]
        while not done and steps < limit:
            images, state = observation_to_policy_observation(
                observation, info.policy_spec)
            started = time.perf_counter()
            chunk, stats = rpc.predict(images, state, task.language)
            latency = (time.perf_counter() - started) * 1000.0
            latencies.append({
                "rpc_roundtrip_milliseconds": latency,
                "server_total_milliseconds": stats.total_milliseconds,
                "server_model_milliseconds": stats.model_milliseconds,
                "server_text_milliseconds": stats.model_text_milliseconds})
            requests += 1
            for action in chunk[:args.replan_steps]:
                command = policy_action_to_command(
                    action, info.policy_spec, not args.no_binarize_gripper)
                observation, _, done, _ = env.step(command)
                steps += 1
                if writer:
                    writer.append_data(observation["agentview_image"][::-1, ::-1])
                if done or steps >= limit:
                    break
        report = {
            "format": "wam-libero-single-episode-v1", "suite": args.suite,
            "task_id": args.task_id, "task": task.name,
            "instruction": task.language, "episode_index": args.episode_index,
            "seed": args.seed, "success": bool(done), "steps": steps,
            "requests": requests, "replan_steps": args.replan_steps,
            "latencies": latencies}
        print(json.dumps(report, indent=2))
        if args.metrics_output:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if writer:
            writer.close()
        rpc.close()
        env.close()
        if libero_config is not None:
            libero_config.cleanup()


if __name__ == "__main__":
    main()
