"""Run the upstream LIBERO-X evaluator against a wam.cpp RPC server."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.rpc import RpcClient


STATE_FIELDS = (
    "eef.position.x", "eef.position.y", "eef.position.z",
    "eef.rotation.axis_angle.x", "eef.rotation.axis_angle.y",
    "eef.rotation.axis_angle.z", "gripper.left", "gripper.right")
ACTION_FIELDS = (
    "eef.delta.position.x", "eef.delta.position.y",
    "eef.delta.position.z", "eef.delta.rotation.axis_angle.x",
    "eef.delta.rotation.axis_angle.y",
    "eef.delta.rotation.axis_angle.z", "gripper.command")


def check_observation_compatibility(policy_spec):
    roles = {view.role for view in policy_spec.images.views}
    if roles != {"scene", "wrist"}:
        raise RuntimeError(
            f"LIBERO-X requires scene/wrist image roles, got {sorted(roles)}")
    if tuple(policy_spec.state.fields) != STATE_FIELDS:
        raise RuntimeError("LIBERO-X state fields differ from the PolicySpec")


def observation_to_policy_observation(observation, policy_spec):
    check_observation_compatibility(policy_spec)
    images = [
        {"name": "scene", "data": np.ascontiguousarray(
            observation["observation/image"], dtype=np.uint8)},
        {"name": "wrist", "data": np.ascontiguousarray(
            observation["observation/wrist_image"], dtype=np.uint8)},
    ]
    state = np.ascontiguousarray(
        observation["observation/state"], dtype=np.float32).reshape(-1)
    if state.shape != (8,):
        raise ValueError(f"LIBERO-X state must be 8D, got {state.shape}")
    return images, state


def check_action_compatibility(policy_spec):
    action = policy_spec.action
    if tuple(action.fields) != ACTION_FIELDS or action.real_dim != 7:
        raise RuntimeError("LIBERO-X requires the frozen 7D action contract")
    if action.representation != "eef_delta_pose" or \
            action.frame != "robot_base":
        raise RuntimeError("LIBERO-X requires robot_base eef_delta_pose actions")


def policy_action_to_command(action, policy_spec, binarize_gripper=True):
    check_action_compatibility(policy_spec)
    command = np.asarray(action, dtype=np.float32).copy()
    if command.ndim != 2 or command.shape[1] != 7:
        raise ValueError(f"LIBERO-X action chunk must be [T,7], got {command.shape}")
    command[:, -1] *= -1.0
    if binarize_gripper:
        command[:, -1] = np.sign(command[:, -1])
    return command


class WamLiberoXPolicy:
    def __init__(self, host, port, descriptor, random_seed=7,
                 binarize_gripper=True):
        self.rpc = RpcClient(host, port, descriptor, "liberox")
        self.info = self.rpc.connect()
        check_observation_compatibility(self.info.policy_spec)
        check_action_compatibility(self.info.policy_spec)
        if not self.info.capabilities.explicit_action_noise:
            raise RuntimeError("LIBERO-X evaluation requires explicit action noise")
        self.rng = np.random.default_rng(random_seed)
        self.binarize_gripper = binarize_gripper
        self.last_prompt = None

    def infer(self, observation):
        prompt = str(observation.get("prompt", ""))
        if self.last_prompt is not None and prompt != self.last_prompt:
            self.rpc.reset()
        self.last_prompt = prompt
        images, state = observation_to_policy_observation(
            observation, self.info.policy_spec)
        action_spec = self.info.policy_spec.action
        noise = self.rng.standard_normal(
            (action_spec.horizon, action_spec.model_dim), dtype=np.float32)
        action, _ = self.rpc.predict(images, state, prompt, noise)
        return {"actions": policy_action_to_command(
            action, self.info.policy_spec, self.binarize_gripper)}

    def get_server_metadata(self):
        return {
            "policy": self.info.architecture,
            "profile": self.info.policy_spec.profile,
            "action_horizon": self.info.policy_spec.action.horizon,
        }

    def reset(self):
        self.rpc.reset()
        self.last_prompt = None

    def close(self):
        self.rpc.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--liberox-root", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18060)
    parser.add_argument("--scene-group", default="LEVEL1")
    parser.add_argument("--load-mode", choices=("bddl", "init"), default="init")
    parser.add_argument("--bddl-root", type=Path)
    parser.add_argument("--init-root", type=Path)
    parser.add_argument("--level5-prompt-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-output", type=Path)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-flip-images", action="store_true")
    parser.add_argument("--save-all-videos", action="store_true")
    parser.add_argument("--no-binarize-gripper", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.liberox_root.resolve()
    sys.path[:0] = [str(root), str(root / "packages/openpi-client/src")]
    policy_holder = {}

    class BoundPolicy:
        def __init__(self, host, port):
            self.impl = WamLiberoXPolicy(
                host, port, args.descriptor, args.seed,
                not args.no_binarize_gripper)
            policy_holder["policy"] = self.impl

        def infer(self, observation):
            return self.impl.infer(observation)

        def get_server_metadata(self):
            return self.impl.get_server_metadata()

        def reset(self):
            return self.impl.reset()

    import openpi_client
    websocket_adapter = types.ModuleType("openpi_client.websocket_client_policy")
    websocket_adapter.WebsocketClientPolicy = BoundPolicy
    sys.modules[websocket_adapter.__name__] = websocket_adapter
    openpi_client.websocket_client_policy = websocket_adapter
    import eval_template

    bddl_root = args.bddl_root or root / "libero/libero_x/bddl"
    init_root = args.init_root or root / "libero/libero_x/init"
    level5_root = (args.level5_prompt_root or
                   root / "libero/libero_x/LEVEL5")
    evaluation = eval_template.Args(
        host=args.host, port=args.port, resize_size=args.resize_size,
        replan_steps=args.replan_steps, scene_group=args.scene_group,
        load_mode=args.load_mode, bddl_root=str(bddl_root),
        init_root=str(init_root), level5_prompt_root=str(level5_root),
        num_trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait, max_steps=args.max_steps,
        flip_images=not args.no_flip_images,
        video_out_path=str(args.output_dir),
        results_out_path=(str(args.results_output)
                          if args.results_output else ""),
        save_all_videos=args.save_all_videos, fps=args.fps, seed=args.seed)
    try:
        eval_template.eval_template(evaluation)
    finally:
        policy = policy_holder.get("policy")
        if policy is not None:
            policy.close()


if __name__ == "__main__":
    main()
