"""Run the upstream RoboTwin evaluator against the wam.cpp 0.5 RPC server."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.rpc import RpcClient


ROLE_MAP = {
    "observation.images.cam_high": "camera_high",
    "observation.images.cam_left_wrist": "camera_left_wrist",
    "observation.images.cam_right_wrist": "camera_right_wrist",
}


def _rgb_u8(value):
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"RoboTwin image must be HWC RGB/RGBA, got {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.max()) <= 1.5:
            array = array * 255.0
        array = np.clip(np.rint(array), 0, 255)
    return np.ascontiguousarray(array, dtype=np.uint8)


class RobotwinPolicy:
    def __init__(self, host, port, descriptor, execute_steps=48,
                 request_log=None, fixed_action_noise_seed=None):
        self.rpc = RpcClient(host, port, descriptor, "robotwin")
        info = self.rpc.connect()
        spec = info.policy_spec
        roles = {view.role for view in spec.images.views}
        if roles != set(ROLE_MAP.values()):
            raise RuntimeError(f"server image roles are incompatible: {sorted(roles)}")
        if spec.state.real_dim != 14 or spec.action.real_dim != 14:
            raise RuntimeError("RoboTwin requires 14D state and action")
        if not 1 <= execute_steps <= spec.action.horizon:
            raise ValueError(f"execute_steps must be within [1,{spec.action.horizon}]")
        self.execute_steps = execute_steps
        self.action_noise = None
        if fixed_action_noise_seed is not None:
            import torch

            generator = torch.Generator(device="cpu").manual_seed(
                fixed_action_noise_seed)
            self.action_noise = torch.randn(
                (spec.action.horizon, spec.action.model_dim),
                generator=generator, dtype=torch.float32).numpy()
        self.last_instruction = None
        self.episode_index = 0
        self.episode_request_counts = {}
        self.request_metrics = []
        self.request_log = request_log

    def set_episode(self, episode_index):
        self.episode_index = int(episode_index)

    def infer(self, observation):
        infer_start = time.perf_counter()
        instruction = str(observation.get("instruct", ""))
        if self.last_instruction is not None and instruction != self.last_instruction:
            self.rpc.reset()
        self.last_instruction = instruction
        images = [{"name": target, "data": _rgb_u8(observation[source])}
                  for source, target in ROLE_MAP.items()]
        state = np.ascontiguousarray(observation["observation.state"], dtype=np.float32).reshape(-1)
        rpc_start = time.perf_counter()
        action, stats = self.rpc.predict(
            images, state, instruction, action_noise=self.action_noise)
        rpc_end = time.perf_counter()
        episode_request_index = self.episode_request_counts.get(
            self.episode_index, 0)
        self.episode_request_counts[self.episode_index] = episode_request_index + 1
        server_timing = {field.name: getattr(stats, field.name)
                         for field in stats.DESCRIPTOR.fields
                         if field.cpp_type != field.CPPTYPE_MESSAGE}
        metric = {
            "episode_index": self.episode_index,
            "episode_request_index": episode_request_index,
            "request_index": len(self.request_metrics),
            "rpc_roundtrip_milliseconds": (rpc_end - rpc_start) * 1000.0,
            "client_infer_milliseconds": (rpc_end - infer_start) * 1000.0,
            "server_timing": server_timing,
        }
        self.request_metrics.append(metric)
        if self.request_log is not None:
            with self.request_log.open("a", encoding="utf-8") as output:
                output.write(json.dumps(metric) + "\n")
        return {"actions": action[:self.execute_steps],
                "server_timing": server_timing}

    def reset(self, _prompt=""):
        self.rpc.reset()
        self.last_instruction = None

    def close(self):
        self.rpc.close()


class DonorFastwamPolicy:
    def __init__(self, host, port, donor_root, execute_steps=24,
                 request_log=None):
        root = str(Path(donor_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from experiments.robotwin.remote_rpc import RpcClient as DonorRpcClient

        self.rpc = DonorRpcClient(host, port, timeout=300.0)
        self.rpc.request({"cmd": "ping"})
        self.execute_steps = execute_steps
        self.last_instruction = None
        self.episode_index = 0
        self.episode_request_counts = {}
        self.request_metrics = []
        self.request_log = request_log

    def set_episode(self, episode_index):
        self.episode_index = int(episode_index)

    def infer(self, observation):
        infer_start = time.perf_counter()
        instruction = str(observation.get("instruct", ""))
        if self.last_instruction is not None and instruction != self.last_instruction:
            self.rpc.request({"cmd": "reset"})
        self.last_instruction = instruction
        donor_observation = {
            "observation": {
                "head_camera": {
                    "rgb": observation["observation.images.cam_high"]},
                "left_camera": {
                    "rgb": observation["observation.images.cam_left_wrist"]},
                "right_camera": {
                    "rgb": observation["observation.images.cam_right_wrist"]},
            },
            "joint_action": {"vector": observation["observation.state"]},
        }
        actions = []
        first_action_milliseconds = 0.0
        for step in range(self.execute_steps):
            request_start = time.perf_counter()
            response = self.rpc.request({
                "cmd": "predict_action",
                "observation": donor_observation if step == 0 else None,
                "instruction": instruction,
            })
            if step == 0:
                first_action_milliseconds = (
                    time.perf_counter() - request_start) * 1000.0
            actions.append(np.asarray(response["action"], dtype=np.float32))
        infer_end = time.perf_counter()
        episode_request_index = self.episode_request_counts.get(
            self.episode_index, 0)
        self.episode_request_counts[self.episode_index] = episode_request_index + 1
        metric = {
            "episode_index": self.episode_index,
            "episode_request_index": episode_request_index,
            "request_index": len(self.request_metrics),
            "rpc_roundtrip_milliseconds": (infer_end - infer_start) * 1000.0,
            "client_infer_milliseconds": (infer_end - infer_start) * 1000.0,
            "donor_first_action_milliseconds": first_action_milliseconds,
            "server_timing": {},
        }
        self.request_metrics.append(metric)
        if self.request_log is not None:
            with self.request_log.open("a", encoding="utf-8") as output:
                output.write(json.dumps(metric) + "\n")
        return {"actions": np.stack(actions, axis=0), "server_timing": {}}

    def reset(self, _prompt=""):
        self.rpc.request({"cmd": "reset"})
        self.last_instruction = None

    def close(self):
        pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--donor-fastwam-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--policy-name", default="wam05")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18060)
    parser.add_argument("--client-gpu", type=int, default=0)
    parser.add_argument("--execute-steps", type=int, default=48)
    parser.add_argument("--fixed-action-noise-seed", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--metrics-output", type=Path)
    return parser.parse_args(argv)


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "stddev": float(array.std()),
        "minimum": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def _latency_distributions(requests):
    def server_values(field):
        return [item["server_timing"][field] for item in requests
                if field in item["server_timing"]]

    return {
        "rpc_roundtrip": _distribution([
            item["rpc_roundtrip_milliseconds"] for item in requests]),
        "client_infer": _distribution([
            item["client_infer_milliseconds"] for item in requests]),
        "server_total": _distribution(server_values("total_milliseconds")),
        "server_model": _distribution(server_values("model_milliseconds")),
        "server_preprocess": _distribution(
            server_values("preprocess_milliseconds")),
        "server_postprocess": _distribution(
            server_values("postprocess_milliseconds")),
        "donor_first_action": _distribution([
            item["donor_first_action_milliseconds"] for item in requests
            if "donor_first_action_milliseconds" in item]),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.donor_fastwam_root is None and args.descriptor is None:
        raise ValueError("--descriptor is required for the wam.cpp backend")
    descriptor = args.descriptor.resolve() if args.descriptor is not None else None
    donor_root = (args.donor_fastwam_root.resolve()
                  if args.donor_fastwam_root is not None else None)
    requested_metrics_output = (args.metrics_output.resolve()
                                if args.metrics_output is not None else None)
    root = args.robotwin_root.resolve()
    config_path = (args.config.resolve() if args.config is not None else
                   root / "policy/giga-pi0/deploy_policy.yml")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.client_gpu)
    os.chdir(root)
    sys.path.insert(0, str(root))
    import inference_client_robotwin as upstream
    from script.test_render import Sapien_TEST

    policy_holder = {}
    evaluation = {}
    request_log = (requested_metrics_output.with_suffix(".requests.jsonl")
                   if requested_metrics_output else None)
    if request_log is not None:
        request_log.parent.mkdir(parents=True, exist_ok=True)
        request_log.unlink(missing_ok=True)

    class BoundPolicy:
        def __init__(self, host, port):
            if donor_root is not None:
                self.impl = DonorFastwamPolicy(
                    host, port, donor_root,
                    args.execute_steps, request_log)
            else:
                self.impl = RobotwinPolicy(
                    host, port, descriptor,
                    args.execute_steps, request_log,
                    args.fixed_action_noise_seed)
            policy_holder["policy"] = self.impl

        def infer(self, observation):
            return self.impl.infer(observation)

        def reset(self, prompt=""):
            return self.impl.reset(prompt)

        def close(self):
            return self.impl.close()

    upstream.WebsocketClientPolicy = BoundPolicy
    upstream_eval_func = upstream.eval_func

    def eval_func_with_episode_reset(task_env, model, observation, instruction,
                                     save_video=False):
        model.impl.set_episode(task_env.test_num)
        if task_env.take_action_cnt == 0:
            model.reset(instruction)
        return upstream_eval_func(task_env, model, observation, instruction,
                                  save_video)

    upstream.eval_func = eval_func_with_episode_reset
    upstream_eval_policy = upstream.eval_policy

    def eval_policy_with_summary(*policy_args, **policy_kwargs):
        result = upstream_eval_policy(*policy_args, **policy_kwargs)
        evaluation["successes"] = int(result[1])
        evaluation["records"] = result[2]
        evaluation["failures"] = [
            {
                "episode_number": int(number),
                "episode_index": int(number) - 1,
                "seed": int(seed),
            }
            for number, seed in result[3]
        ]
        return result

    upstream.eval_policy = eval_policy_with_summary
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config.update({
        "task_name": args.task,
        "task_config": args.task_config,
        "instruction_type": args.instruction_type,
        "save_root": str(root / "logs/single_task_eval"),
        "test_num": args.episodes,
        "seed": args.seed,
        "policy_name": args.policy_name,
        "host": args.host,
        "port": args.port,
        "ckpt_setting": "none",
        "save_video": args.save_video,
    })
    Sapien_TEST()
    upstream.main(config)

    policy = policy_holder["policy"]
    manifest = (root / "logs/single_task_eval/eval_result" / args.task /
                args.policy_name / args.task_config / "none/manifest" /
                f"eval_manifest_seed{args.seed}_test{args.episodes}_"
                f"{args.instruction_type}.json")
    metrics_output = requested_metrics_output or (
        root / "logs/single_task_eval/eval_result" / args.task /
        args.policy_name / args.task_config / "none/metrics" /
        f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    requests = policy.request_metrics
    first_requests = [
        item for item in requests if item["episode_request_index"] == 0]
    second_requests = [
        item for item in requests if item["episode_request_index"] == 1]
    steady_requests = [
        item for item in requests if item["episode_request_index"] >= 2]
    report = {
        "format": "wam-robotwin-eval-metrics-v2",
        "backend": ("fastwam_donor" if args.donor_fastwam_root is not None
                    else "wam_cpp"),
        "task": args.task,
        "task_config": args.task_config,
        "instruction_type": args.instruction_type,
        "policy_name": args.policy_name,
        "episodes": args.episodes,
        "successes": evaluation["successes"],
        "success_rate": evaluation["successes"] / args.episodes,
        "execute_steps": args.execute_steps,
        "fixed_action_noise_seed": args.fixed_action_noise_seed,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "request_count": len(requests),
        "peak_device_memory_bytes": max(
            (item["server_timing"].get("peak_device_memory_bytes", 0)
             for item in requests),
            default=0),
        "request_log": str(request_log) if request_log else "",
        "latency_milliseconds": _latency_distributions(requests),
        "latency_groups_milliseconds": {
            "first_after_reset": _latency_distributions(first_requests),
            "second_graph_setup": _latency_distributions(second_requests),
            "steady_state": _latency_distributions(steady_requests),
        },
        "failures": evaluation["failures"],
        "requests": requests,
    }
    metrics_output.write_text(json.dumps(report, indent=2) + "\n",
                              encoding="utf-8")
    print(f"[Eval] Metrics saved to {metrics_output}")


if __name__ == "__main__":
    main()
