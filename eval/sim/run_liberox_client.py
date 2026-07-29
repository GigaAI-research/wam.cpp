"""Run the upstream LIBERO-X evaluator against a wam.cpp RPC server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wam.remote import Client as RpcClient
from wam.adapters.libero import ACTION_FIELDS, STATE_FIELDS
from wam.adapters.liberox import LiberoXAdapter
from wam.eval import ActionChunkExecutor
from liberox_manifest import (
    ACTION_NOISE_GENERATOR,
    build_manifest,
    build_summary,
    canonical_json,
    load_jsonl,
    manifest_sha256,
    stage_manifest,
    validate_manifest,
    validate_results,
)


class WamLiberoXPolicy:
    def __init__(self, host, port, descriptor, random_seed=7,
                 binarize_gripper=True, execute_steps=None):
        self.rpc = RpcClient(host, port, descriptor, "liberox")
        self.info = self.rpc.connect()
        self.adapter = LiberoXAdapter(binarize_gripper)
        self.adapter.validate(self.info.policy_spec)
        if not self.info.capabilities.explicit_action_noise:
            raise RuntimeError("LIBERO-X evaluation requires explicit action noise")
        self.random_seed = int(random_seed)
        self.binarize_gripper = binarize_gripper
        steps = int(execute_steps or self.info.policy_spec.action.horizon)
        self.executor = ActionChunkExecutor(
            steps, lambda chunk: LiberoXAdapter(binarize_gripper).action(
                chunk, self.info.policy_spec))
        self.last_prompt = None
        self.requests = []

    def infer(self, observation):
        prompt = str(observation.get("prompt", ""))
        if self.last_prompt is not None and prompt != self.last_prompt:
            self.rpc.reset()
        self.last_prompt = prompt
        images, state = self.adapter.observation(
            observation, self.info.policy_spec)
        action_spec = self.info.policy_spec.action
        noise = donor_action_noise(
            self.random_seed, action_spec.horizon, action_spec.model_dim)
        started = time.perf_counter()
        action, stats = self.rpc.predict(images, state, prompt, noise)
        self.requests.append({
            "request_index": len(self.requests),
            "rpc_roundtrip_milliseconds":
                (time.perf_counter() - started) * 1000.0,
            "server_total_milliseconds": stats.total_milliseconds,
            "server_model_milliseconds": stats.model_milliseconds,
            "server_text_milliseconds": stats.model_text_milliseconds,
        })
        self.executor.push(action)
        actions = []
        while self.executor.pending:
            actions.append(self.executor.next())
        return {"actions": np.asarray(actions, dtype=np.float32)}

    def get_server_metadata(self):
        return {
            "policy": self.info.architecture,
            "profile": self.info.policy_spec.profile,
            "action_horizon": self.info.policy_spec.action.horizon,
        }

    def reset(self):
        self.rpc.reset()
        self.executor.reset()
        self.last_prompt = None

    def close(self):
        self.rpc.close()


def donor_action_noise(seed, horizon, dimension):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.randn((int(horizon), int(dimension)), generator=generator,
                        device="cpu", dtype=torch.float32)
            .to(dtype=torch.bfloat16).to(dtype=torch.float32).numpy())


def _parse_indices(value):
    result = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("task-index list contains an empty item")
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if end < start:
                raise ValueError(f"descending task-index range: {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def _torch_load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_save(value, path):
    import torch
    torch.save(value, path)


def _model_identity(info):
    return {
        "architecture": info.architecture,
        "artifact_policy": info.artifact_policy,
        "artifact_bytes": info.artifact_bytes,
        "profile": info.policy_spec.profile,
        "checkpoint_revision": info.policy_spec.checkpoint_revision,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--liberox-root", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18060)
    parser.add_argument("--scene-group", default="LEVEL1")
    parser.add_argument("--load-mode", choices=("bddl", "init"), default="init")
    parser.add_argument("--bddl-root", type=Path)
    parser.add_argument("--init-root", type=Path)
    parser.add_argument("--level5-prompt-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--create-manifest", type=Path)
    parser.add_argument("--stage-manifest-only", type=Path)
    parser.add_argument("--task-indices", default="0")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int)
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
    args = parser.parse_args(argv)
    if args.manifest and args.create_manifest:
        parser.error("--manifest and --create-manifest are mutually exclusive")
    if args.stage_manifest_only and not args.manifest:
        parser.error("--stage-manifest-only requires --manifest")
    if args.create_manifest is None and args.manifest is None and \
            (args.descriptor is None or args.output_dir is None):
        parser.error("direct evaluation requires --descriptor and --output-dir")
    if args.manifest and not args.stage_manifest_only and \
            (args.descriptor is None or args.output_dir is None):
        parser.error("manifest evaluation requires --descriptor and --output-dir")
    if args.episodes_per_task <= 0:
        parser.error("--episodes-per-task must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.liberox_root.resolve()
    source_bddl_root = (args.bddl_root or root / "libero/libero_x/bddl").resolve()
    source_init_root = (args.init_root or root / "libero/libero_x/init").resolve()
    execution = {
        "load_mode": "init",
        "num_steps_wait": args.num_steps_wait,
        "max_steps": args.max_steps,
        "resize_size": args.resize_size,
        "replan_steps": args.replan_steps,
        "flip_images": not args.no_flip_images,
        "binarize_gripper": not args.no_binarize_gripper,
        "action_noise_seed": args.seed,
        "action_noise_generator": ACTION_NOISE_GENERATOR,
    }
    if args.create_manifest:
        manifest = build_manifest(
            source_bddl_root, source_init_root, args.scene_group,
            _parse_indices(args.task_indices), args.episodes_per_task,
            execution, _torch_load)
        validate_manifest(manifest)
        if args.create_manifest.exists():
            raise FileExistsError(f"manifest already exists: {args.create_manifest}")
        args.create_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.create_manifest.write_bytes(canonical_json(manifest))
        print(f"wrote {sum(task['episode_count'] for task in manifest['tasks'])} "
              f"episodes to {args.create_manifest}")
        print(f"sha256={manifest_sha256(manifest)}")
        return

    manifest = None
    selected_tasks = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        validate_manifest(manifest)
        stage_root = (args.stage_manifest_only or
                      (args.output_dir / "manifest-stage")).resolve()
        staged_bddl, staged_init, selected_tasks = stage_manifest(
            manifest, source_bddl_root, source_init_root, stage_root,
            _torch_load, _torch_save, args.task_start, args.task_end)
        if args.stage_manifest_only:
            print(f"bddl_root={staged_bddl}")
            print(f"init_root={staged_init}")
            print(f"manifest_sha256={manifest_sha256(manifest)}")
            return
        args.scene_group = manifest["scene_group"]
        args.load_mode = "init"
        args.bddl_root = staged_bddl
        args.init_root = staged_init
        args.num_steps_wait = manifest["execution"]["num_steps_wait"]
        args.max_steps = manifest["execution"]["max_steps"]
        args.resize_size = manifest["execution"]["resize_size"]
        args.replan_steps = manifest["execution"]["replan_steps"]
        args.seed = manifest["execution"]["action_noise_seed"]
        args.no_flip_images = not manifest["execution"]["flip_images"]
        args.no_binarize_gripper = not manifest["execution"]["binarize_gripper"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_bytes(canonical_json(manifest))
        (args.output_dir / "selection.json").write_bytes(canonical_json({
            "task_start": args.task_start,
            "task_end": (len(manifest["tasks"]) if args.task_end is None
                         else args.task_end),
            "tasks": selected_tasks,
        }))
        if args.results_output is None:
            args.results_output = args.output_dir / "results.jsonl"
        if args.results_output.exists():
            raise FileExistsError(
                f"results already exist; use a new output directory: "
                f"{args.results_output}")

    sys.path[:0] = [str(root), str(root / "packages/openpi-client/src")]
    policy_holder = {}

    class BoundPolicy:
        def __init__(self, host, port):
            self.impl = WamLiberoXPolicy(
                host, port, args.descriptor, args.seed,
                not args.no_binarize_gripper, args.replan_steps)
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

    bddl_root = args.bddl_root or source_bddl_root
    init_root = args.init_root or source_init_root
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
    if manifest is not None:
        results = load_jsonl(args.results_output)
        validate_results(results, selected_tasks)
        policy = policy_holder["policy"]
        summary = build_summary(
            manifest, _model_identity(policy.info), results,
            policy.requests, selected_tasks)
        (args.output_dir / "wam-summary.json").write_bytes(
            canonical_json(summary))
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
