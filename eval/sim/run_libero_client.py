"""Run fixed LIBERO episodes against a wam.cpp RPC server."""

from __future__ import annotations

import argparse
import hashlib
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
MANIFEST_FORMAT = "wam-libero-manifest-v3"
RESULT_FORMAT = "wam-libero-eval-v1"
LATENCY_FIELDS = (
    "rpc_roundtrip_milliseconds", "server_total_milliseconds",
    "server_model_milliseconds", "server_text_milliseconds")


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


def _parse_task_ids(value):
    result = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("task-id list contains an empty item")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending task-id range: {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def _canonical_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) +
            "\n").encode()


def _manifest_hash(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def build_manifest(suite_name, task_ids, episodes, environment_seed,
                   action_noise_seed, execution, available_init_states):
    execution = dict(execution)
    execution.update({
        "environment_seed": int(environment_seed),
        "action_noise_seed": int(action_noise_seed),
        "action_noise_generator":
            "torch_cpu_f32_then_bf16_reset_each_predict",
    })
    entries = []
    for task_id in task_ids:
        count = int(available_init_states[task_id])
        if episodes > count:
            raise ValueError(
                f"task {task_id} has {count} init states, requested {episodes}")
        for episode_index in range(episodes):
            entries.append({
                "episode_id": (f"{suite_name}-task{task_id:02d}-"
                               f"episode{episode_index:03d}"),
                "task_id": task_id,
                "episode_index": episode_index,
            })
    return {"format": MANIFEST_FORMAT, "suite": suite_name,
            "execution": execution, "episodes": entries}


def validate_manifest(value):
    if value.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"manifest format must be {MANIFEST_FORMAT}")
    suite = value.get("suite")
    if suite not in MAX_STEPS:
        raise ValueError(f"manifest suite is invalid: {suite!r}")
    execution = value.get("execution")
    required_execution = {
        "replan_steps", "num_steps_wait", "max_steps", "resolution",
        "binarize_gripper", "explicit_action_noise", "environment_seed",
        "action_noise_seed", "action_noise_generator"}
    if not isinstance(execution, dict) or set(execution) != required_execution:
        raise ValueError("manifest execution fields do not match the v2 contract")
    if not execution["explicit_action_noise"]:
        raise ValueError("fixed LIBERO manifests require explicit action noise")
    if execution["action_noise_generator"] != \
            "torch_cpu_f32_then_bf16_reset_each_predict":
        raise ValueError(
            "LIBERO donor parity requires reset-per-predict PyTorch F32-to-BF16 noise")
    if min(int(execution[name]) for name in (
            "replan_steps", "max_steps", "resolution")) <= 0:
        raise ValueError("manifest execution dimensions must be positive")
    if int(execution["num_steps_wait"]) < 0:
        raise ValueError("manifest num_steps_wait must be non-negative")
    entries = value.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest episodes must be a non-empty list")
    if min(int(execution[name]) for name in (
            "environment_seed", "action_noise_seed")) < 0:
        raise ValueError("manifest seeds must be non-negative")
    required_entry = {"episode_id", "task_id", "episode_index"}
    identifiers = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != required_entry:
            raise ValueError(f"manifest episode {index} fields are invalid")
        identifier = entry["episode_id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"manifest episode {index} has no identifier")
        if identifier in identifiers:
            raise ValueError(f"duplicate manifest episode_id: {identifier}")
        identifiers.add(identifier)
        if min(int(entry[name]) for name in (
                "task_id", "episode_index")) < 0:
            raise ValueError(f"manifest episode {identifier} has a negative value")
    return value


def donor_action_noise(seed, horizon, dimension):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return (torch.randn((int(horizon), int(dimension)), generator=generator,
                        device="cpu", dtype=torch.float32)
            .to(dtype=torch.bfloat16).to(dtype=torch.float32).numpy())


def validate_task_aligned_selection(entries, start, end):
    if start > 0 and entries[start]["task_id"] == entries[start - 1]["task_id"]:
        raise ValueError("episode shard must start at a LIBERO task boundary")
    if end < len(entries) and entries[end]["task_id"] == entries[end - 1]["task_id"]:
        raise ValueError("episode shard must end at a LIBERO task boundary")


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "stddev": float(array.std()), "minimum": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def _load_jsonl(path):
    if not path.exists():
        return []
    result = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {error}") from error
    return result


def _append_jsonl(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _write_jsonl_atomic(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def _model_identity(info):
    return {
        "architecture": info.architecture,
        "artifact_policy": info.artifact_policy,
        "artifact_sha256": info.artifact_sha256,
        "profile": info.policy_spec.profile,
        "checkpoint_revision": info.policy_spec.checkpoint_revision,
    }


def _validate_runtime(info, execution):
    replan_steps = int(execution["replan_steps"])
    if not 1 <= replan_steps <= info.policy_spec.action.horizon:
        raise ValueError(
            f"replan_steps must be within [1,{info.policy_spec.action.horizon}]")
    if execution["explicit_action_noise"] and not \
            info.capabilities.explicit_action_noise:
        raise ValueError("server does not support explicit action noise")


def _summary(manifest, manifest_sha256, model_identity, episodes, requests):
    successes = sum(bool(item["success"]) for item in episodes)
    task_summaries = []
    task_ids = sorted({int(item["task_id"]) for item in episodes
                       if "task_id" in item})
    for task_id in task_ids:
        task_episodes = [item for item in episodes
                         if int(item.get("task_id", -1)) == task_id]
        episode_ids = {item["episode_id"] for item in task_episodes}
        task_requests = [item for item in requests
                         if item.get("episode_id") in episode_ids]
        task_successes = sum(bool(item["success"]) for item in task_episodes)
        task_summaries.append({
            "task_id": task_id,
            "task": task_episodes[0].get("task", ""),
            "instruction": task_episodes[0].get("instruction", ""),
            "completed_episodes": len(task_episodes),
            "successes": task_successes,
            "success_rate": task_successes / len(task_episodes),
            "request_count": len(task_requests),
            "latency": {
                name: _distribution([item[name] for item in task_requests])
                for name in LATENCY_FIELDS},
        })
    return {
        "format": RESULT_FORMAT,
        "manifest_sha256": manifest_sha256,
        "suite": manifest["suite"],
        "model": model_identity,
        "planned_episodes": len(manifest["episodes"]),
        "completed_episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "request_count": len(requests),
        "text_encoder_request_count": sum(
            item["server_text_milliseconds"] > 0.0 for item in requests),
        "latency": {name: _distribution([item[name] for item in requests])
                    for name in LATENCY_FIELDS},
        "tasks": task_summaries,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18060)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--create-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int)
    parser.add_argument("--suite", choices=tuple(MAX_STEPS), default="libero_spatial")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--action-noise-seed", type=int, default=42)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args(argv)
    if args.manifest and args.create_manifest:
        parser.error("--manifest and --create-manifest are mutually exclusive")
    if args.manifest and args.output_dir is None:
        parser.error("--manifest requires --output-dir")
    if not args.create_manifest and args.descriptor is None:
        parser.error("--descriptor is required when running evaluation")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.episode_start < 0:
        parser.error("--episode-start must be non-negative")
    if args.episode_end is not None and args.episode_end <= args.episode_start:
        parser.error("--episode-end must be greater than --episode-start")
    return args


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


class EpisodeRunner:
    def __init__(self, suite, get_libero_path, env_type, rpc, info, execution,
                 video_dir=None):
        self.suite = suite
        self.get_libero_path = get_libero_path
        self.env_type = env_type
        self.rpc = rpc
        self.info = info
        self.execution = execution
        self.video_dir = video_dir
        self.env = None
        self.task_id = None
        self.task = None
        self.init_states = None

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def _select_task(self, task_id):
        if self.task_id == task_id:
            return
        self.close()
        if not 0 <= task_id < self.suite.n_tasks:
            raise ValueError(f"task_id {task_id} is outside this suite")
        self.task_id = task_id
        self.task = self.suite.get_task(task_id)
        self.init_states = self.suite.get_task_init_states(task_id)
        bddl = (Path(self.get_libero_path("bddl_files")) /
                self.task.problem_folder / self.task.bddl_file)
        self.env = self.env_type(
            bddl_file_name=str(bddl),
            camera_heights=int(self.execution["resolution"]),
            camera_widths=int(self.execution["resolution"]))
        self.env.seed(int(self.execution["environment_seed"]))

    def run(self, entry, video_output=None):
        self._select_task(int(entry["task_id"]))
        episode_index = int(entry["episode_index"])
        if not 0 <= episode_index < len(self.init_states):
            raise ValueError(
                f"episode_index {episode_index} is unavailable for task {self.task_id}")
        self.rpc.reset()
        self.env.reset()
        observation = self.env.set_init_state(self.init_states[episode_index])
        writer = None
        if video_output is None and self.video_dir is not None:
            video_output = self.video_dir / f"{entry['episode_id']}.mp4"
        if video_output is not None:
            import imageio.v2 as imageio
            video_output.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video_output), fps=20)
        requests = []
        started = time.perf_counter()
        try:
            done = False
            for _ in range(int(self.execution["num_steps_wait"])):
                if writer:
                    writer.append_data(observation["agentview_image"][::-1, ::-1])
                observation, _, done, _ = self.env.step(
                    [0, 0, 0, 0, 0, 0, -1])
                if done:
                    break
            steps = 0
            limit = int(self.execution["max_steps"])
            while not done and steps < limit:
                images, state = observation_to_policy_observation(
                    observation, self.info.policy_spec)
                noise = donor_action_noise(
                    self.execution["action_noise_seed"],
                    self.info.policy_spec.action.horizon,
                    self.info.policy_spec.action.model_dim)
                request_started = time.perf_counter()
                chunk, stats = self.rpc.predict(
                    images, state, self.task.language, noise)
                roundtrip = (time.perf_counter() - request_started) * 1000.0
                requests.append({
                    "episode_id": entry["episode_id"],
                    "request_index": len(requests),
                    "rpc_roundtrip_milliseconds": roundtrip,
                    "server_total_milliseconds": stats.total_milliseconds,
                    "server_model_milliseconds": stats.model_milliseconds,
                    "server_text_milliseconds": stats.model_text_milliseconds,
                    "peak_device_memory_bytes": stats.peak_device_memory_bytes,
                })
                for action in chunk[:int(self.execution["replan_steps"])]:
                    command = policy_action_to_command(
                        action, self.info.policy_spec,
                        bool(self.execution["binarize_gripper"]))
                    observation, _, done, _ = self.env.step(command)
                    steps += 1
                    if writer:
                        writer.append_data(
                            observation["agentview_image"][::-1, ::-1])
                    if done or steps >= limit:
                        break
            episode = dict(entry)
            episode.update({
                "task": self.task.name, "instruction": self.task.language,
                "success": bool(done), "steps": steps,
                "request_count": len(requests),
                "elapsed_milliseconds": (time.perf_counter() - started) * 1000.0,
            })
            return episode, requests
        finally:
            if writer:
                writer.close()


def _execution(args):
    return {
        "replan_steps": args.replan_steps,
        "num_steps_wait": args.num_steps_wait,
        "max_steps": args.max_steps or MAX_STEPS[args.suite],
        "resolution": args.resolution,
        "binarize_gripper": not args.no_binarize_gripper,
        "explicit_action_noise": True,
    }


def _create_manifest(args, suite):
    task_ids = _parse_task_ids(args.task_ids)
    if any(task_id < 0 or task_id >= suite.n_tasks for task_id in task_ids):
        raise ValueError(f"task ids must be within [0,{suite.n_tasks - 1}]")
    available = {task_id: len(suite.get_task_init_states(task_id))
                 for task_id in task_ids}
    manifest = build_manifest(
        args.suite, task_ids, args.episodes, args.base_seed,
        args.action_noise_seed, _execution(args), available)
    validate_manifest(manifest)
    if args.create_manifest.exists():
        raise FileExistsError(f"manifest already exists: {args.create_manifest}")
    args.create_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.create_manifest.write_bytes(_canonical_json(manifest))
    print(f"wrote {len(manifest['episodes'])} episodes to {args.create_manifest}")
    print(f"sha256={_manifest_hash(manifest)}")


def _run_manifest(args, manifest, suite, get_libero_path, env_type):
    validate_manifest(manifest)
    manifest_sha256 = _manifest_hash(manifest)
    episode_end = (len(manifest["episodes"]) if args.episode_end is None
                   else args.episode_end)
    if episode_end > len(manifest["episodes"]):
        raise ValueError("episode range exceeds the manifest")
    validate_task_aligned_selection(
        manifest["episodes"], args.episode_start, episode_end)
    selection = {"episode_start": args.episode_start,
                 "episode_end": episode_end}
    selected_entries = manifest["episodes"][args.episode_start:episode_end]
    output_dir = args.output_dir.resolve()
    episode_path = output_dir / "episodes.jsonl"
    request_path = output_dir / "requests.jsonl"
    error_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"
    snapshot_path = output_dir / "manifest.json"
    selection_path = output_dir / "selection.json"
    existing_episodes = _load_jsonl(episode_path)
    existing_requests = _load_jsonl(request_path)
    if (existing_episodes or existing_requests or summary_path.exists()) and not args.resume:
        raise FileExistsError(
            f"evaluation output already exists; use --resume: {output_dir}")
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        if _manifest_hash(snapshot) != manifest_sha256:
            raise ValueError("output directory belongs to a different manifest")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(_canonical_json(manifest))
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("output directory belongs to a different episode range")
    else:
        _write_json_atomic(selection_path, selection)
    completed = {item["episode_id"] for item in existing_episodes
                 if item.get("status") == "completed"}
    retained_requests = [item for item in existing_requests
                         if item.get("episode_id") in completed]
    if len(retained_requests) != len(existing_requests):
        existing_requests = retained_requests
        _write_jsonl_atomic(request_path, existing_requests)

    rpc = RpcClient(args.host, args.port, args.descriptor, "libero")
    runner = None
    try:
        info = rpc.connect()
        check_observation_compatibility(info.policy_spec)
        check_action_compatibility(info.policy_spec)
        execution = manifest["execution"]
        _validate_runtime(info, execution)
        identity = _model_identity(info)
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            if previous.get("manifest_sha256") != manifest_sha256:
                raise ValueError("summary belongs to a different manifest")
            if previous.get("model") != identity:
                raise ValueError("resume server model differs from existing results")
        runner = EpisodeRunner(
            suite, get_libero_path, env_type, rpc, info, execution,
            args.video_dir)
        for entry in selected_entries:
            if entry["episode_id"] in completed:
                print(f"skip completed {entry['episode_id']}")
                continue
            print(f"start {entry['episode_id']}", flush=True)
            try:
                episode, requests = runner.run(entry)
            except Exception as error:
                failure = dict(entry)
                failure.update({"success": False, "status": "error",
                                "error": f"{type(error).__name__}: {error}"})
                _append_jsonl(error_path, failure)
                _write_json_atomic(summary_path, _summary(
                    manifest, manifest_sha256, identity,
                    existing_episodes, existing_requests))
                raise
            episode["status"] = "completed"
            for request in requests:
                _append_jsonl(request_path, request)
            _append_jsonl(episode_path, episode)
            existing_requests.extend(requests)
            existing_episodes.append(episode)
            completed.add(entry["episode_id"])
            _write_json_atomic(summary_path, _summary(
                manifest, manifest_sha256, identity,
                existing_episodes, existing_requests))
            print(f"finish {entry['episode_id']} success={episode['success']} "
                  f"steps={episode['steps']} requests={episode['request_count']}",
                  flush=True)
        summary = _summary(manifest, manifest_sha256, identity,
                           existing_episodes, existing_requests)
        _write_json_atomic(summary_path, summary)
        print(json.dumps(summary, indent=2))
    finally:
        if runner is not None:
            runner.close()
        rpc.close()


def _run_single(args, suite, get_libero_path, env_type):
    execution = _execution(args)
    entry = {
        "episode_id": (f"{args.suite}-task{args.task_id:02d}-"
                       f"episode{args.episode_index:03d}"),
        "task_id": args.task_id, "episode_index": args.episode_index,
    }
    execution.update({
        "environment_seed": args.seed,
        "action_noise_seed": args.action_noise_seed,
        "action_noise_generator":
            "torch_cpu_f32_then_bf16_reset_each_predict",
    })
    rpc = RpcClient(args.host, args.port, args.descriptor, "libero")
    runner = None
    try:
        info = rpc.connect()
        check_observation_compatibility(info.policy_spec)
        check_action_compatibility(info.policy_spec)
        _validate_runtime(info, execution)
        runner = EpisodeRunner(suite, get_libero_path, env_type, rpc, info,
                               execution)
        episode, requests = runner.run(entry, args.video_output)
        report = {"format": "wam-libero-single-episode-v2",
                  "model": _model_identity(info), "episode": episode,
                  "requests": requests}
        print(json.dumps(report, indent=2))
        if args.metrics_output:
            _write_json_atomic(args.metrics_output, report)
    finally:
        if runner is not None:
            runner.close()
        rpc.close()


def main(argv=None):
    args = parse_args(argv)
    root = args.libero_root.resolve()
    sys.path.insert(0, str(root))
    libero_config = _configure_libero(root)
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite_name = args.suite
        if args.manifest:
            manifest = json.loads(args.manifest.read_text())
            validate_manifest(manifest)
            suite_name = manifest["suite"]
        suite = benchmark.get_benchmark_dict()[suite_name]()
        if args.create_manifest:
            _create_manifest(args, suite)
        elif args.manifest:
            _run_manifest(args, manifest, suite, get_libero_path,
                          OffScreenRenderEnv)
        else:
            _run_single(args, suite, get_libero_path, OffScreenRenderEnv)
    finally:
        if libero_config is not None:
            libero_config.cleanup()


if __name__ == "__main__":
    main()
