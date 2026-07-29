"""Frozen-manifest helpers for the upstream LIBERO-X evaluator."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from wam.eval import (canonical_json, distribution, load_jsonl)

MANIFEST_FORMAT = "wam-liberox-manifest-v1"
RESULT_FORMAT = "wam-liberox-eval-v1"
ACTION_NOISE_GENERATOR = "torch_cpu_f32_then_bf16_reset_each_predict"
LATENCY_FIELDS = (
    "rpc_roundtrip_milliseconds",
    "server_total_milliseconds",
    "server_model_milliseconds",
    "server_text_milliseconds",
)


def manifest_sha256(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _natural_key(path):
    return [int(item) if item.isdigit() else item.lower()
            for item in re.split(r"(\d+)", Path(path).name)]


def discover_tasks(bddl_root, init_root, scene_group, load_init):
    level = "LEVEL4" if scene_group.upper() == "LEVEL5" else scene_group.upper()
    bddl_dir = Path(bddl_root) / level
    init_dir = Path(init_root) / level
    bddl_files = sorted(bddl_dir.glob("*.bddl"), key=_natural_key)
    if not bddl_files:
        raise FileNotFoundError(f"no BDDL files under {bddl_dir}")
    tasks = []
    for index, bddl_path in enumerate(bddl_files):
        init_path = init_dir / f"{bddl_path.stem}.init"
        if not init_path.is_file():
            raise FileNotFoundError(f"missing init file: {init_path}")
        states = load_init(init_path)
        tasks.append({
            "task_index": index,
            "bddl_file": bddl_path.name,
            "bddl_sha256": file_sha256(bddl_path),
            "init_file": init_path.name,
            "init_sha256": file_sha256(init_path),
            "init_count": len(states),
        })
    return tasks


def build_manifest(bddl_root, init_root, scene_group, task_indices,
                   episodes_per_task, execution, load_init):
    available = discover_tasks(
        bddl_root, init_root, scene_group, load_init)
    selected = []
    for task_index in sorted(set(int(value) for value in task_indices)):
        if not 0 <= task_index < len(available):
            raise ValueError(f"task index {task_index} is outside the scene group")
        task = dict(available[task_index])
        if not 1 <= episodes_per_task <= task["init_count"]:
            raise ValueError(
                f"task {task_index} has {task['init_count']} init states, "
                f"requested {episodes_per_task}")
        task["episode_count"] = int(episodes_per_task)
        selected.append(task)
    if not selected:
        raise ValueError("manifest must select at least one task")
    return {
        "format": MANIFEST_FORMAT,
        "scene_group": scene_group.upper(),
        "execution": dict(execution),
        "tasks": selected,
    }


def validate_manifest(value):
    if value.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"manifest format must be {MANIFEST_FORMAT}")
    if value.get("scene_group") not in {
            "LEVEL1", "LEVEL2", "LEVEL3", "LEVEL4", "LEVEL5"}:
        raise ValueError("manifest scene_group must be LEVEL1-LEVEL5")
    execution = value.get("execution")
    required_execution = {
        "load_mode", "num_steps_wait", "max_steps", "resize_size",
        "replan_steps", "flip_images", "binarize_gripper",
        "action_noise_seed", "action_noise_generator",
    }
    if not isinstance(execution, dict) or set(execution) != required_execution:
        raise ValueError("manifest execution fields do not match the contract")
    if execution["load_mode"] != "init":
        raise ValueError("fixed LIBERO-X manifests require load_mode=init")
    if execution["action_noise_generator"] != ACTION_NOISE_GENERATOR:
        raise ValueError("manifest action-noise generator is incompatible")
    if min(int(execution[name]) for name in (
            "max_steps", "resize_size", "replan_steps")) <= 0:
        raise ValueError("manifest execution dimensions must be positive")
    if int(execution["num_steps_wait"]) < 0 or \
            int(execution["action_noise_seed"]) < 0:
        raise ValueError("manifest seeds and wait steps must be non-negative")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest tasks must be a non-empty list")
    required_task = {
        "task_index", "bddl_file", "bddl_sha256", "init_file",
        "init_sha256", "init_count", "episode_count",
    }
    indices = set()
    filenames = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != required_task:
            raise ValueError("manifest task fields do not match the contract")
        index = int(task["task_index"])
        if index < 0 or index in indices:
            raise ValueError(f"invalid or duplicate task index: {index}")
        indices.add(index)
        if task["bddl_file"] in filenames:
            raise ValueError(f"duplicate BDDL file: {task['bddl_file']}")
        filenames.add(task["bddl_file"])
        for field, suffix in (("bddl_file", ".bddl"),
                              ("init_file", ".init")):
            name = task[field]
            if Path(name).name != name or not name.endswith(suffix):
                raise ValueError(f"unsafe manifest filename: {name!r}")
        for field in ("bddl_sha256", "init_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", task[field]):
                raise ValueError(f"invalid SHA256 in {field}")
        count = int(task["init_count"])
        episodes = int(task["episode_count"])
        if not 1 <= episodes <= count:
            raise ValueError("episode_count must be an init-state prefix")
    return value


def select_tasks(manifest, task_start=0, task_end=None):
    tasks = manifest["tasks"]
    end = len(tasks) if task_end is None else int(task_end)
    if not 0 <= int(task_start) < end <= len(tasks):
        raise ValueError("task shard is outside the manifest")
    return tasks[int(task_start):end]


def stage_manifest(manifest, source_bddl_root, source_init_root, stage_root,
                   load_init, save_init, task_start=0, task_end=None):
    validate_manifest(manifest)
    selected = select_tasks(manifest, task_start, task_end)
    level = "LEVEL4" if manifest["scene_group"] == "LEVEL5" else \
        manifest["scene_group"]
    bddl_out = Path(stage_root) / "bddl" / level
    init_out = Path(stage_root) / "init" / level
    bddl_out.mkdir(parents=True, exist_ok=True)
    init_out.mkdir(parents=True, exist_ok=True)
    expected = {task["bddl_file"] for task in selected}
    existing = {path.name for path in bddl_out.glob("*.bddl")}
    if existing - expected:
        raise ValueError("staging BDDL directory contains unselected tasks")
    expected_init = {task["init_file"] for task in selected}
    existing_init = {path.name for path in init_out.glob("*.init")}
    if existing_init - expected_init:
        raise ValueError("staging init directory contains unselected tasks")
    for task in selected:
        source_bddl = Path(source_bddl_root) / level / task["bddl_file"]
        source_init = Path(source_init_root) / level / task["init_file"]
        if file_sha256(source_bddl) != task["bddl_sha256"]:
            raise ValueError(f"BDDL source hash changed: {source_bddl}")
        if file_sha256(source_init) != task["init_sha256"]:
            raise ValueError(f"init source hash changed: {source_init}")
        states = load_init(source_init)
        if len(states) != int(task["init_count"]):
            raise ValueError(f"init-state count changed: {source_init}")
        (bddl_out / task["bddl_file"]).write_bytes(source_bddl.read_bytes())
        destination = init_out / task["init_file"]
        save_init(states[:int(task["episode_count"])], destination)
    return Path(stage_root) / "bddl", Path(stage_root) / "init", selected


def validate_results(results, selected_tasks):
    planned = {(task["bddl_file"], episode)
               for task in selected_tasks
               for episode in range(int(task["episode_count"]))}
    observed = []
    for result in results:
        observed.append((Path(result["task_file"]).name,
                         int(result["episode_index"])))
    if len(observed) != len(set(observed)):
        raise ValueError("results contain duplicate episodes")
    if set(observed) != planned:
        missing = sorted(planned - set(observed))
        extra = sorted(set(observed) - planned)
        raise ValueError(f"results differ from manifest; missing={missing}, extra={extra}")


def build_summary(manifest, model, results, requests, selected_tasks):
    successes = sum(bool(result["success"]) for result in results)
    return {
        "format": RESULT_FORMAT,
        "manifest_sha256": manifest_sha256(manifest),
        "model": model,
        "planned_episodes": sum(int(task["episode_count"])
                                for task in selected_tasks),
        "completed_episodes": len(results),
        "successes": successes,
        "success_rate": successes / len(results) if results else 0.0,
        "request_count": len(requests),
        "latency": {
            field: distribution([request[field] for request in requests])
            for field in LATENCY_FIELDS
        },
    }
