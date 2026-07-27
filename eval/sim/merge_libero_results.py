"""Merge non-overlapping LIBERO result shards for one frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_libero_client import (
    _load_jsonl,
    _manifest_hash,
    _summary,
    _write_json_atomic,
    _write_jsonl_atomic,
    validate_manifest,
)


def merge(manifest_path, shard_paths, output_dir):
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    digest = _manifest_hash(manifest)
    planned = manifest["episodes"]
    ordinal = {entry["episode_id"]: index
               for index, entry in enumerate(planned)}
    coverage = [False] * len(planned)
    episodes = {}
    requests = []
    request_keys = set()
    model = None

    for shard in shard_paths:
        snapshot = json.loads((shard / "manifest.json").read_text())
        if _manifest_hash(snapshot) != digest:
            raise ValueError(f"shard uses a different manifest: {shard}")
        selection = json.loads((shard / "selection.json").read_text())
        start = int(selection["episode_start"])
        end = int(selection["episode_end"])
        if not 0 <= start < end <= len(planned):
            raise ValueError(f"shard selection is invalid: {shard}")
        if any(coverage[start:end]):
            raise ValueError(f"shard selections overlap: {shard}")
        coverage[start:end] = [True] * (end - start)

        shard_summary = json.loads((shard / "summary.json").read_text())
        if shard_summary.get("manifest_sha256") != digest:
            raise ValueError(f"shard summary digest differs: {shard}")
        if model is None:
            model = shard_summary["model"]
        elif shard_summary.get("model") != model:
            raise ValueError(f"shard model identity differs: {shard}")

        selected_ids = {entry["episode_id"] for entry in planned[start:end]}
        shard_episodes = _load_jsonl(shard / "episodes.jsonl")
        shard_requests = _load_jsonl(shard / "requests.jsonl")
        for episode in shard_episodes:
            identifier = episode.get("episode_id")
            if identifier not in selected_ids or episode.get("status") != "completed":
                raise ValueError(f"shard contains an invalid episode: {identifier}")
            if identifier in episodes:
                raise ValueError(f"duplicate completed episode: {identifier}")
            episodes[identifier] = episode
        requests_by_episode = {}
        for request in shard_requests:
            identifier = request.get("episode_id")
            if identifier not in selected_ids:
                raise ValueError("shard request is outside its selection")
            key = (identifier, int(request["request_index"]))
            if key in request_keys:
                raise ValueError(f"duplicate request record: {key}")
            request_keys.add(key)
            requests_by_episode.setdefault(identifier, []).append(key[1])
            requests.append(request)
        for episode in shard_episodes:
            identifier = episode["episode_id"]
            expected = int(episode["request_count"])
            actual = sorted(requests_by_episode.get(identifier, []))
            if actual != list(range(expected)):
                raise ValueError(
                    f"episode request records are incomplete: {identifier}")

    if not all(coverage):
        raise ValueError("shard selections do not cover the complete manifest")
    if set(episodes) != set(ordinal):
        missing = sorted(set(ordinal) - set(episodes))
        raise ValueError(f"shards have incomplete episodes: {missing[:3]}")

    ordered_episodes = sorted(episodes.values(),
                              key=lambda item: ordinal[item["episode_id"]])
    requests.sort(key=lambda item: (
        ordinal[item["episode_id"]], int(item["request_index"])))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite merge output: {output_dir}")
    output_dir.mkdir(parents=True)
    _write_json_atomic(output_dir / "manifest.json", manifest)
    _write_jsonl_atomic(output_dir / "episodes.jsonl", ordered_episodes)
    _write_jsonl_atomic(output_dir / "requests.jsonl", requests)
    summary = _summary(
        manifest, digest, model, ordered_episodes, requests)
    summary["source_shards"] = [str(path) for path in shard_paths]
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = merge(args.manifest.resolve(),
                    [path.resolve() for path in args.shard],
                    args.output_dir.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
