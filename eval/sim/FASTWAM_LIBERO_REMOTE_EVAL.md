# FastWAM LIBERO Remote Evaluation

This path runs the wam.cpp FastWAM model server and the upstream LIBERO simulator in separate
containers. WebSocket/Protobuf RPC carries raw RGB images, state, instruction, and optional
action noise. The server owns prompt formatting, UMT5 tokenization and text encoding; the client
never sends embeddings.

## Frozen Gate

- Policy profile: `fastwam_libero_2cam224_minmax`
- LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Images: `scene`, `wrist`
- State: 8D end-effector pose and gripper state
- Action: `[32, 7]` robot-base end-effector delta pose
- Language: Wan UMT5 BF16 `[128, 4096]` external embedding
- Default execution: 30 wait steps, 10 actions per replan

The UMT5 implementation and weights are deployment dependencies. They are not copied into the
FastWAM GGUF and are not tracked by wam.cpp Git.

## Server

Run in `gwp_zyj`. Select a GPU with enough memory for both the 11.32 GiB FastWAM GGUF and the
approximately 11 GiB UMT5 encoder.

```bash
docker exec -it gwp_zyj bash
cd /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$PWD/eval

python -u eval/sim/run_libero_server.py \
  --library build-fastwam-cuda-release/libwam_c_api.so \
  --descriptor build-fastwam-cuda-release/wam.desc \
  --model build-fastwam-gate-b/fastwam-libero-policy-v2.gguf \
  --tokenizer /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl \
  --text-encoder /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors \
  --language-python-root /testessfs10/users/yejun.zeng/codes/FastWAM/src \
  --language-cache-capacity 32 \
  --host 0.0.0.0 \
  --port 18160 \
  --device 0 \
  --random-seed 0
```

Find the server container address from the host:

```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' gwp_zyj
```

## Single Episode Client

Run in a container with LIBERO, robosuite and MuJoCo. The client creates a temporary upstream
LIBERO path configuration when `LIBERO_CONFIG_PATH` is not already set.

```bash
docker exec -it liberox bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate liberox
export MUJOCO_GL=egl

python -u /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/eval/sim/run_libero_client.py \
  --libero-root /testessfs10/users/yejun.zeng/codes/vla.cpp/third_party/LIBERO \
  --descriptor /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/build-fastwam-cuda-release/wam.desc \
  --suite libero_spatial \
  --task-id 0 \
  --episode-index 0 \
  --seed 0 \
  --host 172.17.0.6 \
  --port 18160 \
  --replan-steps 10 \
  --num-steps-wait 30 \
  --metrics-output /tmp/fastwam-libero-single.json
```

Replace `172.17.0.6` with the address reported for `gwp_zyj`. A successful protocol connection
still performs the server-side PolicySpec/EnvironmentContract check before creating a session.

## Verified Result

The Gate B smoke used `libero_spatial`, task 0, init-state 0, seed 0. It succeeded in 85 control
steps with 9 RPC requests. Server total latency across all requests had mean 749.42 ms and median
649.62 ms. The first request included 405.42 ms of UMT5 execution; subsequent requests hit the
prompt cache. This result verifies integration only and does not replace a fixed-manifest
multi-episode success-rate evaluation.

## Fixed Manifest Evaluation

Generate the manifest in the LIBERO container. Creation validates every selected task and its
available init-state count. The manifest freezes simulator seed, explicit action-noise seed,
image resolution, wait steps, replan steps, gripper conversion and maximum episode steps.

```bash
python -u /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/eval/sim/run_libero_client.py \
  --libero-root /testessfs10/users/yejun.zeng/codes/vla.cpp/third_party/LIBERO \
  --create-manifest /tmp/libero-spatial-task0-20.json \
  --suite libero_spatial \
  --task-ids 0 \
  --episodes 20 \
  --base-seed 0 \
  --action-noise-seed 100000 \
  --replan-steps 10 \
  --num-steps-wait 30
```

`--task-ids` accepts comma-separated ids and inclusive ranges, for example `0,2-4`. Run the
frozen manifest after starting the same server command shown above:

```bash
python -u /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/eval/sim/run_libero_client.py \
  --libero-root /testessfs10/users/yejun.zeng/codes/vla.cpp/third_party/LIBERO \
  --descriptor /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.5/build-gate-b-serving-checkpoint-cuda/wam.desc \
  --manifest /tmp/libero-spatial-task0-20.json \
  --output-dir /tmp/libero-spatial-task0-20-results \
  --host 172.17.0.6 \
  --port 18160
```

The output directory contains:

```text
manifest.json
episodes.jsonl
requests.jsonl
errors.jsonl       # created only after an episode error
summary.json
```

Each completed episode and request is flushed before the next episode. To continue an interrupted
run, repeat the command with `--resume`. Resume verifies the manifest hash and model identity,
removes orphan request records, and skips only episodes with `status=completed`.

The runner sends deterministic explicit action noise with shape `[action.horizon,
action.model_dim]`. This keeps manifest evaluation independent of the server session RNG while
still calling `Reset` before every episode.

The two-episode runner smoke used task 0, init states 0 and 1, and manifest SHA256
`6a5e7bbe738f89fbe6e27f99a9618f0070990b47385dbed554cc36a15f334a7f`. Both episodes
succeeded in 85 and 94 controller steps. The run produced 19 requests, server-total mean
658.23 ms and median 601.50 ms. A subsequent `--resume` invocation skipped both episodes and
preserved the same counts and summary.

The frozen task-0 formal manifest has SHA256
`dee7258aa9ab60be52d4a62ba69a30b2ebacb36259bab66690bca44aa16a7d3a`.
It completed `18/20` episodes successfully (`90.0%`) with 247 action-chunk requests.
Server-total latency had mean 581.49 ms, P50 567.87 ms and P95 650.54 ms. RPC
round-trip latency had mean 586.52 ms, P50 572.99 ms and P95 655.85 ms.

For a long manifest, non-overlapping ordinal ranges may run in separate output
directories and against separate servers:

```bash
python eval/sim/run_libero_client.py ... --manifest MANIFEST \
  --output-dir SHARD_0 --episode-start 0 --episode-end 50
```

Every shard snapshots the complete manifest and a `selection.json`. Merge only
after all ranges have completed:

```bash
python eval/sim/merge_libero_results.py \
  --manifest MANIFEST \
  --shard SHARD_0 --shard SHARD_1 --shard SHARD_2 --shard SHARD_3 \
  --output-dir MERGED_RESULTS
```

The merger rejects different manifest hashes, model identities, overlapping or
missing ranges, duplicate episodes and requests outside a shard selection.
The merged summary includes suite-level and per-task success rates, request
counts and latency distributions.

## Formal Full-Suite Result

The standard `libero_spatial`, `libero_object`, `libero_goal` and `libero_10`
suites were evaluated with 20 fixed init states for each of their 10 tasks.
The four frozen manifests therefore contain 800 episodes in total. Long runs
used non-overlapping shards against four identical servers. The RPC latency
includes concurrent-server queueing; server-total latency measures the server's
own request execution.

| Suite | Manifest SHA256 | Success | Requests | Server total mean/P50/P95 (ms) | RPC mean/P50/P95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| `libero_spatial` | `6b5780721be6a51ab9f79ef383ff744878bd308be48c03dadd9f690d0bed1513` | 22/200 (11.0%) | 7,333 | 645.43 / 602.35 / 1026.23 | 694.63 / 614.32 / 1101.84 |
| `libero_object` | `380cf170f6d1a9eb86946e69624d0aa61cdbdab79ab24f0845dd781d75cc4365` | 9/200 (4.5%) | 7,809 | 604.29 / 602.20 / 633.48 | 678.64 / 616.27 / 1071.41 |
| `libero_goal` | `9b9370ff6507eb8633fdfddc5340408b6ffe59e4e00d65d1555d41bca71d9542` | 29/200 (14.5%) | 7,212 | 612.64 / 598.89 / 637.02 | 745.72 / 624.43 / 1154.43 |
| `libero_10` | `2796ebab967ac1c14f21416cb6c3933272268527263fe9ccd177759ab1c7a13a` | 0/200 (0.0%) | 14,000 | 608.38 / 598.14 / 632.98 | 725.93 / 616.99 / 1141.20 |

The aggregate benchmark result is `60/800` (`7.5%`). The only non-zero
per-task results were spatial tasks 0/3/5 (`18/20`, `3/20`, `1/20`), object
tasks 1/8 (`7/20`, `2/20`), and goal tasks 1/8 (`18/20`, `11/20`). All 10
`libero_10` tasks scored `0/20`. No episode failed because of RPC, model
execution, input validation or CUDA errors. The low success rate is therefore
a model/checkpoint effectiveness result, not an integration failure.
