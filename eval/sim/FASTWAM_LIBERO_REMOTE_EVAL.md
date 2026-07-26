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
