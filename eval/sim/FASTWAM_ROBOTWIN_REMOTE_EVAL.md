# FastWAM RoboTwin Remote Evaluation

FastWAM RoboTwin uses the common FastWAM engine with a RoboTwin PolicySpec:
three named images on a 384x320 canvas, 14D dual-arm qpos state/action, a
32-step action horizon, z-score normalization and identity action recovery.

## Convert

```bash
export PYTHONPATH=$PWD/build-fastwam-cpu-clean/_deps/llama-src/gguf-py
python scripts/convert/convert_fastwam.py \
  --checkpoint /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --vae /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors \
  --norm-stats /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  --policy-profile profiles/fastwam_robotwin_3cam384_zscore.json \
  --num-inference-steps 10 \
  --out build-fastwam-gate-b/fastwam-robotwin-policy-v3.gguf
```

The audited v3 conversion contains 1741 tensors and is 12,150,404,960 bytes. Run
`wam-inspect`, `wam-validate`, and the `wam_fastwam_robotwin_artifact` CTest
before serving it.

## Server

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/python \
python -u eval/sim/run_robotwin_server.py \
  --library build-gate-b-serving-checkpoint-cuda/libwam_c_api.so \
  --descriptor build-gate-b-serving-checkpoint-cuda/wam.desc \
  --model build-fastwam-gate-b/fastwam-robotwin-policy-v3.gguf \
  --tokenizer /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl \
  --text-encoder /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors \
  --language-python-root /testessfs10/users/yejun.zeng/codes/FastWAM/src \
  --host 0.0.0.0 --port 18162 --device 0 --random-seed 0
```

## Client

Run in `robotwin_zyj`. The donor evaluation contract uses 24 actions per
replan. Select a client GPU with enough memory for RoboTwin Curobo warmup. Set
`CUDA_VISIBLE_DEVICES` before Python starts; the script-level option is a
logical device index after that mask.

```bash
CUDA_VISIBLE_DEVICES=6 python -u eval/sim/run_robotwin_client.py \
  --robotwin-root /testessfs10/users/yejun.zeng/codes/RoboTwin \
  --descriptor build-gate-b-serving-checkpoint-cuda/wam.desc \
  --task beat_block_hammer --episodes 1 --seed 0 \
  --task-config demo_randomized --instruction-type unseen \
  --policy-name fastwam \
  --host SERVER_HOST --port 18162 \
  --execute-steps 24 --fixed-action-noise-seed 0 --client-gpu 0
```

The fixed action-noise seed reproduces the donor behavior: every replan creates
a fresh CPU PyTorch generator with seed 0. Session-generated advancing noise is
valid for stochastic evaluation, but is not comparable with this donor
baseline.

The earlier `beat_block_hammer` integration smoke completed all 400 simulator
steps and 17 action-chunk requests without RPC, preprocessing, normalization or
CUDA errors. It did not solve the task (`0/1`). Server-total latency had mean
380.38 ms and RPC round-trip latency had mean 386.61 ms. This validates the
vertical path only; it is not a success-rate benchmark. That smoke used
`demo_clean`, so it is not directly comparable with the donor FastWAM baseline,
which uses `demo_randomized`.

The v3 artifact adds the state normalization clamp required by the donor
`SingleFieldLinearNormalizer.forward` implementation. Do not use the older v2
artifact for a formal baseline comparison.

## Donor comparison

After starting the donor server from FastWAM's `ROBOTWIN_REMOTE_EVAL.md`, use
the same client/evaluator through its donor adapter:

```bash
CUDA_VISIBLE_DEVICES=6 python -u eval/sim/run_robotwin_client.py \
  --robotwin-root /testessfs10/users/yejun.zeng/codes/RoboTwin \
  --donor-fastwam-root /testessfs10/users/yejun.zeng/codes/FastWAM \
  --task beat_block_hammer --episodes 100 --seed 0 \
  --task-config demo_randomized --instruction-type unseen \
  --policy-name fastwam_baseline \
  --host SERVER_HOST --port 18163 \
  --execute-steps 24 --client-gpu 0 \
  --metrics-output eval/results/fastwam_robotwin_baseline/donor_metrics.json
```

Keep the generated manifest unchanged, then run the wam.cpp client with the
same task, split, policy name, episode count and seed. The client records the
manifest SHA-256 in both reports. The audited 100-episode result and latency
interpretation are in `eval/sim/FASTWAM_ROBOTWIN_BASELINE_AUDIT.md`.
