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
  --out build-fastwam-gate-b/fastwam-robotwin-policy-v2.gguf
```

The frozen conversion contains 1741 tensors and is 12,150,404,864 bytes. Run
`scripts/inspect/inspect_fastwam.py` and the `wam_fastwam_robotwin_artifact`
CTest before serving it.

## Server

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=$PWD/eval \
python -u eval/sim/run_robotwin_server.py \
  --library build-gate-b-serving-checkpoint-cuda/libwam_c_api.so \
  --descriptor build-gate-b-serving-checkpoint-cuda/wam.desc \
  --model build-fastwam-gate-b/fastwam-robotwin-policy-v2.gguf \
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
  --host SERVER_HOST --port 18162 \
  --execute-steps 24 --client-gpu 0
```

The initial `beat_block_hammer` integration smoke completed all 400 simulator
steps and 17 action-chunk requests without RPC, preprocessing, normalization or
CUDA errors. It did not solve the task (`0/1`). Server-total latency had mean
380.38 ms and RPC round-trip latency had mean 386.61 ms. This validates the
vertical path only; it is not a success-rate benchmark.
