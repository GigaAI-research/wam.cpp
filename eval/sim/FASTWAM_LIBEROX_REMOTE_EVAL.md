# FastWAM LIBERO-X Remote Evaluation

LIBERO-X zero-shot evaluation reuses the LIBERO FastWAM checkpoint and
PolicySpec. Environment-specific image keys and gripper postprocessing remain
in the LIBERO-X client; the C++ engine and GGUF contain no LIBERO-X branch.

## Server

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/eval \
python -u eval/sim/run_liberox_server.py \
  --library build-gate-b-serving-checkpoint-cuda/libwam_c_api.so \
  --descriptor build-gate-b-serving-checkpoint-cuda/wam.desc \
  --model build-fastwam-gate-b/fastwam-libero-policy-v2.gguf \
  --tokenizer /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl \
  --text-encoder /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors \
  --language-python-root /testessfs10/users/yejun.zeng/codes/FastWAM/src \
  --host 0.0.0.0 --port 18163 --device 0 --random-seed 7
```

## Client

Run in the `liberox` container:

```bash
MUJOCO_GL=egl python -u eval/sim/run_liberox_client.py \
  --liberox-root /testessfs10/users/yejun.zeng/codes/LIBERO-X \
  --descriptor build-gate-b-serving-checkpoint-cuda/wam.desc \
  --host SERVER_HOST --port 18163 \
  --scene-group LEVEL1 --load-mode init \
  --output-dir build-fastwam-gate-b/liberox-level1 \
  --num-trials-per-task 1 --num-steps-wait 10 \
  --max-steps 1200 --resize-size 224 --replan-steps 10
```

The adapter injects the wam RPC policy before importing the upstream evaluator.
This also avoids importing the Python-3.10-only openpi WebSocket client in the
current Python 3.9 LIBERO-X container. Upstream LIBERO-X files are unchanged.

The initial one-task LEVEL1 integration smoke completed all 200 configured
steps without RPC or engine errors. It did not solve the task (`0/1`); this is
an integration result, not a success-rate benchmark. The EGL destructor warning
printed at interpreter shutdown is upstream cleanup behavior and the process
still exits successfully.
