# FastWAM LIBERO-X Remote Evaluation

## Current status

The LIBERO-X environment integration is complete: the client/server adapter,
z-score PolicySpec, GGUF artifact checks, same-input donor/wam.cpp action
parity, and fixed-manifest rollout path have all been exercised. Model
readiness is still pending.

The existing `step_050000.pt` and its converted GGUF are integration fixtures,
not release or benchmark checkpoints. A training audit found that the run
restored a cosine scheduler with the old `T_max=28500`; its learning rate
reached the minimum around step 30k and then increased again. The frozen task 0
was present in training, but its exact SCENE1 layout had only two independent
demonstrations. The run also had no held-out validation and no training-time
simulator evaluation. Consequently, the recorded `0/10` establishes
donor/wam.cpp result agreement only and must not be used to judge wam.cpp model
quality.

The commands below preserve the completed integration evidence. Do not publish
the step 50k GGUF as a supported LIBERO-X model or extend it into a formal
benchmark. After retraining with a corrected scheduler, convert the new
checkpoint to GGUF and repeat the artifact, same-input action, fixed-manifest,
success-rate, and latency gates.

Two distinct evaluation profiles exist:

- the initial integration smoke reused the released LIBERO checkpoint as a
  zero-shot LIBERO-X policy;
- the checkpoint below was trained on the LIBERO-X dataset and must use the
  `fastwam_liberox_2cam224_zscore` PolicySpec.

Do not use the LIBERO min-max profile for the specialized checkpoint. Its saved
training config sets `norm_default_mode: z-score` and points to the LIBERO-X
dataset. Environment-specific image keys and gripper postprocessing remain in
the LIBERO-X client; the C++ engine and GGUF contain no environment branch.

## Audited integration artifact

```bash
PYTHONPATH=$PWD/build-gate-b-serving-checkpoint-cuda/_deps/llama-src/gguf-py \
python scripts/convert/convert_fastwam.py \
  --checkpoint /testessfs10/users/yejun.zeng/codes/FastWAM/runs/liberox_train_2cam224_1e-4/2026-07-09_06-53-13/checkpoints/weights/step_050000.pt \
  --vae /testessfs10/users/yejun.zeng/codes/FastWAM/checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors \
  --norm-stats /testessfs10/users/yejun.zeng/codes/FastWAM/runs/liberox_train_2cam224_1e-4/2026-07-09_06-53-13/dataset_stats.json \
  --policy-profile profiles/fastwam_liberox_2cam224_zscore.json \
  --out build-fastwam-gate-b/fastwam-liberox-step050000-zscore-policy-v2.gguf \
  --num-inference-steps 10
```

The inspector reports 1741 tensors, two 224x224 cameras, an 8D state,
`[32,7]` actions, and profile `fastwam_liberox_2cam224_zscore`. The earlier
`fastwam-liberox-step050000-policy-v1.gguf` used min-max statistics and is
invalid for this checkpoint.

## Server

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/python \
python -u eval/sim/run_liberox_server.py \
  --library build-gate-b-serving-checkpoint-cuda/libwam_c_api.so \
  --descriptor build-gate-b-serving-checkpoint-cuda/wam.desc \
  --model build-fastwam-gate-b/fastwam-liberox-step050000-zscore-policy-v2.gguf \
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

## Fixed-manifest parity gate

The first specialized-checkpoint gate freezes LEVEL1 task 0 and source init
state 0. Its manifest is generated under the ignored evaluation output tree:

```text
build-fastwam-gate-b/liberox-parity-manifest-v1/manifest.json
```

Manifest SHA256: `4790e73484f15ca817d2777f99fad9a7903fb40b579c6f99db1618cc144bb409`.
Both backends use environment seed 0, action-noise seed 7, BF16-rounded donor
noise, 10 denoise steps, replan horizon 10, image flipping, gripper inversion
and binarization, and the same staged BDDL/init fixture.

The donor must be launched with `--task liberox_train_2cam224_1e-4`. The old
`liberox_uncond_2cam224_1e-4` task is the LIBERO zero-shot config and does not
select the specialized LIBERO-X processor contract.

| Backend | Result | Requests | Rollout | Mean model | P50 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| donor Python | 0/1 | 120 | 130.83 s | 643.99 ms | 636.00 ms | 673.00 ms |
| wam.cpp | 0/1 | 120 | 95.14 s | 338.37 ms | 333.15 ms | 368.51 ms |

This passes serving and result-level parity for one frozen episode and shows a
1.90x mean model-latency speedup. It is not a success-rate benchmark and does
not replace an independent same-input action MAE gate. The EGL destructor
warning printed at interpreter shutdown is upstream cleanup behavior; both
clients completed and wrote their result and summary files.

## Same-input action gate

The independent gate captures one upstream observation after the exact
LIBERO-X flip and padded resize, then executes donor PyTorch and wam.cpp with
the same BF16 UMT5 embedding, mask, normalized state and explicit BF16-rounded
noise. The specialized checkpoint passes its recorded cross-implementation
tolerance:

```text
real action MAE:  0.000623663
real action max:  0.00775671
tolerance:        mean <= 0.0009, max <= 0.01
gripper signs:    32/32 equal
```

The tighter released-LIBERO regression remains unchanged at mean `0.0009` and
max `0.005`; after the LIBERO-X pixel/VAE BF16 operation-boundary fixes it
passes with MAE `0.000716154` and max `0.00390625`.

## Formal fixed-manifest runner

Create a manifest inside the LIBERO-X environment. Task indices follow the
upstream evaluator's natural filename order. A task's episodes must be a
contiguous prefix of its saved init tensor so the upstream `episode_index` and
`env.seed(episode_index)` semantics remain unchanged.

```bash
python eval/sim/run_liberox_client.py \
  --liberox-root /testessfs10/users/yejun.zeng/codes/LIBERO-X \
  --create-manifest /tmp/fastwam-liberox-level1-task0-10-v1.json \
  --scene-group LEVEL1 --task-indices 0 --episodes-per-task 10 \
  --num-steps-wait 10 --max-steps 1200 --resize-size 224 \
  --replan-steps 10 --seed 7
```

The manifest freezes source BDDL/init SHA256 values and all execution
parameters. The verified manifest is checked in as
`eval/manifests/fastwam_liberox_level1_task0_10_v1.json`. For donor evaluation,
materialize an isolated upstream-compatible tree and pass the printed roots to
`eval_template.py`:

```bash
python eval/sim/run_liberox_client.py \
  --liberox-root /testessfs10/users/yejun.zeng/codes/LIBERO-X \
  --manifest eval/manifests/fastwam_liberox_level1_task0_10_v1.json \
  --stage-manifest-only build-fastwam-gate-b/liberox-level1-task0-10-v1/stage
```

For wam.cpp, the client validates and stages the same source files, delegates
the rollout to the unchanged upstream evaluator, validates the exact result
set, and writes `manifest.json`, `selection.json`, `results.jsonl`, and
`wam-summary.json`:

```bash
MUJOCO_GL=egl python eval/sim/run_liberox_client.py \
  --liberox-root /testessfs10/users/yejun.zeng/codes/LIBERO-X \
  --descriptor build-gate-b-serving-checkpoint-cuda/wam.desc \
  --manifest eval/manifests/fastwam_liberox_level1_task0_10_v1.json \
  --output-dir build-fastwam-gate-b/liberox-level1-task0-10-v1/wam \
  --host SERVER_HOST --port 18171
```

The first formal manifest has SHA256
`458b441e34d2566333430db97848b537e4967a32cea7f2889486a80f7897e5c1`.
Its upstream task 0 is `SCENE1/open the middle drawer of the wooden cabinet`,
not the manually selected `SCENE10` task used by the earlier smoke gate.

| Backend | Success | Agreement | Requests | Rollout |
| --- | ---: | ---: | ---: | ---: |
| donor Python | 0/10 | reference | 1200 | 1223.51 s |
| wam.cpp | 0/10 | 10/10 | 1200 | 883.79 s |

wam.cpp rollout was 1.38x faster. Its RPC round-trip mean/p50/p95 were
`340.82/337.28/361.87 ms`; server model mean/p50/p95 were
`326.57/322.96/346.78 ms`. The donor's upstream wire protocol does not return
per-request phase timings, so only same-run rollout duration is compared here.
Both backends failed every state without initialization, RPC, or runtime
errors. This establishes result-level parity for the frozen task, but a 10-run
`0%` result is insufficient to assess the specialized checkpoint's overall
LIBERO-X quality.
