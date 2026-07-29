# GWP05 RoboTwin Gate A

This gate is local-only. The GGUF, checkpoint, observation, prompt embedding,
noise, and output tensors are not committed. The committed C++ test owns the
comparison logic and the JSON contract freezes its tolerances before the first
run.

The formal artifact is a 14D z-score RoboTwin profile with three named views,
a 384x320 canvas, a 48x14 explicit action-noise tensor, and masked
add-current-state recovery. It uses `mot-vae-bf16-qkv-v1`; all GWP, VAE, and T5
weights needed by the selected runtime are stored as BF16. The complete
PolicySpec is authoritative. No legacy 32D quantile fallback is allowed.

## Current Status

As of 2026-07-25:

- conversion and the independent Python inspector pass;
- the C++ GGUF reader, PolicySpec validator, GWP artifact loader, and
  `Backend::cpu_metadata` load pass;
- the independent PyTorch oracle was captured on an A800 with PyTorch
  2.7.1+cu126, BF16, deterministic algorithms, and TF32 disabled;
- the CUDA+cuDNN public Session parity passes on SM80, including prefix cache,
  explicit action noise, timing phases, and bit-exact reset/repeat;
- schema-v2 image preprocessing uses PIL-compatible truncate/renormalize at
  antialias boundaries; the isolated legacy migration profile retains its
  historical clamp behavior and is checked by the legacy F32 oracle.

Gate A is complete. The commands below reproduce the accepted result.

## 0. Reproduce The Formal Artifact

The converter never overwrites an existing GGUF and deliberately does not hash
the complete 23 GB output. The manifest records the exact profile, converter
sources, component counts/dtypes, and output size.

Run the following commands inside `gwp_zyj`.

```bash
python scripts/convert/convert_gwp05.py \
  --checkpoint /testessfs10/users/yejun.zeng/share/gwp/gwp05_robotwin_all_32gpus_frompre/models/checkpoint_epoch_9_step_100000/transformer_ema \
  --base-model /testessfs10/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers \
  --norm-stats /testessfs10/users/yejun.zeng/share/gwp/gwp05_robotwin_all_32gpus_frompre/models/checkpoint_epoch_9_step_100000/transformer_ema/norm_stats_delta.json \
  --policy-spec profiles/gwp05_robotwin_14d_zscore.json \
  --weight-policy mot-vae-bf16-qkv \
  --out /testessfs10/users/yejun.zeng/share/gwp/models/robotwin/gwp05-robotwin-14d-zscore-mot-vae-bf16-qkv.gguf

build/wam-inspect \
  /testessfs10/users/yejun.zeng/share/gwp/models/robotwin/gwp05-robotwin-14d-zscore-mot-vae-bf16-qkv.gguf
build/wam-validate \
  /testessfs10/users/yejun.zeng/share/gwp/models/robotwin/gwp05-robotwin-14d-zscore-mot-vae-bf16-qkv.gguf
```

## 1. Capture The Independent PyTorch Oracle

The fixture comes from frame 20 of one real `beat_block_hammer` episode. It
uses an explicit NumPy PCG64 noise tensor, so PyTorch and C++ do not depend on
matching RNG implementations. The prompt uses the training-time limit of 64
tokens; the inference server's generic 512-token default is not the checkpoint
contract.

```bash
cd /testessfs10/users/yejun.zeng/codes/gwp/github/wam.cpp-0.6

nvidia-smi -L
python tests/reference/capture_gwp05_robotwin_pytorch.py \
  --inference-script /testessfs10/users/yejun.zeng/codes/gwp/giga-models/projects/wam/giga-world-policy-0-5/scripts/inference_server_robotwin.py \
  --checkpoint /testessfs10/users/yejun.zeng/share/gwp/gwp05_robotwin_all_32gpus_frompre/models/checkpoint_epoch_9_step_100000/transformer_ema \
  --base-model /testessfs10/models/huggingface/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers \
  --norm-stats /testessfs10/users/yejun.zeng/share/gwp/gwp05_robotwin_all_32gpus_frompre/models/checkpoint_epoch_9_step_100000/transformer_ema/norm_stats_delta.json \
  --input-dir /testessfs10/users/yejun.zeng/share/gwp/gate-a/gwp05_robotwin_14d/input \
  --output-dir /testessfs10/users/yejun.zeng/share/gwp/gate-a/gwp05_robotwin_14d/reference \
  --device cuda:0
```

The capture must produce `reference-manifest.json` plus processed
image, normalized state, prompt embedding/mask, VAE latent, normalized action,
and recovered action arrays. The script refuses to overwrite a nonempty output
directory.

## 2. Run Public Model/Session Parity

Use the compute capability of the selected GPU (`80` for A800, `89` for
RTX 4090):

```bash
cmake -S . -B build-gate-a-cuda \
  -DWAM_BUILD_TESTS=ON \
  -DWAM_CUDA=ON \
  -DWAM_CUDNN=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DWAM_TEST_GWP05_ROBOTWIN_GGUF=/testessfs10/users/yejun.zeng/share/gwp/models/robotwin/gwp05-robotwin-14d-zscore-mot-vae-bf16-qkv.gguf \
  -DWAM_TEST_GWP05_ROBOTWIN_INPUT_DIR=/testessfs10/users/yejun.zeng/share/gwp/gate-a/gwp05_robotwin_14d/input \
  -DWAM_TEST_GWP05_ROBOTWIN_REFERENCE_DIR=/testessfs10/users/yejun.zeng/share/gwp/gate-a/gwp05_robotwin_14d/reference

cmake --build build-gate-a-cuda --target \
  wam_gwp05_robotwin_artifact wam_gwp05_robotwin_parity --parallel
ctest --test-dir build-gate-a-cuda --output-on-failure \
  -R 'wam_gwp05_robotwin_(artifact|parity)'
```

The parity test independently re-runs PolicySpec image/state preprocessing,
loads the model through the public API with explicit CUDA/BF16 options, enables
the runtime prefix cache, predicts from the frozen noise, inverts final action
recovery to inspect normalized action, compares the recovered RoboTwin action,
and requires reset/repeat determinism. A failure must be localized at the
earliest boundary; final-action tolerance must not be widened to hide a
preprocessing or model-space mismatch.

The hard recovered-action non-regression envelope is MAE `<= 0.006` and maximum
error `<= 0.04`. It was frozen before this run from the known A800 C++/PyTorch
BF16 baseline (`0.005315709 / 0.034212112` at the worst historical fixture),
not fitted to this checkpoint. The executable also reports the stricter
PyTorch target of `0.001 / 0.01` separately. Passing the non-regression Gate does
not erase a reported strict-target failure; such a result belongs to BF16
kernel-alignment work, not PolicySpec or RoboTwin action recovery.

## Accepted Result

| Boundary | Mean absolute error | Maximum absolute error |
| --- | ---: | ---: |
| processed image | `0.00114409` | `0.00764728` |
| normalized state | `0` | `0` |
| normalized action | `0.00140892` | `0.00781274` |
| recovered action | `0.000479634` | `0.00390983` |

The recovered action also passes the stricter `0.001 / 0.01` PyTorch target.
The PyTorch reference manifest SHA256 is
`c287f36ddecfee8dd886a70d663bb1e9cf593701ef8800a2a1141a6f1314ff08`.
