# wam.cpp 0.5 migration baseline

This document identifies the immutable oracle for the 0.6 structural migration.
Large models, fixtures, simulator trees, and rollout outputs remain outside Git.

## Source baseline

- Commit: `d20c7ec8e25a8b2c52c5066978cd9a51ec1723e4`
- Annotated tag: `v0.5-migration-baseline`
- Source branch at freeze: `refactor/0.5`
- 0.6 worktree branch: `refactor/0.6`
- Freeze date: 2026-07-28

The tagged source was clean and matched `private-origin/refactor/0.5` when the
worktree was created. The 0.6 branch must compare numerical behavior against
this tag, not against a moving development checkout.

## Verified build and test baseline

The tagged revision records a Release CPU build with all 20 configured tests
passing and a CUDA 12.4/A800 `sm_80` build with all 24 configured tests passing.
The CUDA run includes RPC, language-provider, GWP05 RoboTwin BF16 parity, and
FastWAM LIBERO artifact/action parity gates. See the tagged `README.md` for the
summary and the gate documents below for reproduction commands.

Phase 1 changes build ownership only. It must preserve the public C ABI v3 and
must not alter GWP05 or FastWAM tensor math, preprocessing, scheduling, cache,
or action decoding.

## GWP05 oracle

- Checkpoint revision: `checkpoint_epoch_9_step_100000/transformer_ema`
- Checkpoint root: `/testessfs10/users/yejun.zeng/share/gwp/gwp05_robotwin_all_32gpus_frompre/models/checkpoint_epoch_9_step_100000/transformer_ema`
- GGUF: `/testessfs10/users/yejun.zeng/share/gwp/models/robotwin/gwp05-robotwin-14d-zscore-mot-vae-bf16-qkv.gguf`
- Fixed input: `/testessfs10/users/yejun.zeng/share/gwp/gate-a/gwp05_robotwin_14d/input`
- PyTorch reference: `/testessfs10/users/yejun.zeng/share/gwp/gate-a/gwp05_robotwin_14d/reference`
- Simulator root: `/testessfs10/users/yejun.zeng/codes/RoboTwin`
- Formal result: `87/100` on the frozen `beat_block_hammer` manifest, with 378
  action-chunk requests and no RPC/runtime failure.

The exact conversion and parity commands are in `GWP05_ROBOTWIN_GATE_A.md`.
The rollout manifest, metrics, request records, and result paths are in
`../../eval/sim/ROBOTWIN_REMOTE_EVAL.md`.

## FastWAM oracle

- Release checkpoint: `libero_uncond_2cam224.pt`
- GGUF relative to the tagged checkout:
  `build-fastwam-gate-b/fastwam-libero-policy-v2.gguf`
- Upstream FastWAM root: `/testessfs10/users/yejun.zeng/codes/FastWAM`
- LIBERO root: `/testessfs10/users/yejun.zeng/codes/vla.cpp/third_party/LIBERO`
- LIBERO revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Simulator versions: MuJoCo 3.3.2 and robosuite 1.4.0
- Formal result: donor `1935/2000` and wam.cpp `1938/2000` over the same four
  suites and ordered 50 init states per task.
- Fixed-fixture final action error: MAE `0.00085519999`, maximum absolute error
  `0.003662109375`.

The artifact and numerical contract is in `FASTWAM_LIBERO_GATE_B.md`. Deployment
paths and commands are in `../../eval/sim/FASTWAM_LIBERO_REMOTE_EVAL.md`; the
full rollout audit is in `../../eval/sim/FASTWAM_LIBERO_BASELINE_AUDIT.md`.

## Required regression policy

Structural commits may run the default CPU suite on every change. Any commit
that changes model sources, GGML dependency/patches, precision, graph building,
preprocessing, cache behavior, or action recovery must additionally rerun the
corresponding external artifact and numerical parity gates above. Simulator
success-rate reruns are required when end-to-end policy behavior changes; they
are not replaced by a successful compile or metadata-only test.
