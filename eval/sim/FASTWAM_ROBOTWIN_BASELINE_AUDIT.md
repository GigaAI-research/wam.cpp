# FastWAM RoboTwin Baseline Audit

Audit date: 2026-07-27

## Frozen donor baseline

The launch contract in the donor `ROBOTWIN_REMOTE_EVAL.md` selects:

- checkpoint: `fastwam_release/robotwin_uncond_3cam_384.pt`
- dataset statistics: `fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json`
- Hydra task: `robotwin_uncond_3cam_384_1e-4`
- simulator task/config: `beat_block_hammer` / `demo_randomized`
- instruction split: `unseen`
- 100 episodes with seed 0

The checkpoint is 12,041,813,092 bytes. Its top-level payload records step
29355 and BF16 dtype. The proprio projection consumes 14 values, the action
projection consumes 14 values, and the release config defines 33 frames: one
observation plus a 32-step action horizon. The model file does not embed a
dataset-statistics identity or digest, so the release document and co-located
file names remain the provenance link between the two files.

## Normalization audit

The selected statistics file is 88,715 bytes and has SHA-256:

```text
7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095
```

The donor RoboTwin config sets `norm_default_mode: z-score` and
`use_stepwise_action_norm: false`. Therefore both state and action use the
14-value `global_mean` and `global_std` arrays. The current converter follows
that selection; all four GGUF statistics tensors are bit-exact with the
corresponding float32 arrays in the JSON file.

The donor normalizer clamps every forward-normalized value to `[-5, 5]` and
does not clamp the backward action denormalization. Consequently the PolicySpec
must encode `normalization.state.output_clamp = [-5, 5]` and no action output
clamp. The older `fastwam-robotwin-policy-v2.gguf` omitted the state clamp and
must not be used for formal evaluation. The audited v3 artifact was regenerated
from the corrected profile; it contains 1741 tensors and is 12,150,404,960
bytes.

## Observation and action contract

The current profile matches the donor deployment path:

- camera order is high, left wrist, right wrist;
- high view is resized to 320x256 and occupies the top of the canvas;
- wrist views are resized to 160x128 and occupy the lower left/right;
- the resulting RGB canvas is 320x384 and maps uint8 pixels to `[-1, 1]`;
- state comes from RoboTwin `observation["joint_action"]["vector"]`;
- output is a 32x14 absolute joint-position chunk with identity recovery;
- 24 actions are executed before replanning;
- diffusion uses 10 inference steps.

## Evaluation compatibility

The original wam.cpp smoke used `demo_clean`, while the donor FastWAM command
uses `demo_randomized`. The client now exposes the simulator task config,
instruction split and policy name as explicit CLI arguments. A donor-comparable
run must pass:

```text
--task-config demo_randomized --instruction-type unseen --policy-name fastwam
```

## Frozen-manifest result

Both backends were evaluated against this exact manifest:

```text
/testessfs10/users/yejun.zeng/codes/RoboTwin/logs/single_task_eval/eval_result/beat_block_hammer/fastwam_baseline/demo_randomized/none/manifest/eval_manifest_seed0_test100_unseen.json
SHA-256: 4430ca7626ca96af38e6b4428f70c67edcf93e274cd1cd58b3e33be00d8de137
```

The manifest contains 100 `demo_randomized` / `unseen` episodes. Both runs use
24 executed actions per replan. The donor recreates a CPU PyTorch generator
with seed 0 for every inference request; the wam.cpp client therefore supplies
the same fixed 32x14 action-noise tensor on every request. The donor and wam.cpp
servers ran sequentially on physical GPU 5. RoboTwin ran in an isolated
container on physical GPU 6, because exposing every host GPU allowed SAPIEN to
select a nearly full device and fail while allocating Vulkan buffers.

| Backend | Success | Requests | Mean chunk RPC | P50 | P95 | Steady mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| donor Python | 98/100 | 546 | 636.29 ms | 633.76 ms | 665.29 ms | 636.69 ms |
| wam.cpp v3, old text semantics | 1/100 | 1693 | 360.10 ms | 353.56 ms | 398.66 ms | 356.39 ms |
| wam.cpp v3, donor text semantics | 98/100 | 538 | 362.78 ms | 350.65 ms | 417.64 ms | 352.91 ms |

After the text fix, the wam.cpp request is 1.75x faster by the all-request mean
and 1.80x faster in steady state. Its server-side all-request mean is 358.05 ms
total, including 342.89 ms model and 15.12 ms preprocess. Peak device memory
reported by wam.cpp is 12,150,282,496 bytes.

The donor RPC returns one action at a time. The adapter measures one common
24-action chunk as the first model request plus 23 queue-drain requests, so its
636.29 ms chunk measurement includes more transport calls than wam.cpp. Its
first-action request, which contains the actual donor inference, averages
619.10 ms (p50 616.83 ms, p95 647.56 ms). This does not change the latency
ranking.

Before the text fix, 99 wam.cpp episodes consumed the full 17 replans despite
faster individual requests. After the fix, wam.cpp matches the donor's 98%
success rate and uses slightly fewer model requests. Donor failures are seeds
100079 and 100138; corrected wam.cpp failures are seeds 100138 and 100147.

Raw reports are intentionally ignored by Git and remain at:

```text
eval/results/fastwam_robotwin_baseline/donor_metrics.json
eval/results/fastwam_robotwin_baseline/donor_metrics.requests.jsonl
eval/results/fastwam_robotwin_baseline/wam_cpp_v3_metrics.json
eval/results/fastwam_robotwin_baseline/wam_cpp_v3_metrics.requests.jsonl
eval/results/fastwam_robotwin_baseline/wam_cpp_v3_text_fix_metrics.json
eval/results/fastwam_robotwin_baseline/wam_cpp_v3_text_fix_metrics.requests.jsonl
```

## Gate verdict

Checkpoint, statistics, PolicySpec, serving topology and language padding are
controlled. Gate B quality parity passes: both donor Python and corrected
wam.cpp achieve `98/100` on the same frozen manifest.

## Root-cause diagnosis

The primary regression was the server-owned external embedding provider. Donor
`FastWAM.encode_prompt()` encodes with the tokenizer mask, zeros every padded
embedding row, and then replaces the attention mask with all ones. The current
`WanUmt5EmbeddingProvider` returns the unmodified encoder output and original
0/1 tokenizer mask. In the frozen diagnostic prompt only 31 of 128 tokenizer
positions are valid, so the two model paths expose fundamentally different
cross-attention contexts.

With the same three images, state and seed-0 action noise, the current formal
donor action versus wam.cpp action has MAE `0.19322`, maximum absolute error
`0.98131`, and RMSE `0.28906`. Applying donor padding semantics to the wam.cpp
input reduces those errors to MAE `0.01512`, maximum `0.08763`, and RMSE
`0.02680`. The text semantics alone account for MAE `0.19390` in the Python
reference and `0.19531` in the C++ path.

The causal rollout uses the original frozen 100-episode manifest through a
one-episode diagnostic alias. Its record is seed `100000` with instruction
`Grab the hammer with claw and smooth head and strike the block`. Without
changing C++ image preprocessing, a temporary provider applying only donor
text padding semantics solves the task at control step 103 in five action-chunk
requests. This record failed in the original wam.cpp run. A second diagnostic
that also applies donor PIL resizing succeeds at step 104, so exact PIL resize
is not required for this record's recovery.

Image preprocessing remains a secondary parity issue. PIL resize quantizes the
resized image back to uint8, while the C++ reference resize keeps interpolated
float values. On the fixed numerical fixture the composite image difference is
MAE `0.00173` in `[-1,1]`; under donor text semantics it changes denormalized
action by MAE `0.01834` and maximum `0.10954`. With identical composite image
and text, residual C++ engine error is MAE `0.00481` and maximum `0.02738`.

`WanUmt5EmbeddingProvider` now reproduces donor zero-padding/all-ones-mask
semantics and has a boundary regression test with deliberately nonzero padded
encoder rows. The frozen rerun above confirms that this fixes policy quality.

The subsequent numerical tightening did not relax a parity threshold. The C++
uint8 two-pass bilinear path now reproduces Pillow exactly on all 368,640 RGB
values in the fixed three-camera RoboTwin fixture; its previous normalized
image MAE was `0.00173156` with maximum `0.007774`. With exact PIL images and
donor text, the residual action error was MAE `0.00413806` and maximum
`0.02464616` against the independent donor output.

FastWAM timesteps and deltas are now quantized to BF16 like the donor scheduler,
and the action update rounds the multiply and add at the same BF16 boundaries.
The RoboTwin fixture action error decreases to MAE `0.00330836` and maximum
`0.02190769`. On the independent LIBERO fixture it decreases from MAE
`0.00090151`, maximum `0.00549316` to MAE `0.00085520`, maximum `0.00366211`.
