# Support Matrix

This document separates runtime implementation, numerical validation, simulator
integration, and model quality. A successful synthetic test is not a real
checkpoint or simulator result.

## Runtime

| Surface | Status | Scope |
| --- | --- | --- |
| C++17 API | Supported | Model, Session, Pipeline, structured errors |
| C ABI | Supported | ABI v4, explicit ownership and init/free pairs |
| Python local | Supported | Installable `wam` package over C ABI v4 |
| Python remote | Supported | `wam.rpc.v06`, binary Protobuf/WebSocket |
| CPU metadata | Supported | Inspect/validate without weight allocation |
| CUDA prediction | Supported | Explicit device and compute precision |
| CPU prediction | Limited | Only architecture/profile paths that declare it |
| Dynamic plugins | Not supported | Built-in compile-time registration only |
| gRPC / ROS2 | Not supported | No stable 0.6 contract |
| Multi-GPU/offload | Not supported | One explicit device per model |

## Model and environment evidence

| Model / environment | Artifact and parity | Simulator evidence | Release interpretation |
| --- | --- | --- | --- |
| GWP05 / RoboTwin | Schema-v3 GGUF and independent A800 BF16 public-Session parity passed | v0.6 `beat_block_hammer` smoke 1/1; frozen 100 episodes 87/100 | Validated vertical path for the audited checkpoint |
| FastWAM / LIBERO | Schema-v3 artifact and same-input action parity passed | v0.6 smoke 1/1; four suites, 40 tasks, 2000 episodes: donor 96.75%, wam.cpp 96.90% | Validated vertical path for the released LIBERO checkpoint |
| FastWAM / RoboTwin | Artifact and serving path exercised | Smoke completed without runtime failure, task result 0/1 | Integration only; not model-quality support |
| FastWAM / LIBERO-X | Artifact and same-input action gate passed for audited fixture | Fixed manifest agrees at result level; audited checkpoint is 0/10 | Integration fixture only; checkpoint training audit blocks release |
| GWP05 / LIBERO or LIBERO-X | No release GGUF/parity evidence in Git | Not validated | Unsupported until real checkpoint gates pass |

Detailed frozen evidence and caveats are in:

- `eval/sim/ROBOTWIN_REMOTE_EVAL.md`
- `eval/sim/FASTWAM_LIBERO_BASELINE_AUDIT.md`
- `eval/sim/FASTWAM_ROBOTWIN_BASELINE_AUDIT.md`
- `eval/sim/FASTWAM_LIBEROX_REMOTE_EVAL.md`

## Verified build matrix

As of 2026-07-29, the default CPU and CUDA 12.4 + cuDNN `sm_80`
configurations each pass 44/44 tests. The Runtime-only build with GWP05,
FastWAM, and Serving disabled passes 30/30. The
matrix covers public headers, pure-C/C++ install consumers, artifact/Policy
contracts, lifecycle, C ABI, Python wheel installation, RPC, adapters, and eval
fixtures, including the release documentation contract.

The CPU workflow is configured to run the default Release build and CTest suite
on every push and pull request. CUDA and real-asset gates remain opt-in because
hosted CI does not own the required GPU, GGUF, simulator, or frozen replay
assets.

External numerical gates are configured with the grouped CMake variables listed
in the root `CMakeLists.txt`, including GWP05 real/RoboTwin GGUF and reference
paths plus FastWAM LIBERO/RoboTwin GGUF and replay paths. Partial groups fail at
configure time; absent groups are explicitly skipped.

## Known limitations

- The SDK does not download checkpoints, tokenizers, or language encoders.
- WebSocket serving does not provide authentication, TLS, quotas, or process
  supervision.
- GWP05 and FastWAM prediction is primarily validated on CUDA/A800 BF16/F32
  paths; portable CPU builds mainly enforce metadata and contract behavior.
- Real checkpoint parity and simulator smoke are external gates and cannot run
  from a clean Git checkout alone.
- RoboTwin and LIBERO-X upstream evaluators do not expose an audited policy
  factory in the revisions available during migration, so their runner startup
  boundary still injects the remote policy without modifying upstream files.
- Schema-v2 and legacy GWP05 assets are migration oracles, not a 0.6 artifact
  compatibility promise. New artifacts must be schema v3.
- The FastWAM LIBERO-X step-50k checkpoint has an invalid resumed scheduler
  trajectory and insufficient task diversity; its numbers must not be presented
  as model quality.
