# wam.cpp 0.6 Architecture

This document describes the implemented 0.6 structure and its dependency
rules. Migration history and phase evidence live in `TODO.md` and Git history;
they are not part of the runtime contract.

## Design goals

- Keep the public API small and independent of GGML, GGUF, model, and simulator
  implementation types.
- Make one prediction readable as a linear pipeline.
- Keep checkpoint policy semantics separate from model mathematics and
  environment/controller semantics.
- Share code only after GWP05 and FastWAM use it with the same meaning.
- Reject unsupported inputs and artifacts with stable, field-level errors.
- Make a new model or environment a local extension rather than a framework
  rewrite.

## Dependency direction

```text
applications / serving / adapters / evaluation
                       |
                 public wam API
                       |
            artifact + policy + runtime
                       |
              model implementations
                       |
                 GGML backend
```

Dependencies only point downward. In particular:

- public headers do not include internal or GGML headers;
- Artifact code does not depend on Policy or a concrete model;
- Policy code does not depend on a concrete model;
- model network code does not parse artifacts or know serving/environments;
- GWP05 and FastWAM do not include each other;
- core/model code contains no RoboTwin, LIBERO, or LIBERO-X behavior;
- adapters and runners never select preprocessing from architecture ids.

`tests/contracts/source_boundaries.cmake` and
`tests/python/test_phase8_boundaries.py` enforce these rules.

## Public C++ API

Stable headers live in `include/wam/`:

| Header | Responsibility |
| --- | --- |
| `error.h` | Stable ErrorCode, message, and field/reason details |
| `runtime_config.h` | Backend, precision, device, logging, tuning, Session seed |
| `observation.h` | Borrowed named images and typed tensor/language views |
| `prediction.h` | Owned action chunk, auxiliary tensors, and telemetry |
| `policy_spec.h` | Artifact-declared input/output policy contract |
| `model.h` | Immutable shared model resources and Session creation |
| `session.h` | Mutable prediction/reset state |
| `pipeline.h` | Convenience owner of one Model and Session |
| `c_api.h` | C ABI v4 with explicit initialization and ownership |

`Model` is movable and owns a shared `ModelImpl`. `Session` owns a private
`SessionImpl` plus a shared reference to the model implementation, so an existing
Session remains valid after the public Model handle is destroyed. `Pipeline`
combines one Model and Session for the shortest local path.

Observation buffers are borrowed for the duration of `predict()`. Prediction
buffers are owned. C ABI inputs follow the same borrowing rule; returned handles,
strings, predictions, and errors use matching `wam_c_*_free` functions.

## Artifact layer

`src/artifact/` understands storage, not model meaning:

- `gguf_reader.*`: bounded metadata/tensor table parsing and tensor reads;
- `artifact_view.*`: one GGUF or resolved artifact bundle;
- `manifest.*`: strict `wam-bundle-v1` parsing and root-confined resources;
- `tensor_spec.*`: reusable name/dtype/shape validation.

The layer may say that a key or tensor is missing, malformed, or inaccessible.
It does not know which tensors GWP05 or FastWAM require. That knowledge belongs
to `models/*/contract.*`.

A bundle composes a GGUF with optional tokenizer/language encoder paths. It is
not a second source of PolicySpec, runtime tuning, or environment configuration.

## Policy layer

`src/policy/` implements checkpoint-declared, architecture-independent policy
semantics:

- parse and validate schema-v3 PolicySpec;
- validate named image/state/language/action-noise inputs;
- resize/compose RGB images with frozen reference semantics;
- normalize and pad state;
- validate or deterministically generate action noise;
- crop, unnormalize, and recover the final real action chunk.

PolicySpec is authoritative for image roles/transforms, state and action field
order/dimensions, language mode/length, normalization statistics, action frame,
gripper encoding, and recovery. Model-private hidden sizes, layers, schedulers,
latent geometry, and cache layout remain architecture contract fields.

Adapters do not repeat Policy processing. Networks consume prepared tensors and
return model-space tensors; they do not produce controller-specific commands.

## Runtime layer

`src/runtime/` and `src/model_internal.*` own common lifecycle behavior:

- `ModelImpl`/`SessionImpl` private interfaces;
- compile-time architecture registry and capability descriptors;
- public RuntimeConfig validation before resource allocation;
- structured logger callback and phase telemetry;
- unsupported architecture/backend/precision/language-mode errors.

`src/models/builtin_modules.cpp` is the only built-in registration list. Version
0.6 deliberately has no dynamic plugin loader.

## GGML backend

`src/backends/ggml/` owns code genuinely shared by both models:

- backend/device/context lifetime;
- immutable weight storage and tensor I/O;
- graph allocation/build/compute lifetime;
- shared graph operations;
- opt-in debug dumps.

Backend code does not contain Policy, artifact-parser, architecture, or
environment branches. CUDA is explicit and requires a configured architecture;
CPU metadata mode validates artifacts without allocating model weights.

## Model modules

Each model is a vertical module:

```text
src/models/<architecture>/
├── module.*        # architecture descriptor and factory
├── contract.*      # private metadata, tensor schema, geometry
├── model.*         # ModelImpl/SessionImpl and lifecycle
├── pipeline.*      # readable inference stages
├── state.*         # immutable resources and mutable session state
└── networks/       # checkpoint mathematical components
```

Extra scheduler/cache/resource files exist only when required by that model.
GWP05 contains UMT5, vision VAE, MoT and prefix/action cache behavior. FastWAM
contains vision VAE, proprio projector, video DiT, action DiT, and its scheduler.
Their file counts differ because their mathematics and cache lifecycles differ,
not because the framework requires matching templates.

`module.cpp` declares capabilities and a factory; it does not run inference.
`contract.cpp` validates all model-specific metadata/tensors before large device
allocation. `model.cpp` maps common lifecycle to private resources/session state.
`pipeline.cpp` expresses a complete prediction without hiding the main stages in
callbacks. `networks/` implements math only.

See `docs/adding-a-model.md` for the extension and test sequence.

## Python SDK and serving

`python/wam/` is the installed SDK:

- `local.py`: C ABI-backed Model, Session, Pipeline, config, and Prediction;
- `remote.py`: synchronous `wam.rpc.v06` Client;
- `language.py`/`resources.py`: explicit tokenizer/encoder discovery and providers;
- `serving/core.py`: transport-neutral protocol/session state machine;
- `serving/websocket.py`: binary WebSocket transport only;
- `serving/cli.py`: generic `wam-serve` process entry;
- `adapters/`: environment contracts and semantic conversion;
- `eval/`: reusable action execution, manifest, metrics, video, and results.

The service checks the selected EnvironmentContract against ModelInfo/PolicySpec
before creating a Session. Each connection performs an exact v0.6 handshake and
uses contiguous request ids. Structured fatal errors close the connection;
recoverable input errors keep it usable.

When a model does not declare concurrent sessions, `ServiceCore` serializes
inference across connections. WebSocket receive/send remains asynchronous.

Environment runners and protocol tests import these installed modules directly;
there is no second native bridge, language provider, RPC implementation, or
server state machine under `eval/`.

## Environment and evaluation boundary

An `EnvironmentContract` contains canonical image roles, ordered state/action
fields, dimensions, representation/frame, and gripper encoding. Adapters convert
raw simulator observations into named RGB/state inputs and decoded action chunks
into controller commands.

Environment runners own upstream import/setup, reset, step, done/success, and
close. `ActionChunkExecutor` owns queue/truncation/reset mechanics. Manifest,
metrics, video, and result utilities are shared without introducing a generic
simulator base class.

Synthetic adapter tests prove conversion and control-flow contracts. Real
checkpoint numerical parity and real simulator smoke/benchmarks are separate
external release gates.

## Error and telemetry contract

Errors carry a stable ErrorCode, actionable message, and zero or more field/reason
details. Expected validation failures never collapse into a generic “inference
failed”. Validation happens as early as possible: runtime options, bundle,
PolicySpec, model contract, environment compatibility, then request payload.

Prediction telemetry separates preprocessing, vision/text/prefill/decode model
time, postprocessing, total time, architecture-specific named timings, and peak
device memory. Model-specific timings do not form a second inconsistent total.

## Build and release gates

The default suite covers public-header isolation, C++/pure-C install consumers,
artifact and Policy contracts, backend RAII, both synthetic model pipelines,
C ABI/Python lifecycle, wheel installation, RPC state/error/concurrency, and
adapter/eval fixtures.

Large assets remain opt-in through grouped absolute `WAM_TEST_*` CMake variables.
Release evidence distinguishes:

1. default contract tests;
2. real artifact validation;
3. independent same-input numerical parity;
4. real simulator smoke;
5. frozen multi-episode benchmark and model quality.

The current status and limitations are in `docs/support-matrix.md`. Missing
external assets are skipped explicitly and never treated as parity success.

## Explicit non-goals for 0.6

- dynamic model plugins;
- a generic graph/DAG execution framework;
- architecture component virtual interfaces without a real shared consumer;
- gRPC, ROS2, multi-GPU/offload, or automatic device partitioning;
- runtime download of models or language resources;
- compatibility guarantees for the 0.5 C++ API, C ABI v3, RPC v0.5, or legacy
  artifact formats.
