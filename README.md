# wam.cpp 0.6

`wam.cpp` is a C++17 runtime and Python SDK for deploying robot policy models
as GGUF artifacts. It provides one execution path for local C++/Python
inference, remote Protobuf/WebSocket serving, and simulator evaluation.

Version 0.6 includes GWP05 and FastWAM model implementations, a stable C++ API,
C ABI v4, an installable Python package, and adapters for RoboTwin, LIBERO, and
LIBERO-X. CUDA is the primary prediction backend. CPU mode is intended for
artifact inspection, contract validation, and development tests.

The runtime owns model loading, observation preprocessing, execution, and
action recovery. Environment adapters own simulator-specific camera, state,
and controller conventions. Model code therefore does not depend on RoboTwin,
LIBERO, or LIBERO-X.

## Model and simulator support

The table distinguishes a usable end-to-end path from an adapter that is still
being qualified. A model being present in the source tree does not by itself
mean that every environment is supported.

| Model | RoboTwin | LIBERO | LIBERO-X |
| --- | --- | --- | --- |
| **GWP05** | **Ready**<br>Source weights: external<br>GGUF: convert locally | **Preparing** | **Preparing** |
| **FastWAM** | **Ready**<br>Source weights: external<br>GGUF: convert locally | **Ready**<br>Source weights: external<br>GGUF: convert locally | **Ready**<br>Source weights: preparing<br>GGUF: convert locally |

Status definitions:

- **Ready**: the model, GGUF contract, serving path, and simulator client run
  end to end for this combination.
- **Preparing**: the model is still being trained; evaluation and public
  release will follow.

The source weights and GGUF artifacts referenced in the table will be released
soon.

## Quick start

### Requirements

- CMake 3.22 or newer
- A C++17 compiler and `patch`
- Python 3.9 or newer
- `protoc` when serving is enabled
- A CUDA toolkit for model prediction
- cuDNN for the validated GWP05 BF16 VAE path

The first CMake configure downloads the pinned llama.cpp `b9866` archive. For
an offline build, set `WAM_LLAMA_ARCHIVE` to a local archive or
`WAM_LLAMA_SOURCE_DIR` to an existing llama.cpp source tree.

Build the CPU metadata and test configuration:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWAM_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For CUDA prediction, configure an explicit architecture. The following example
targets NVIDIA SM80:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DWAM_CUDA=ON -DWAM_CUDNN=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda --parallel
python -m pip install .
```

Inspect and validate a GGUF without creating a prediction session:

```bash
./build-cuda/wam-inspect /path/to/policy.gguf
./build-cuda/wam-validate /path/to/policy.gguf
```

Run one prediction from an NPZ input. The input must contain an F32 `state`
array and HWC U8 `image.<role>` arrays matching the contract printed by
`wam-inspect`.

```bash
wam-predict /path/to/policy-bundle input.npz \
  --library "$PWD/build-cuda/libwam_c_api.so" \
  --instruction "pick up the cup" \
  --backend cuda --precision bf16 --output action.npy
```

An artifact bundle is a directory containing the GGUF, its manifest, and any
local tokenizer or language-encoder resources required by the model.

## Download and convert GGUF

### Pre-converted models

Pre-converted GGUF files are not currently distributed from this repository.
When public artifacts are available, their source-weight and GGUF links will be
listed in the support matrix.

### Converter dependencies

The conversion scripts require NumPy, PyTorch, Safetensors, and llama.cpp's
`gguf-py` package. Running CMake once makes the pinned `gguf-py` source
available under the build tree:

```bash
python -m pip install numpy torch safetensors
export PYTHONPATH="$PWD/build/_deps/llama-src/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
```

Converters refuse to overwrite an existing output. Use `--dry-run` first to
check checkpoint structure, PolicySpec geometry, and conversion options without
writing a GGUF.

### Convert GWP05

The GWP05 converter combines the policy checkpoint, UMT5 text encoder, Wan VAE,
normalization statistics, and PolicySpec into one schema-v3 GGUF. The
`--base-model` directory must be a Wan2.2 TI2V Diffusers model containing
`text_encoder/` and `vae/`.

```bash
python scripts/convert/convert_gwp05.py \
  --checkpoint /path/to/gwp05/transformer_ema \
  --base-model /path/to/Wan2.2-TI2V-Diffusers \
  --norm-stats /path/to/gwp05/norm_stats.json \
  --policy-spec profiles/gwp05_robotwin_14d_zscore.json \
  --weight-policy mot-vae-bf16 \
  --out /path/to/gwp05-robotwin.gguf \
  --dry-run
```

Remove `--dry-run` after the inputs pass validation. Available storage policies
are `source-f32`, `mot-bf16`, and `mot-vae-bf16`; select one explicitly rather
than inferring precision from the output filename.

### Convert FastWAM

FastWAM stores the action policy, VAE, proprioceptive projector, normalization
statistics, and PolicySpec in a schema-v3 GGUF. Its UMT5 tokenizer and text
encoder remain external deployment resources.

For LIBERO:

```bash
python scripts/convert/convert_fastwam.py \
  --checkpoint /path/to/fastwam/checkpoint.pt \
  --vae /path/to/fastwam/vae.safetensors \
  --norm-stats /path/to/fastwam/norm_stats.json \
  --policy-profile profiles/fastwam_libero_2cam224_minmax.json \
  --num-inference-steps 10 \
  --out /path/to/fastwam-libero.gguf \
  --dry-run
```

Environment-specific profiles are also provided for RoboTwin and LIBERO-X:

- `profiles/fastwam_robotwin_3cam384_zscore.json`
- `profiles/fastwam_liberox_2cam224_zscore.json`

`--num-inference-steps` is required because the supported value belongs to the
deployment profile rather than the network weights. The released FastWAM
LIBERO and RoboTwin evaluation configurations both use 10 steps. Remove
`--dry-run` to write the artifact, then inspect and validate it:

```bash
./build-cuda/wam-inspect /path/to/policy.gguf
./build-cuda/wam-validate /path/to/policy.gguf
```

These tools produce BF16/F32 artifacts. General-purpose GGUF quantization is
not part of the 0.6 release.

## Local and remote inference

### Python API

```python
import numpy as np
import wam

with wam.Pipeline.load(
    "/path/to/policy-bundle",
    library="build-cuda/libwam_c_api.so",
    runtime_config=wam.RuntimeConfig(backend="cuda", precision="bf16"),
) as pipeline:
    prediction = pipeline.predict(
        images,
        np.asarray(state, dtype=np.float32),
        instruction="pick up the cup",
    )
```

See [examples/python/predict.py](examples/python/predict.py) for the minimal
local function.

### C++ API

Install the CMake package and link the stable public target:

```bash
cmake --install build-cuda --prefix "$PWD/install"
cmake -S your-app -B your-app/build -DCMAKE_PREFIX_PATH="$PWD/install"
```

```cmake
find_package(wam 0.6 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE wam::core)
```

```cpp
#include <wam/wam.h>

wam::Pipeline pipeline = wam::Pipeline::load("policy.gguf");
wam::Prediction prediction = pipeline.predict(observation);
```

Application-owned buffers referenced by `wam::Observation` only need to remain
alive for the duration of `predict()`. A compilable example is available in
[examples/cpp/predict.cpp](examples/cpp/predict.cpp).

### Remote serving

The server derives architecture, observation geometry, action shape, and
language mode from the artifact. It rejects an incompatible environment before
creating a session.

```bash
wam-serve \
  --library "$PWD/build-cuda/libwam_c_api.so" \
  --descriptor "$PWD/build-cuda/wam.desc" \
  --model /path/to/policy-bundle \
  --environment robotwin \
  --backend cuda --precision bf16 \
  --host 0.0.0.0 --port 18060
```

When a bundle does not declare its resources, pass `--tokenizer` explicitly.
FastWAM also requires `--text-encoder`; use `--language-python-root` when the
encoder implementation is outside the active Python environment.

```python
import wam

with wam.Client("127.0.0.1", 18060, "build-cuda/wam.desc", "robotwin") as client:
    action, stats = client.predict(images, state, "pick up the cup")
```

The wire protocol is binary Protobuf over WebSocket using package
`wam.rpc.v06`. See
[examples/python/remote_predict.py](examples/python/remote_predict.py) for the
minimal client function.

## Simulation

Simulation uses a separate client process so that simulator dependencies do not
enter the C++ runtime:

```text
RoboTwin / LIBERO / LIBERO-X client
                  |
         WebSocket + Protobuf
                  |
              wam-serve
                  |
          C ABI / C++ Session
                  |
           GWP05 / FastWAM
```

The setup scripts pin simulator revisions but do not install the simulators'
system, Conda, or GPU dependencies. Complete those steps using the upstream
project instructions.

### RoboTwin

Fetch the pinned RoboTwin source:

```bash
bash eval/sim/setup_robotwin.sh
```

Start `wam-serve` with `--environment robotwin`, then run one episode:

```bash
python eval/sim/run_robotwin_client.py \
  --robotwin-root eval/sim/RoboTwin \
  --descriptor build-cuda/wam.desc \
  --task beat_block_hammer \
  --task-config demo_clean \
  --episodes 1 \
  --host 127.0.0.1 --port 18060 \
  --metrics-output results/robotwin-smoke.json \
  --save-video
```

GWP05 is the release-ready RoboTwin path. FastWAM can use the same transport
and adapter but remains under integration testing.

### LIBERO

Fetch the pinned LIBERO source:

```bash
bash eval/sim/setup_libero.sh
```

Start `wam-serve` with `--environment libero` and the FastWAM tokenizer and text
encoder resources, then run one episode:

```bash
python eval/sim/run_libero_client.py \
  --libero-root eval/sim/LIBERO \
  --descriptor build-cuda/wam.desc \
  --suite libero_spatial \
  --task-ids 0 \
  --episodes 1 \
  --host 127.0.0.1 --port 18060 \
  --metrics-output results/libero-smoke.json \
  --video-output results/libero-smoke.mp4
```

The client also supports fixed manifests, episode ranges, and `--resume` for
repeatable or sharded evaluation. Run it with `--help` for those controls.

### LIBERO-X

The LIBERO-X adapter and client are present, but automatic environment setup is
not implemented in 0.6 and model quality is still being qualified. Prepare an
upstream LIBERO-X checkout manually, start `wam-serve` with
`--environment liberox`, and inspect the available client options with:

```bash
python eval/sim/run_liberox_client.py --help
```

This path is intended for integration work and is not currently presented as a
release-quality evaluation workflow.

## Inference performance

The following open-loop results were measured on 2026-07-30 using the current
0.6 implementation on one NVIDIA A800 80GB PCIe GPU, CUDA 12.4.131, cuDNN 9,
SM80, and BF16 compute. Each case ran in a fresh process with 8 warm-up requests
followed by 30 measured requests.

Inputs use fixed real evaluation fixtures, explicit action noise, and prepared
language embeddings. The results measure local C ABI/C++ Session inference;
they do not include UMT5 encoding, WebSocket transport, simulator stepping, or
video recording.

| Model / environment | Image input | Denoise steps | Steady total p50 | Decode p50 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| GWP05 / RoboTwin | 3 x 640 x 480, composed to 384 x 320 | 10 | **122.01 ms** | 76.20 ms | 8.20 req/s |
| FastWAM / RoboTwin | 3 x 640 x 480, composed to 384 x 320 | 10 | **211.27 ms** | 147.70 ms | 4.73 req/s |
| FastWAM / LIBERO | 2 x 224 x 224, composed to 224 x 448 | 10 | **198.02 ms** | 146.75 ms | 5.05 req/s |

The FastWAM measurements use the optimized execution path: VideoDiT K/V tensors
remain resident on the device, and one persistent unrolled graph executes the
complete action-denoise schedule.

The released FastWAM LIBERO and RoboTwin evaluation configurations both use 10
denoise steps. Decode remains the dominant cost. The 10-step LIBERO replay gate
against the official PyTorch implementation remains within its frozen tolerance
(`mean_abs=0.000853`, `max_abs=0.005493`).

## Architecture and extension

The dependency direction is fixed:

```text
applications / evaluation / serving
                  |
            public wam API
                  |
     artifact + policy + runtime
                  |
         model implementations
                  |
            GGML backend
```

- `include/wam/` is the stable user-facing C++ API. Public headers never expose
  GGML or model-private types.
- `src/artifact/` reads GGUF, manifests, and generic tensor descriptions.
- `src/policy/` implements reusable observation preprocessing and action
  recovery.
- `src/runtime/` owns models, sessions, registration, errors, and telemetry.
- `src/models/` contains model contracts, pipelines, state, caches, and network
  mathematics.
- `src/backends/ggml/` owns shared GGML contexts, weights, graphs, tensor I/O,
  and backend operations.
- `python/wam/` exposes local inference, serving, resources, and environment
  adapter contracts.
- `eval/sim/` contains simulator setup and evaluation clients.

To add a model, define one architecture contract, register its module, keep the
inference sequence readable in its model-local pipeline, and place only model
mathematics below `networks/`. Do not put GGUF parsing, simulator conventions,
or service code in a network implementation.

To add an environment, implement the Python adapter contract and simulator
client using public model metadata. An environment adapter may map cameras,
state, and actions, but must not depend on a model's private C++ types.

Every extension should include contract tests, malformed-input tests, install
consumption coverage, and an explicit real-asset gate before its support status
is promoted to **Ready**.

## Build options and limitations

- `WAM_BUILD_GWP05` and `WAM_BUILD_FASTWAM` independently select model entry
  points.
- `WAM_BUILD_SERVING` controls the C ABI and RPC descriptor.
- `WAM_BUILD_APPS` controls `wam-inspect` and `wam-validate`.
- `WAM_CUDA` enables CUDA and requires an explicit CUDA architecture.
- `WAM_CUDNN` enables the GWP05 BF16 VAE path and requires CUDA.

External numerical and simulator gates are enabled only when their grouped
`WAM_TEST_*` assets are configured. Missing external assets are reported as
skipped, never as passed.

Version 0.6 does not include gRPC, ROS2, dynamic plugins, multi-GPU/offload,
automatic model downloads, or general-purpose quantization.

## License

See [LICENSE](LICENSE).
