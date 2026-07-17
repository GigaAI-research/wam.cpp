# wam.cpp

`wam.cpp` is an inference runtime for world-action models (WAM) and vision-language-action models (VLA). It uses GGUF artifacts and the pinned llama.cpp/ggml backend to provide
reproducible C++/CUDA execution, model conversion, inference, and serving.

The initial v0.1 release supports the GWP-0.5 action policy from
[`open-gigaai/giga-world-policy`](https://github.com/open-gigaai/giga-world-policy).


## 1. Overview

The v0.1 runtime provides:

- `libwam_core` and a public C++ API;
- `libwam_c_api` and a serialized-Proto in-process service bridge;
- `wam-cli`, `wam-bench`, and `wam-compare`;
- GWP-0.5 checkpoint to GGUF conversion;
- server-side compatibility with the upstream GWP-0.5 client.

## 2. Current Support

### 2.1 Supported Models

| Model | Status |
| --- | --- |
| GWP-0.5 | Supported |
| Other WAM/VLA models | planned

wam.cpp for GWP-0.5 supports three language residency policies:

- `external_embedding`: never loads T5 and accepts precomputed embeddings;
- `fixed`: encodes one configured token prompt during model loading, releases
  T5, and then loads VAE plus MoT;
- `resident`: keeps T5 loaded and accepts arbitrary request tokens or
  embeddings.


### 2.2 Supported Hardware And Performance

| GPU | Precision | Latency | Device-memory |
| --- | --- | ---: | ---: |
| NVIDIA RTX 4090 | BF16 | 85 ms | 12.15 GB  |
| NVIDIA A800 | BF16 | 108 ms | 13.07 GB |


## 3. Roadmap

### 3.1 Simulation Environments

- LIBERO
- LIBERO-X;
- RoboTwin;
- ...

### 3.2 Model Architectures

- FastWAM
- DreamZero;
- pi0/0.5;
- StarVLA;
- OpenVLA;
- ...

### 3.3 Hardware Backends

- broader NVIDIA GPU coverage;
- more hardward backends.

### 3.4 Model Compression and Inference Acceleration

- Low-bit weight and activation quantization；
- Diffusion cache；
- Token pruning and visual-token reduction
- KV-cache optimization；
- ...

## 4. Quick Start

### 4.1 Prerequisites

For a CUDA GWP-0.5 build:

- Linux with a C++17 compiler, OpenMP, `git`, and `patch`;
- CMake 3.22 or newer;
- `protoc` available during the build so `wam.desc` can be packaged;
- CUDA Toolkit and cuDNN compatible with the selected GPU;
- Python 3.10 or newer for conversion and the compatibility adapter;
- enough disk space for the checkpoints and a 23.5 GB packed-BF16 GGUF.

Use CUDA architecture `80` for A800 and `89` for RTX 4090. A CPU-only build can
validate compilation, installation, and metadata handling, but it
cannot run GWP-0.5 prediction.

### 4.2 Clone and Build wam.cpp

Clone the repository or extract a release source archive:

```bash
git clone https://github.com/GigaAI-research/wam.cpp.git
cd wam.cpp
```

The build pins llama.cpp revision `b9866` and automatically
applies the required wam.cpp patches. By default CMake downloads the pinned
archive. For an offline or audited build, provide exactly one of:

```bash
export WAM_LLAMA_ARCHIVE=/path/to/llama.cpp-b9866.tar.gz
# or:
export WAM_LLAMA_SOURCE_DIR=/path/to/clean/llama.cpp-b9866
```

Configure a CUDA build:

```bash
export WAM_CUDA_ARCH=89  # 80 for A800; 89 for RTX 4090

cmake_args=(
  -S . -B build-cuda
  -DCMAKE_BUILD_TYPE=Release
  -DWAM_CUDA=ON
  -DWAM_CUDNN=ON
  -DCMAKE_CUDA_ARCHITECTURES="${WAM_CUDA_ARCH}"
  -DWAM_BUILD_APPS=ON
  -DWAM_BUILD_GIGA_WORLD_POLICY=ON
  -DWAM_PROTOC_EXECUTABLE="$(command -v protoc)"
)
if [[ -n "${WAM_LLAMA_ARCHIVE:-}" ]]; then
  cmake_args+=("-DWAM_LLAMA_ARCHIVE=${WAM_LLAMA_ARCHIVE}")
fi
if [[ -n "${WAM_LLAMA_SOURCE_DIR:-}" ]]; then
  cmake_args+=("-DWAM_LLAMA_SOURCE_DIR=${WAM_LLAMA_SOURCE_DIR}")
fi

cmake "${cmake_args[@]}"
cmake --build build-cuda --parallel
```

For a CPU-only build check, replace the CUDA/cuDNN/architecture options with
`-DWAM_CUDA=OFF`.

### 4.3 Install wam.cpp

Choose a user-writable install prefix; no directory needs to be prepared in
advance:

```bash
export WAM_PREFIX="${HOME}/.local/wam"
cmake --install build-cuda --prefix "${WAM_PREFIX}"

test -x "${WAM_PREFIX}/bin/run_inference_openloop_wam-cpp.sh"
test -f "${WAM_PREFIX}/lib/libwam_c_api.so" || \
  test -f "${WAM_PREFIX}/lib64/libwam_c_api.so"
test -f "${WAM_PREFIX}/share/wam/proto/wam.desc"
"${WAM_PREFIX}/bin/run_inference_openloop_wam-cpp.sh" --help
```

### 4.4 GWP-0.5 Open-Loop Inference

#### 4.4.1 Prepare the GigaWorld-Policy Environment

Use the frozen compatible upstream revision:

```bash
export GWP_ROOT="${HOME}/src/giga-world-policy"

git clone https://github.com/open-gigaai/giga-world-policy.git "${GWP_ROOT}"
git -C "${GWP_ROOT}" checkout 5d55073a6508de7354c83679d9028f4010ff6cb2

conda create -n gigaworld-policy python=3.11 -y
conda activate gigaworld-policy
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "${GWP_ROOT}/requirements.txt"
python -m pip install --no-deps -e "${GWP_ROOT}/third_party/giga-train"
python -m pip install --no-deps -e "${GWP_ROOT}/third_party/giga-models"
python -m pip install --no-deps -e "${GWP_ROOT}/third_party/giga-datasets"
python -m pip install -r \
  "${WAM_PREFIX}/share/wam/integrations/giga-world-policy/giga-world-policy.txt"
```

#### 4.4.2 Prepare Dataset, Norm Stats, and T5 Embeddings

Follow the upstream dataset contract. For a LeRobot v3.0 dataset:

```bash
cd "${GWP_ROOT}"

python scripts/compute_wam_task_norm.py \
  --data-root /data/lerobot-v3 \
  --output /models/gwp/norm_stats.json \
  --model-dim 32 \
  --action-horizon 48

python scripts/compute_t5_embedding.py \
  --root /data/lerobot-v3 \
  --wan_path /models/Wan2.2-TI2V-5B-Diffusers \
  --device cuda \
  --t5_folder_name t5_embedding
```

The open-loop client reads the three camera views, state, and precomputed
prompt embedding from this prepared dataset. Use data and checkpoints whose
licenses permit your intended use.

#### 4.4.3 Download or Convert GGUF

From the wam.cpp source tree:

```bash
cd /path/to/wam.cpp
python -m pip install -r requirements/convert.txt

export LLAMA_SOURCE="${WAM_LLAMA_SOURCE_DIR:-${PWD}/build-cuda/_deps/llama-src}"
export PYTHONPATH="${LLAMA_SOURCE}/gguf-py${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/convert/convert_gwp05.py \
  --checkpoint /models/gwp/transformer_ema \
  --base-model /models/Wan2.2-TI2V-5B-Diffusers \
  --norm-stats /models/gwp/norm_stats.json \
  --weight-policy mot-vae-bf16-qkv \
  --out /models/gwp/gwp05-packed-bf16.gguf

python scripts/inspect/inspect_gguf.py \
  /models/gwp/gwp05-packed-bf16.gguf \
  --expect-policy mot-vae-bf16-qkv-v1
```

#### 4.4.4 Start the wam.cpp Server

In terminal 1, with the GigaWorld-Policy Python environment active:

```bash
export CUDA_VISIBLE_DEVICES=0
export GIGA_WORLD_POLICY_ROOT="${GWP_ROOT}"
export WAM_MODEL=/models/gwp/gwp05-packed-bf16.gguf
export WAM_LANGUAGE_POLICY=external_embedding
export WAM_PRECISION=bf16
export DEVICE=0
export HOST=127.0.0.1
export PORT=11444

"${WAM_PREFIX}/bin/run_inference_openloop_wam-cpp.sh" server
```

`DEVICE` is the logical index inside `CUDA_VISIBLE_DEVICES`. With
`external_embedding`, the normal upstream client supplies the request prompt
embedding. For a trusted server-side fallback, additionally set:

```bash
export FIXED_T5_PATH=/models/gwp/task_prompt_embedding.pt
```

To expose the one-filename replacement inside the upstream checkout, apply the
additive patch while keeping the frozen Git HEAD unchanged:

```bash
git -C "${GWP_ROOT}" apply \
  "${WAM_PREFIX}/share/wam/integrations/giga-world-policy/upstream/0001-add-wam-cpp-entrypoints.patch"

cd "${GWP_ROOT}/scripts"
WAM_INSTALL_PREFIX="${WAM_PREFIX}" \
  ./run_inference_openloop_wam-cpp.sh server
```

The ready line is:

```text
Server is ready and listening on tcp://127.0.0.1:11444
```

#### 4.4.5 Start the Original Upstream Client

In terminal 2, activate the same GigaWorld-Policy environment and keep `HOST`
and `PORT` identical to the server:

```bash
conda activate gigaworld-policy

export CHECKPOINT=/models/gwp/transformer_ema
export NORM_STATS=/models/gwp/norm_stats.json
export DATA_PATH=/data/lerobot-v3
export BASE_MODEL=/models/Wan2.2-TI2V-5B-Diffusers
export HOST=127.0.0.1
export PORT=11444

cd "${GWP_ROOT}/scripts"
./run_inference_openloop.sh client
```

The client remains upstream code. wam.cpp only replaces the server-side policy
execution.
