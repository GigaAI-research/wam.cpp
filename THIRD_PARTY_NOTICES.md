# Third-Party Notices

wam.cpp source code is licensed under the MIT License. Model checkpoints,
converted artifacts, replay data, CUDA, and cuDNN are separate works and are
not licensed by the repository MIT License.

## Runtime and bundled-source dependencies

| Dependency | Pinned version | Use | License |
| --- | --- | --- | --- |
| llama.cpp / ggml | tag `b9866`, archive SHA-256 `fb4f9e480938406dd17a27e9a3b4b99fb558685d8ad44040778062c5e999d8ae` | GGUF, graph runtime, CPU/CUDA backends | MIT |
| nlohmann/json | copy shipped by pinned llama.cpp | CLI/replay JSON | MIT |
| stb_image 2.30 | copy shipped by pinned llama.cpp | PNG/JPEG decode | Public domain or MIT |
| gguf-py | copy shipped by pinned llama.cpp | converter and inspector | MIT |

The llama.cpp source is patched at configure time. Patch hashes are:

- `llama-b9866-native-bf16.patch`: `e9f21832755d3e5425c2bd35fab3c0c0a00e667343dd01aedc4bf3d8ed146eef`
- `llama-b9866-p7-fusions.patch`: `4e5fb3d09bec3cee09a8dd3aeb3d31d3a7d507ba5b260a491bcee56399188f54`

The upstream license files remain in the fetched llama.cpp source. Binary
distributions must reproduce the applicable notices.

## Optional tool dependencies

| Dependency | Validated version | Use | License |
| --- | --- | --- | --- |
| PyTorch | 2.7.1 | checkpoint conversion | BSD-3-Clause |
| NumPy | 1.26.4 | conversion and replay validation | BSD-3-Clause |
| safetensors | 0.5.3 | Wan weight input | Apache-2.0 |
| protobuf | 6.33.6 | contract tests and dynamic client messages | BSD-3-Clause |
| grpcio | 1.82.0 | optional Python gRPC transport | Apache-2.0 |

These Python packages are not vendored or redistributed by wam.cpp.

## NVIDIA software

CUDA and cuDNN are optional system dependencies governed by NVIDIA's license
terms. They are not covered by the wam.cpp MIT License.

## Models and replay data

Wan/GWP checkpoints and GGUF artifacts retain their source licenses. The
converter does not grant new rights. The historical fixed and independent
replay captures have no recorded redistribution approval and must not be
included in a public distribution until their owner approves it.
