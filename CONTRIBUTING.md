# Contributing

Build and install the CPU runtime before submitting a change:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DWAM_CUDA=OFF -DWAM_BUILD_APPS=ON
cmake --build build --parallel
cmake --install build --prefix build/install
python3 scripts/release/check_source_tree.py .
```

Keep public headers independent of ggml, CUDA, Protobuf, and gRPC. The public
repository contains the deployable GWP-0.5 runtime and open-loop integration;
do not add internal replay captures or validation payloads.

Do not commit build outputs, model files, dumps, profiles, credentials, or
private filesystem paths. Do not add replay/model payloads unless their
redistribution rights and provenance are documented.

Changes to public API, Proto field numbers, artifact schema, C ABI, or precision
policy require an explicit compatibility review and version update.

New model implementations must keep registration, artifact validation, input
semantics, and compute-engine ownership inside their model directory. Reuse
`models/common` only for genuinely model-independent mechanisms.
