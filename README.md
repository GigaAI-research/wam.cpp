# wam.cpp 0.5

This branch is being rebuilt in verified vertical slices. Slice 7 provides the formal
`wam_core` target, model/session lifecycle, architecture registry, a pinned GGUF reader,
a validated internal `PolicySpecDraft`, the GWP05 metadata/input contract, and the private
GWP05 engine connected to the public Model/Session lifecycle. PolicySpec-driven image,
state, action-noise, and action-decode operations now form the boundary around that engine,
with a structured C ABI and WebSocket/Protobuf evaluation serving path above it.

`Backend::cpu_metadata` loads and cross-validates draft or known legacy GWP05 metadata without
allocating model weights and deliberately rejects session creation. `Backend::automatic` or
`Backend::cuda` loads the private engine, exposes action capability, and supports public
load/create/predict/reset/free lifecycle. The current verified legacy F32 path accepts canonical
named RGB images and raw state, sends only a prepared composite image, normalized/padded state,
and complete action noise into the engine, and returns decoded F32
`[horizon, real_action_dim]` actions. Slice 7 adds the WebSocket/Protobuf server and
RoboTwin client adapter. FastWAM now has an opt-in CUDA/BF16 engine and verified
LIBERO Gate B paths; unsupported profiles still fail explicitly and never return
placeholder actions.

```bash
cmake -S . -B build -DWAM_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

llama.cpp `b9866` is pinned by archive hash and the GWP engine patches are applied during
Slice 4 builds. Offline builds can set
`WAM_LLAMA_ARCHIVE=/path/to/b9866.tar.gz` or point `WAM_LLAMA_SOURCE_DIR` at an existing
source tree. The default build enables the CPU reference backend; CUDA is opt-in with
`WAM_CUDA=ON` and an explicit `CMAKE_CUDA_ARCHITECTURES`, while the BF16 VAE build path
also requires `WAM_CUDNN=ON`. The current GWP contract accepts decoded `rgb_u8` named images, little-endian F32
state, tokens or an external embedding as declared by PolicySpec, and optional little-endian
F32 `[horizon, model_action_dim]` action noise.

The model entry points can be configured independently with `WAM_BUILD_GWP05` and
`WAM_BUILD_FASTWAM`. Ignored upstream simulator directories remain managed by the eval setup
scripts.

Environment/runtime integration and deployable model readiness are tracked separately.
FastWAM LIBERO passes its formal MuJoCo 3.3.2 four-suite gate: donor Python
achieves `1935/2000` (`96.75%`) and wam.cpp achieves `1938/2000` (`96.90%`)
on the same 50 ordered init states per task. See
`eval/sim/FASTWAM_LIBERO_BASELINE_AUDIT.md` for the frozen manifests and
interpretation.

For FastWAM LIBERO-X, the client/server adapter, z-score PolicySpec, GGUF artifact gate,
same-input numerical parity, and fixed-manifest donor/wam.cpp rollout agreement are complete.
The audited step 50k training run has an invalid resumed cosine-scheduler trajectory and only
two independent demonstrations for the exact frozen SCENE1 task, so its GGUF is an integration
fixture rather than a supported model. A reliable LIBERO-X checkpoint must be retrained and
pass the same gates before success-rate claims are published; see
`eval/sim/FASTWAM_LIBEROX_REMOTE_EVAL.md`.

Slice 4A through Slice 6 add opt-in external gates for a real GWP05 GGUF, the frozen donor tensor
manifest, private-engine F32 stage parity, and public multi-session lifecycle parity. See
`tests/reference/README.md`. These gates keep models and replay payloads out of
Git. A formal 14-dimension z-score RoboTwin PolicySpec GGUF passes the
converter/inspector/artifact gates and independent public CUDA/BF16 numerical
parity on A800. The Slice 7 cross-container `beat_block_hammer` smoke test also succeeded at
step 106 (`1/1`). The subsequent fixed-manifest 100-episode benchmark completed at
`87/100` (`87.0%`) with 378 action-chunk requests and no RPC/runtime failure; steady-state
client infer, RPC, and server-total means were `132.29`, `130.77`, and `124.98 ms` on an
A800 BF16 server. See `eval/sim/ROBOTWIN_REMOTE_EVAL.md` for the manifest digest, latency
groups, and result paths.

## Verified checkpoint (2026-07-29)

The current vertical implementation passes Release CPU and CUDA 12.4 builds. The CPU build
passes all 20 configured tests. The A800 `sm_80` CUDA build passes all 24 configured tests,
including WebSocket/Protobuf RPC, language-provider padding semantics, the formal GWP05
RoboTwin BF16 public-Session parity gate, and the FastWAM LIBERO artifact and action-parity
gates. A real prediction through the Python/C ABI bridge returns the expected `[48, 14]`
GWP05 action and preserves the engine's named vision, text, prefill, and decode timings.
External GGUF, numerical fixtures, checkpoints, simulator checkouts, and raw rollout results
remain outside Git.

Phase 7 publishes C ABI v4 and the installable `wam` Python SDK. The current default CPU
and CUDA 12.4 + cuDNN `sm_80` matrices each pass 41/41 tests; the Runtime-only build with
GWP05, FastWAM, and Serving disabled passes 27/27. These matrices include pure-C and C++
install consumers, wheel build and isolated installation, context-managed native loading,
structured error ownership, bundle language-resource discovery, and the `wam-predict`
entry point. See `docs/python-sdk.md`. Real-model cross-language parity remains an opt-in
external gate because its GGUF, frozen observations, and reference actions are not stored
in Git.
