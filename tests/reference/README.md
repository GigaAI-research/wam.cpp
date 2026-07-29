# GWP05 External References

The committed manifest contains no checkpoint, replay, absolute internal path,
or full-model hash. It freezes the input hashes, selected tensor hashes,
shapes, dtypes, byte order, action boundaries, and parity tolerances needed to
port the donor engine.

The primary engine oracle is one F32 CPU run of the complete MoT graph with a
real 32-dimension quantile GGUF and explicit action noise. The prefix-cache
oracle remains a separately identified audited F32 CUDA capture because CUDA
was unavailable while the primary oracle was refreshed. They must not be
treated as one execution.

Configure the external gate with explicit local paths:

```bash
cmake -S . -B build-external-gwp05 \
  -DWAM_TEST_GWP05_REAL_GGUF=/models/gwp05-f32.gguf \
  -DWAM_TEST_GWP05_REPLAY_ROOT=/private/replay/gwp05 \
  -DWAM_TEST_GWP05_MATERIALIZED_INPUT_DIR=/private/replay/gwp05-ppm-f32 \
  -DWAM_TEST_GWP05_STAGE_ROOT=/private/stages/gwp05-f32-complete \
  -DWAM_TEST_GWP05_CACHE_STAGE_ROOT=/private/stages/gwp05-f32-cache \
  -DWAM_TEST_GWP05_DONOR_EXECUTABLE=/private/bin/gwp-bench
cmake --build build-external-gwp05
ctest --test-dir build-external-gwp05 --output-on-failure \
  -R 'wam_gwp05_(real_artifact|donor_reference|engine_reference)'
```

`WAM_TEST_GWP05_MATERIALIZED_INPUT_DIR` contains decoded P6 PPM images plus
`state.f32`, `noise.f32`, and `t5_embedding.f32`. It is separate from the
distribution-blocked replay manifest so the engine test does not add a PNG
decoder or copy private payloads into this repository. The private-engine gate
compares the complete F32 CPU path through normalized `[48,32]` model-space
action and verifies reset/repeat determinism with explicit action noise. The
public API portion reloads the same artifact, compares the
decoded `[48,14]` action with the donor, verifies two seeded sessions share one
weight residency while keeping RNG/cache state isolated, and checks reset plus
model-handle-before-session lifetime behavior.

`gwp05_reference_manifest.py --write` is only used deliberately when replacing
the oracle. Normal validation omits `--write` and fails if any selected payload
or executable differs.

`gwp05_robotwin_checkpoint_contract.json` records the separate current 14D
z-score RoboTwin contract, formal packed-BF16 GGUF descriptor, and predeclared
Gate A tolerances and accepted metrics. Conversion, inspection, C++ artifact
loading, independent PyTorch BF16 capture, and public CUDA/BF16 Session parity
pass on A800. See `GWP05_ROBOTWIN_GATE_A.md` for the exact local commands and
the distinction between runtime parity and simulator task success.
