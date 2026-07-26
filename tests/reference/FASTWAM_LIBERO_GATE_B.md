# FastWAM LIBERO Gate B

The first Gate B profile is `fastwam_libero_2cam224_minmax`. It was selected because the
release checkpoint has a complete action-only Python reference, a materialized BF16 donor
GGUF, fixed image/state/external-embedding/action-noise inputs, and a passing final action
comparison on A800.

The path-free facts used by the gate are frozen in
`fastwam_libero_gate_b_contract.json`. Model, GGUF, embedding, and numerical tensor payloads
remain external to Git.

Gate B does not accept the donor implementation as framework code. The donor establishes the
private FastWAM mathematics and numerical oracle only. The 0.5 target must:

1. load a schema-v2 PolicySpec artifact for the selected profile;
2. reuse the common named-image, state normalization, explicit action-noise, and action decode
   functions already used by GWP05;
3. expose only an action chunk and reject video/world public inputs or outputs;
4. pass artifact, input, lifecycle, and fixed-fixture CUDA/BF16 action parity gates;
5. keep architecture selection in the registry and environment compatibility in the server;
6. resolve or explicitly document the donor intermediate-tensor mismatches before Gate B is
   marked complete.

Gate B passes for the selected profile. The schema-v2 target contains 1741 tensors and uses
separate `fastwam.norm_eps=1e-6` model metadata and `1e-8` policy normalization epsilon. Its
public CUDA/BF16 `Model/Session` output is bit-identical to the donor C++ `[32, 7]` action. Both
compare to the independent PyTorch reference at MAE `0.0009015057502048356` and maximum absolute
error `0.0054931640625`, within the frozen `0.001/0.01` thresholds.

Some donor video K/V tensors and the final two velocity steps exceed the donor's stricter
intermediate threshold against PyTorch. Because the migrated runtime is bit-identical at the
final donor boundary, this is recorded as a pre-existing BF16/CUDA donor-reference limitation,
not a migration regression. Kernel or dependency changes must rerun the fixed fixture and may
not relax the final action thresholds.

The same public session gate covers explicit noise, session-generated noise, RNG advancement,
and exact RNG replay after `session_reset()`. Multiple sessions share a model-owned engine lock;
`concurrent_sessions=false` remains the advertised scheduling capability.

The serving extension of Gate B also passes. RPC continues to carry a raw instruction; the
Python server owns the exact Wan UMT5 tokenizer/encoder used by FastWAM preprocessing and passes
the resulting BF16 `[128, 4096]` embedding plus I32 mask through C ABI v3. The provider uses a
bounded prompt cache, so repeated instructions do not rerun the text encoder.

The cross-container `libero_spatial` task-0, init-state-0 rollout completed successfully in 85
controller steps and 9 action-chunk requests. The first request reported 405.42 ms of external
text-model time; cached requests reported zero text-model time. This is an integration gate,
not a benchmark success-rate result. The exact deployment commands and dependencies are recorded
in `eval/sim/FASTWAM_LIBERO_REMOTE_EVAL.md`.
