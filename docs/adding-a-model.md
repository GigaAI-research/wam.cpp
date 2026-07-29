# Adding a Model

A model extension should be local to `src/models/<architecture>/` plus one
registration line and its tests. It must not add architecture branches to the
public API, Policy layer, serving code, or environment adapters.

## Required module shape

```text
src/models/example/
├── module.cpp              # descriptor and factory only
├── module.h
├── contract.cpp            # architecture metadata/tensor/geometry validation
├── contract.h
├── model.cpp               # ModelImpl and SessionImpl lifecycle
├── model.h
├── pipeline.cpp            # readable inference stages
├── pipeline.h
├── state.cpp               # immutable resources and mutable session state
├── state.h
└── networks/               # checkpoint mathematical components
```

Only create additional files when the model has a real responsibility such as
a scheduler or prefix cache. Do not mirror another module's filenames
mechanically.

## Extension sequence

1. Define the GGUF architecture id, schema-v3 profile, model-private metadata,
   tensor names, shapes, and dtypes. Public input/output geometry belongs to
   PolicySpec, not duplicated architecture keys.
2. Implement `contract.*` using `ArtifactView`/`GgufReader` and tensor schema
   helpers. Reject unsupported variants before weight allocation with an
   `incompatible_artifact` error and field-level details.
3. Put mathematical forward operations in `networks/`. Network code cannot
   parse GGUF, read environment variables, know simulator names, or perform
   policy normalization/action recovery.
4. Implement immutable model resources and per-session mutable state. A Session
   must remain usable after its public Model handle is released and `reset()`
   must have deterministic seed/cache semantics.
5. Write `pipeline.cpp` so the stages can be read top to bottom: validate input,
   prepare model tensors, encode observations/language, run the model, decode
   normalized actions, and return telemetry. Shared Policy functions perform
   resize, normalization, padding, and final action recovery.
6. Keep `module.cpp` small. Return one `ArchitectureDescriptor` containing the
   stable architecture id, capability declaration, and model factory.
7. Add the module to `src/models/builtin_modules.cpp`, create a separate CMake
   object target and feature option, then append its objects to `wam_core`.

No dynamic plugin ABI exists in 0.6. Built-in registration is intentional: it
keeps compiler/linker failures visible and avoids a second lifecycle contract.

## Tests required before registration

- Synthetic artifact success plus missing/wrong metadata and tensor failures.
- PolicySpec/contract geometry conflict tests.
- Input role, dtype, shape, and language-mode validation.
- Model/Session load, predict, reset, destruction, and deterministic-noise tests.
- Pipeline stage parity against an independent reference using a real GGUF.
- Final decoded action parity, not only normalized model output.
- CPU metadata build, CUDA build for compute modules, install consumer, C ABI,
  Python local, and WebSocket remote parity where supported.

Large GGUF and replay data remain external. Add grouped absolute `WAM_TEST_*`
cache variables and make an incomplete group fail configuration. A missing
group must produce an explicit skip message and cannot count as release parity.

Run the default checks while developing:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWAM_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The boundary test in `tests/contracts/source_boundaries.cmake` is the executable
definition of allowed dependency direction. Extend that test when adding a new
module so the new network tree cannot depend on artifact, Policy, serving, or a
different concrete model.
