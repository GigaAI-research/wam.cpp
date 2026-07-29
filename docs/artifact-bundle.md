# Artifact Bundle

A wam artifact is either one GGUF file or a directory containing a strict
`manifest.json`. Use a bundle when raw instructions require tokenizer or
external language-encoder resources.

```text
policy-bundle/
├── manifest.json
├── model.gguf
├── tokenizer/
└── language_encoder.safetensors
```

The v1 manifest is a flat JSON object:

```json
{
  "format": "wam-bundle-v1",
  "manifest_schema_version": 1,
  "model": "model.gguf",
  "tokenizer": "tokenizer/tokenizer.json",
  "language_encoder": "language_encoder.safetensors"
}
```

`format`, `manifest_schema_version`, and `model` are required. `tokenizer` and
`language_encoder` may be omitted or `null`. Unknown fields are rejected so a
misspelled deployment key cannot silently change behavior.

All resource paths are relative to the bundle root. Absolute paths, `..`,
missing files, symlinks that resolve outside the root, and duplicate JSON keys
are rejected. The runtime never downloads missing resources. Explicit Python
or CLI resource arguments take precedence over manifest language resources;
the GGUF model path always comes from the manifest.

The GGUF contains the architecture contract and schema-v3 PolicySpec. Model
geometry, required tensor names/dtypes, image transforms, state normalization,
language contract, action normalization, and action recovery are therefore
validated before prediction. The bundle manifest is deployment composition,
not a second PolicySpec.

Use the installed tools before deploying:

```bash
wam-inspect policy-bundle
wam-validate policy-bundle
```

`wam-inspect` performs metadata inspection without allocating weights.
`wam-validate` selects the architecture module and validates its full artifact
contract using the CPU metadata backend. Both emit one JSON object and return a
nonzero status with stable code/message/details on failure.

Bundle v1 deliberately has no automatic download, arbitrary nested metadata,
environment selection, runtime tuning, or integrity policy. Deployment systems
may keep their own provenance and content-addressing records outside this
minimal runtime manifest.
