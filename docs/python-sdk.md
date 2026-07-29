# Python SDK

Install the SDK from the repository and point it at the installed C ABI library:

```bash
python -m pip install .
export WAM_C_API_LIBRARY=/path/to/libwam_c_api.so
```

The shortest local path is a context-managed pipeline. A bundle directory may
declare `tokenizer` and `language_encoder` in `manifest.json`; a GGUF deployment
can pass those paths explicitly to `Pipeline.load`.

```python
import wam

with wam.Pipeline.load("policy-bundle") as pipeline:
    prediction = pipeline.predict(images, state, instruction="pick up the cup")
    action_chunk = prediction.action
```

Prepared token or BF16 embedding arrays can be passed instead of `instruction`.
The SDK never downloads language resources. Missing resources raise
`LanguageResourceError`; native failures raise `WamError` with a stable `code`
and field-level `details`.

`wam-predict MODEL INPUT.npz --instruction "..."` provides the same local path.
The NPZ must contain `state` and one or more `image.<role>` HWC U8 arrays. It may
instead contain `token_ids` plus `attention_mask`, or `embedding` plus
`embedding_attention_mask`.

The remote `wam.Client` uses `wam.rpc.v06`. Remote failures raise `RemoteError`
with stable RPC code, field-level details, and a fatal flag.

Bundle layout and resource precedence are specified in
[Artifact Bundle](artifact-bundle.md). Server startup, handshake, recovery, and
production boundaries are documented in [Serving](serving.md). Runnable local
and remote functions are under `examples/python/`.

## C ABI ownership

Call every `wam_c_*_init` initializer before passing an input structure. Input
buffers are borrowed for the duration of a call. Model/session handles, metadata
strings, predictions, and errors returned by the library belong to the caller and
must be released with their matching `wam_c_*_free` function. Every free function
accepts `NULL`; allocations must not be released through another allocator.
