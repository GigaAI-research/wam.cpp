# Adding an Environment

An environment extension translates simulator observations and controller
commands. It does not select model architecture, resize checkpoint images,
normalize state, recover actions, or implement RPC.

## Define the contract

Add `python/wam/adapters/<environment>.py` with one frozen
`EnvironmentContract` and one `Adapter` implementation. The contract contains:

- environment id;
- canonical image roles;
- ordered state and action field names;
- real state and action dimensions;
- action representation and frame;
- controller gripper encoding.

`check_environment(model.metadata, contract)` compares these fields with the
artifact PolicySpec. The generic server calls it before Session creation, so an
incompatible checkpoint fails before the first episode.

Implement `observation(raw, policy_spec)` to return named HWC RGB U8 arrays and
one contiguous F32 state vector. Implement `action(chunk, policy_spec)` to map a
rank-2 decoded action chunk into the controller convention. Validate source keys,
rank, shape, dtype, and field order at this boundary.

Register the adapter in `python/wam/adapters/__init__.py`. Registration must not
contain model ids or choose behavior from `metadata["architecture"]`.

## Connect evaluation

Keep simulator lifecycle in `eval/sim/run_<environment>_client.py`: upstream
import/setup, reset, raw observation acquisition, step, done/success, and close.
Use the formal components instead of copying them:

```python
from wam.adapters import create_adapter
from wam.eval import ActionChunkExecutor, Metrics, ResultWriter

adapter = create_adapter("example")
executor = ActionChunkExecutor(
    execute_steps=10,
    action_transform=lambda chunk: adapter.action(chunk, policy_spec))
```

Add a small `run_<environment>_server.py` only when a fixed environment command
is useful. It should pass the adapter contract to `wam.serving.cli.serve_main`;
it must not duplicate server arguments, protocol handling, or compatibility
logic. The generic alternative is:

```bash
wam-serve ... --environment example
```

## Acceptance tests

- Contract mismatch fails before Session creation.
- Fixed synthetic observation maps to exact canonical roles and state fields.
- Fixed action chunk maps to exact controller commands, including gripper signs.
- `ActionChunkExecutor` covers horizon truncation, reset, and early termination.
- One fixed-seed synthetic episode runs without importing model internals.
- One real simulator smoke episode and then a frozen manifest evaluation run.
- Source-boundary tests reject duplicated conversion functions, server contracts,
  and architecture branches in the runner/adapter.

Real simulator success is an external release gate. A synthetic adapter test
proves interface behavior only and must not be reported as model quality or
simulator readiness. ROS2 should be added as a separate environment only after
a real robot contract and test assets exist.
