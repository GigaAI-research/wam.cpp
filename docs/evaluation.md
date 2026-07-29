# Evaluation

Evaluation is an application layer above the public `wam` API. It owns simulator
lifecycle, deterministic episode selection, action execution policy, metrics,
videos, and result persistence. The runtime does not know environment names or
hold an action queue.

## Shared components

`wam.eval.ActionChunkExecutor` truncates each prediction to `execute_steps`,
applies one adapter action transform, queues commands, and stops on Gym or
Gymnasium termination. `reset()` clears stale actions between episodes.

`wam.eval` also provides canonical JSON manifest helpers, latency distributions,
`Metrics`, atomic JSON/JSONL writers, append-only `ResultWriter`, and an optional
`VideoWriter`. Video support imports `imageio` only when used.

A runner should have this visible flow:

```text
load frozen manifest
  -> reset simulator and Client
  -> adapter.observation(raw)
  -> Client.predict(...)
  -> ActionChunkExecutor
  -> adapter.action(chunk)
  -> simulator.step(command)
  -> record request and episode
```

The environment-specific runner owns success semantics and upstream imports.
It must not reimplement protocol envelopes, language encoding, PolicySpec image
processing, normalization, action recovery, or architecture selection.

## Reproducibility

A formal run freezes model/profile identity, simulator revision, task and init
state ordering, environment seed, action-noise seed, wait/execute/max steps,
camera geometry, and controller conventions. Write each completed request and
episode before starting the next one so interrupted runs retain evidence.

Resume logic must compare the exact manifest and model identity, reject
overlapping shard selections, and skip only completed episodes. Report success
rate separately from integration status and include latency distribution rather
than a single mean.

## Acceptance levels

1. Adapter contract test: fixed arrays verify roles, fields, shapes, and action
   convention without a simulator.
2. Synthetic episode: fixed seed verifies adapter, queue, step, metrics, and
   persistence wiring. It does not prove simulator or model correctness.
3. Real smoke: one real checkpoint completes one pinned simulator episode
   without protocol/runtime failure.
4. Numerical parity: the same observation, language input, noise, and seed match
   an independent model reference before closed-loop rollout comparison.
5. Benchmark: a frozen multi-episode manifest reports success, failures,
   request counts, latency groups, and hardware/software context.

Current runner commands and frozen evidence remain under `eval/sim/`. Published
readiness and benchmark summaries are indexed in
[Support Matrix](support-matrix.md). Large checkpoints, simulator checkouts,
videos, replay tensors, and raw results stay outside Git.
