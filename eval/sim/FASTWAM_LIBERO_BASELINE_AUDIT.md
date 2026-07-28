# FastWAM LIBERO Baseline Audit

Audit date: 2026-07-28

## Frozen contract

The official FastWAM donor and wam.cpp were evaluated with the released
`libero_uncond_2cam224.pt` checkpoint and its matching min-max dataset
statistics. Both paths used:

- LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`;
- MuJoCo 3.3.2 and robosuite 1.4.0;
- all 40 tasks from `libero_spatial`, `libero_object`, `libero_goal`, and
  `libero_10`;
- the same 50 ordered init states per task, for 2,000 episodes total;
- environment seed 42 once per task;
- action-noise seed 42 recreated for every predict as CPU F32 noise and then
  rounded to BF16;
- 30 initial wait steps, 10 executed actions per replan, and at most 400
  controller steps.

The four wam.cpp manifests have these SHA-256 identities:

| Suite | Manifest SHA-256 |
| --- | --- |
| `libero_spatial` | `ec8cd770e1661b758697ecf59f8aaf898ae4782197e44f9a90c3189727df592c` |
| `libero_object` | `0ca84bfccd641a8c9fe52b33ac924c15e7eaec05690cfc8a6951c903ddb0b595` |
| `libero_goal` | `229b02d01b1b0e30c2eb063c28f47d13e70bba0f7fa279ec0a0f731209cd68a6` |
| `libero_10` | `2b81ea6dc3fa79eadaa4233f0ebbdeb07c9c50364069d4315afbb6d0b1cd696c` |

## Result

| Suite | Donor Python | wam.cpp | Delta | Exact task outcome sets |
| --- | ---: | ---: | ---: | ---: |
| `libero_spatial` | 485/500 | 488/500 | +3 | 4/10 |
| `libero_object` | 500/500 | 496/500 | -4 | 6/10 |
| `libero_goal` | 479/500 | 481/500 | +2 | 3/10 |
| `libero_10` | 471/500 | 473/500 | +2 | 3/10 |
| **Total** | **1935/2000 (96.75%)** | **1938/2000 (96.90%)** | **+3** | **16/40** |

The aggregate success-rate difference is 0.15 percentage points. Exact task
outcome sets require every one of a task's 50 init states to have the same
boolean result; 16 of 40 tasks meet that stricter condition. Small action
differences can move marginal rollouts across the binary success boundary, so
the remaining per-state differences do not by themselves contradict the
independent same-input numerical action gate.

wam.cpp issued 32,058 model requests. The operational latency measurements
were:

| Suite | Requests | RPC mean | Server total mean | Model mean |
| --- | ---: | ---: | ---: | ---: |
| `libero_spatial` | 5,605 | 649.20 ms | 579.63 ms | 569.79 ms |
| `libero_object` | 6,675 | 1233.01 ms | 918.68 ms | 906.99 ms |
| `libero_goal` | 5,982 | 1156.20 ms | 750.81 ms | 740.45 ms |
| `libero_10` | 13,796 | 1047.46 ms | 1043.62 ms | 1032.67 ms |

These are end-to-end operational measurements, not isolated kernel
benchmarks. The simulator and server shared host GPU resources during the long
run, and the donor does not emit equivalent per-request phase timings.

## Verdict

FastWAM LIBERO passes the 0.5 simulator quality-parity gate. This conclusion
combines the previously frozen same-input numerical action gate with the
2,000-episode result above; it does not require every stochastic closed-loop
episode to have an identical boolean outcome.

Raw manifests, episode records, request timings, videos, checkpoints, GGUF
files, and the generated JSON comparison remain outside Git. They are local
evaluation artifacts and can be regenerated with the commands in
`FASTWAM_LIBERO_REMOTE_EVAL.md`.
