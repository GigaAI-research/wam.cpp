# wam.cpp 0.5 Phase 1 Skeleton

This directory currently contains declaration-only C++ and evaluation interfaces for
architecture review. Function bodies, inference behavior, GGUF schema parsing, protocol
changes, simulator checkout setup, and simulator integration are intentionally absent.

`WAM_PHASE1_DECLARATIONS_ONLY=1` identifies the CMake target as non-runnable. The source
files include their corresponding headers so strict syntax checks can validate dependency
direction and header completeness without creating placeholder inference behavior.

`eval/sim/` contains fail-fast client/server/setup entry-point declarations. The ignored
`RoboTwin/`, `LIBERO/`, and `LIBERO-X/` directories will hold revision-pinned upstream
checkouts once their setup scripts are implemented.
