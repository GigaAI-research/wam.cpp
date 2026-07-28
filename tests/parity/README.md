# Parity tests

Numerical parity gates compare wam.cpp outputs with frozen, independent model
references. During Phase 1 the existing gated binaries remain under
`tests/integration` so their source and numerical behavior do not move together
with the build refactor. New parity tests belong in this directory.
