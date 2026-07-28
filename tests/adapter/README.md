# Adapter tests

Adapter tests cover environment-to-canonical observation and action mappings
without loading model internals. The existing LIBERO-X adapter check remains in
`tests/rpc` during Phase 1 because its current fixture exercises the complete
remote protocol path. New isolated adapter tests belong in this directory.
