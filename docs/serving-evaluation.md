# Serving and Evaluation

The installed `wam-serve` command starts the generic v0.6 WebSocket service:

```bash
wam-serve --library /path/libwam_c_api.so --descriptor /path/wam.desc \
  --model policy-bundle --environment robotwin --backend cuda --precision bf16
```

`ServiceCore` owns protocol state, Model Session lifecycle, error mapping and
the model concurrency gate. `WebSocketTransport` owns only binary frames,
message limits, Protobuf serialization and WebSocket close codes. Tests can
call `ServiceConnection.handle()` without opening a socket.

The wire protocol is `wam.rpc.v06`. A connection begins with request 0 and an
exact protocol/environment handshake. Subsequent request ids are contiguous.
Invalid predict payloads may be recoverable; protocol, compatibility and
internal failures are fatal. Errors contain a stable code and repeated
field/reason details.

Formal environment adapters live in `wam.adapters`:

- `RoboTwinAdapter`: three named cameras and the frozen dual-arm 14D contract.
- `LiberoAdapter`: scene/wrist images, EEF state conversion and 7D controller command.
- `LiberoXAdapter`: OpenPI observation keys and LIBERO-X gripper convention.

Adapters translate simulator semantics only. Checkpoint resize, normalization,
padding and action recovery remain in the C++ Policy layer.

`wam.eval` provides `ActionChunkExecutor`, canonical manifest helpers, latency
metrics, atomic JSON/JSONL result writers and an optional video writer. The
environment runner continues to own reset, step, done and success semantics.
