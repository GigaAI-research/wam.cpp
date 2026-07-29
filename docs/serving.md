# Serving

wam.cpp 0.6 serves policy inference as binary Protobuf messages over WebSocket.
The wire package is `wam.rpc.v06`; v0.5 clients are intentionally incompatible.

## Start a server

Build with `WAM_BUILD_SERVING=ON` and install the Python SDK. CMake generates
`wam.desc` when `protoc` is available.

```bash
wam-serve --library /path/libwam_c_api.so --descriptor /path/wam.desc \
  --model /path/policy-bundle --environment robotwin \
  --backend cuda --precision bf16 --host 0.0.0.0 --port 18060
```

The CLI discovers architecture and PolicySpec from the artifact. Do not pass
camera count, state/action dimensions, normalization, or action recovery. The
selected EnvironmentContract is checked before Session creation. Language
resources come from explicit CLI options or the bundle; nothing is downloaded.

## Client lifecycle

```python
import wam

with wam.Client("server", 18060, "wam.desc", "robotwin") as client:
    model_info = client.model_info
    action, stats = client.predict(images, state, "pick up the cup")
    client.reset()
```

Each connection starts with request id 0 and an exact protocol/environment
handshake. Later request ids are contiguous. `close()` releases the server
Session; context managers close both normal and exceptional paths.

`RemoteError` exposes `code`, `details`, and `fatal`. Shape/input failures are
normally recoverable. Protocol, artifact compatibility, failed preconditions,
and internal failures close the connection. Callers should fix a recoverable
request or reset; after a fatal error they must create a new Client.

## Internal boundary

`ServiceCore` and `ServiceConnection` own protocol state, Session lifecycle,
language preparation, error mapping, and inference concurrency. They do not
open sockets. `WebSocketTransport` owns binary frame enforcement, message-size
limits, serialization, and close codes. This separation allows deterministic
core tests without binding a port.

When `ModelInfo.capabilities.concurrent_sessions` is false, one async lock
serializes inference across connections while leaving socket handling
concurrent. A model that declares concurrent sessions is responsible for its
backend/resource safety.

The protocol has no authentication, TLS termination, admission control,
multi-tenant isolation, or automatic restart. Production deployments should
place it behind an authenticated TLS proxy/process supervisor and restrict
network access. gRPC is not part of 0.6.
