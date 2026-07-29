from __future__ import annotations

import logging

LOG = logging.getLogger("wam.websocket")


class WebSocketTransport:
    def __init__(self, service, host="0.0.0.0", port=18060,
                 max_message_bytes=64 << 20):
        self.service = service
        self.host = host
        self.port = int(port)
        self.max_message_bytes = int(max_message_bytes)

    async def handler(self, websocket):
        connection = self.service.connection()
        try:
            while not connection.closed:
                payload = await websocket.recv()
                if isinstance(payload, str):
                    response = connection.error(
                        connection.expected_request_id, 1,
                        "text WebSocket frames are not accepted",
                        field="frame", fatal=True)
                elif len(payload) > self.max_message_bytes:
                    response = connection.error(
                        connection.expected_request_id, 1,
                        "WebSocket message exceeds configured limit",
                        field="frame", fatal=True)
                else:
                    request = self.service.types["ClientEnvelope"]()
                    try:
                        request.ParseFromString(payload)
                    except Exception as error:
                        response = connection.error(
                            connection.expected_request_id, 1,
                            f"invalid protobuf: {error}", field="envelope",
                            fatal=True)
                    else:
                        response = await connection.handle(request)
                await websocket.send(response.SerializeToString())
            if response.WhichOneof("payload") == "error" and response.error.fatal:
                await websocket.close(code=1002)
        except Exception as error:
            if "ConnectionClosed" not in type(error).__name__:
                LOG.exception("WebSocket connection failed")
        finally:
            connection.close()

    async def serve(self):
        from websockets.asyncio.server import serve
        async with serve(
                self.handler, self.host, self.port, compression=None,
                max_size=self.max_message_bytes, ping_interval=120,
                ping_timeout=600) as server:
            LOG.info("ready ws://%s:%d environment=%s architecture=%s",
                     self.host, self.port,
                     self.service.contract.environment_id,
                     self.service.model.metadata["architecture"])
            await server.serve_forever()
