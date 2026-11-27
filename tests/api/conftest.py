"""Real Redis and REST fixtures for the native API's public workflows."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import typing as t
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import redis
from upstash_redis.asyncio import Redis as UpstashRedis


@pytest.fixture
def anyio_backend() -> str:
    """Select the asyncio backend used by the SDK.

    Returns:
        Backend name.
    """
    return "asyncio"


@pytest.fixture
def redis_client() -> t.Iterator[redis.Redis]:
    """Run Redis on a short Unix socket rather than macOS's long pytest temp path.

    Yields:
        Official client connected to an isolated server.

    Raises:
        RuntimeError: If Redis exits or cannot become ready.
    """
    with tempfile.TemporaryDirectory(prefix="ee-redis-", dir="/tmp") as directory:
        socket_path = Path(directory) / "redis.sock"
        process = subprocess.Popen(
            [
                "redis-server",
                "--port",
                "0",
                "--unixsocket",
                str(socket_path),
                "--unixsocketperm",
                "700",
                "--save",
                "",
                "--appendonly",
                "no",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = redis.Redis(unix_socket_path=str(socket_path), decode_responses=True)
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Redis exited before creating its test socket")
                if socket_path.exists():
                    client.ping()
                    break
                threading.Event().wait(0.02)
            else:
                raise RuntimeError("Redis test socket did not become ready")
            yield client
        finally:
            client.close()
            process.terminate()
            process.wait(timeout=5)


@pytest.fixture
async def upstash_client(redis_client: redis.Redis) -> t.AsyncIterator[UpstashRedis]:
    """Connect the official async SDK to real Redis through its REST contract.

    Args:
        redis_client: Official synchronous client for the isolated server.

    Yields:
        Official SDK using plain-JSON REST responses.
    """

    class Handler(BaseHTTPRequestHandler):
        """Translate REST command arrays without emulating Redis operations."""

        def do_POST(self) -> None:
            """Execute the actual Redis command and return its REST representation."""
            length = int(self.headers["Content-Length"])
            command = json.loads(self.rfile.read(length))
            try:
                result = redis_client.execute_command(*command)
                if isinstance(result, set):
                    result = list(result)
                elif command[0].upper() == "SET" and result is True:
                    result = "OK"
                elif isinstance(result, bool):
                    result = int(result)
                payload = {"result": result}
            except redis.ResponseError as error:
                payload = {"error": str(error)}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress local fixture access logging.

            Args:
                format: HTTP server log template.
                args: Template arguments.
            """

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = UpstashRedis(
        url=f"http://127.0.0.1:{server.server_port}",
        token="fixture-token",
        rest_encoding=None,
        rest_retries=0,
    )
    try:
        yield client
    finally:
        await client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
