"""Redis fixtures for API state tests."""

from __future__ import annotations

import json
import subprocess
import threading
import typing as t
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import redis
from upstash_redis.asyncio import Redis as UpstashRedis


@pytest.fixture
def anyio_backend() -> str:
    """Select asyncio for the Upstash SDK.

    Returns:
        The AnyIO backend name.
    """
    return "asyncio"


@pytest.fixture
def redis_client(tmp_path: Path) -> t.Iterator[redis.Redis]:
    """Provide a synchronous client connected to an ephemeral Redis server.

    Args:
        tmp_path:
            Pytest temporary directory.

    Yields:
            A Redis client backed by a socket-local server.

    Raises:
            RuntimeError:
                If the local Redis server fails to start.
    """
    socket_path = tmp_path / "redis.sock"
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
            try:
                client.ping()
                break
            except redis.ConnectionError:
                threading.Event().wait(0.05)
        else:
            raise RuntimeError("Redis server did not start")
        yield client
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=5)


@pytest.fixture
def upstash_client(redis_client: redis.Redis) -> t.Iterator[UpstashRedis]:
    """Provide the official async SDK over a local Redis-command REST adapter.

    Args:
        redis_client: Local synchronous Redis client.

    Yields:
        An official Upstash Redis SDK client.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            """Execute the submitted command against local Redis."""
            length = int(self.headers["Content-Length"])
            command = json.loads(self.rfile.read(length))
            try:
                result = redis_client.execute_command(*command)
                if isinstance(result, set):
                    result = sorted(result)
                elif isinstance(result, bool):
                    result = int(result)
                elif command[0].upper() == "SET" and result:
                    result = "OK"
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
            """Suppress fixture HTTP logging."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = UpstashRedis(
        url=f"http://127.0.0.1:{server.server_port}",
        token="fixture-token",
        rest_encoding=None,
    )
    try:
        yield client
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
