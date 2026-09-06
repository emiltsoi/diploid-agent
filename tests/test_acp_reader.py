"""Regression tests for the ACP stdout reader.

A `devin acp` child can emit single JSON-RPC lines far larger than the asyncio
default 64 KiB stream limit (e.g. a ``tool_call_update`` carrying raw tool
output).  ``StreamReader.readline()`` then raises ``ValueError`` and the reader
task exits silently: the pipe fills, the child blocks on write, and in-flight
prompts hang until their multi-hour outer timeouts.  These tests drive a real
subprocess so the failure mode is exercised end to end.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from diploid_agent.acp_client import AcpClient, AcpTransportError
from diploid_agent.acp_client.transport import AcpTransport

# Minimal JSON-RPC server speaking just enough ACP for the client handshake
# and a prompt that emits a configurable oversized or malformed line.
_FAKE_ACP = r"""
import json
import os
import sys

BIG_LINE = int(os.environ.get("FAKE_ACP_BIG_LINE", "0"))
GARBAGE = os.environ.get("FAKE_ACP_GARBAGE", "0") == "1"


def _send(obj):
    sys.stdout.buffer.write(json.dumps(obj).encode() + b"\n")
    sys.stdout.buffer.flush()


def _send_raw(data):
    sys.stdout.buffer.write(data + b"\n")
    sys.stdout.buffer.flush()


for raw in sys.stdin.buffer:
    try:
        msg = json.loads(raw)
    except Exception:
        continue
    method = msg.get("method")
    mid = msg.get("id")
    if mid is None:
        continue  # notification (e.g. session/cancel); nothing to answer
    if method == "initialize":
        _send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": 1,
            "agentInfo": {"title": "fake-acp", "version": "0.0.1"},
            "authMethods": [],
        }})
    elif method == "session/new":
        _send({"jsonrpc": "2.0", "id": mid,
               "result": {"sessionId": "s-1", "configOptions": []}})
    elif method == "session/prompt":
        sid = msg.get("params", {}).get("sessionId", "s-1")
        if GARBAGE:
            _send_raw(b"\x80\x81\x82 not valid utf-8 or json \xff")
        if BIG_LINE:
            _send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t-1",
                    "rawOutput": "x" * BIG_LINE,
                },
            }})
        _send({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": sid,
            "update": {"sessionUpdate": "agent_message_chunk",
                       "content": {"type": "text", "text": "ok"}},
        }})
        _send({"jsonrpc": "2.0", "id": mid, "result": {"stopReason": "end_turn"}})
    else:
        _send({"jsonrpc": "2.0", "id": mid, "result": {}})
"""


@pytest.fixture
def fake_acp(tmp_path: Path) -> Path:
    script = tmp_path / "fake_acp.py"
    script.write_text(_FAKE_ACP)
    return script


def _make_client(fake_acp: Path) -> AcpClient:
    return AcpClient(
        model="swe-1-7",
        agent_bin=sys.executable,
        start_args=[str(fake_acp)],
        api_key="test-key",
        timeout=30.0,
        startup_timeout=15.0,
        control_timeout=10.0,
        watchdog_interval=3600.0,
        watchdog_timeout=3600.0,
    )


def test_oversized_update_line_is_processed(
    fake_acp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A >64 KiB JSON-RPC line must not kill the stdout reader.

    Regression test: with the asyncio default limit this line raised
    ``ValueError`` in ``readline()``, the reader exited, and the in-flight
    prompt hung until its outer timeout while the child blocked on write.
    """
    monkeypatch.setenv("FAKE_ACP_BIG_LINE", str(200 * 1024))
    client = _make_client(fake_acp)
    try:
        updates: list[dict[str, Any]] = []
        result = client.send_message("s-1", "hi", on_update=updates.append)
        assert result.reply == "ok"
        assert result.stop_reason == "end_turn"
        assert any(u.get("rawOutput") and len(u["rawOutput"]) == 200 * 1024 for u in updates)
        assert client.health()
    finally:
        client.close()


def test_reader_death_fails_inflight_fast(
    fake_acp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the reader still dies, in-flight calls must fail fast, not hang.

    Shrink the stream limit so a moderately sized line kills the reader; the
    done-callback must mark the transport unhealthy and unblock the caller.
    """
    monkeypatch.setenv("FAKE_ACP_BIG_LINE", str(32 * 1024))
    client = _make_client(fake_acp)
    # Force the stream limit below the line the fake server will emit.
    client._transport._stream_limit = 8 * 1024
    try:
        start = time.monotonic()
        with pytest.raises(AcpTransportError):
            client.send_message("s-1", "hi", timeout=30.0)
        assert time.monotonic() - start < 15.0
        assert not client.health()
        assert client._transport_healthy is False
    finally:
        client.close()


def test_malformed_line_is_skipped(
    fake_acp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-UTF-8 line must be skipped, not kill the reader."""
    monkeypatch.setenv("FAKE_ACP_GARBAGE", "1")
    client = _make_client(fake_acp)
    try:
        result = client.send_message("s-1", "hi")
        assert result.reply == "ok"
        assert result.stop_reason == "end_turn"
        assert client.health()
    finally:
        client.close()


class _DoneTask:
    """Minimal stand-in for a finished asyncio task."""

    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc

    def cancelled(self) -> bool:
        return False

    def exception(self) -> BaseException | None:
        return self._exc


class _CancelledTask(_DoneTask):
    def cancelled(self) -> bool:
        return True


class _FakeProc:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.pid = 12345
        self.stderr = object()


class _FakeClient:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_prompts: dict[str, Any] = {}
        self._loop: Any = None


def test_on_reader_done_marks_transport_unhealthy() -> None:
    """A reader that ends while the child is alive must fail the transport."""
    client = _FakeClient()
    transport = AcpTransport(client)
    transport._proc = _FakeProc()
    transport._initialized = True
    transport._transport_healthy = True
    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    transport._inflight_future = inflight
    task = _DoneTask()
    transport._reader_task = task  # type: ignore[assignment]

    transport._on_reader_done(task)  # type: ignore[arg-type]

    assert transport._transport_healthy is False
    assert inflight.done()
    assert isinstance(inflight.exception(), TimeoutError)


def test_on_reader_done_ignores_cancelled_and_dead_child() -> None:
    """Normal shutdown (cancelled task or dead child) is not a failure."""
    client = _FakeClient()
    transport = AcpTransport(client)
    transport._proc = _FakeProc()
    transport._initialized = True
    transport._transport_healthy = True

    cancelled = _CancelledTask()
    transport._reader_task = cancelled  # type: ignore[assignment]
    transport._on_reader_done(cancelled)  # type: ignore[arg-type]
    assert transport._transport_healthy is True

    done = _DoneTask()
    transport._reader_task = done  # type: ignore[assignment]
    transport._proc.returncode = -15  # child already exited: EOF is expected
    transport._on_reader_done(done)  # type: ignore[arg-type]
    assert transport._transport_healthy is True


class _FakeProcForStart:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.pid = 12345

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return 0


def test_concurrent_ensure_started_runs_one_start(monkeypatch) -> None:
    """Concurrent _ensure_started calls must serialize, not double-start.

    Without lifecycle serialization, two racing callers each create a new
    event loop and spawn a child; a task scheduled on one loop then awaits
    futures on the other ("attached to a different loop") and the extra
    subprocess is leaked.
    """
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
    starts: list[bool] = []

    async def fake_start() -> None:
        starts.append(True)
        await asyncio.sleep(0.3)  # hold the start open so the race can trigger
        client._proc = _FakeProcForStart()  # type: ignore[assignment]

    monkeypatch.setattr(client, "_start_transport", fake_start)
    monkeypatch.setattr(client, "_close_transport", _noop_close)

    errors: list[BaseException] = []

    def run() -> None:
        try:
            client._ensure_started()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=run)
    t2 = threading.Thread(target=run)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    try:
        assert not errors
        assert len(starts) == 1
        assert client._initialized
    finally:
        client.close()


async def _noop_close() -> None:
    return None


def test_send_message_smoke(fake_acp: Path) -> None:
    """Happy path: initialize, prompt, chunked reply, prompt response."""
    client = _make_client(fake_acp)
    try:
        result = client.send_message("s-1", "hi")
        assert result.reply == "ok"
        assert result.stop_reason == "end_turn"
        assert client.health()
    finally:
        client.close()
