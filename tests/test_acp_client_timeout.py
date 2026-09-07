"""Tests for AcpClient hard-timeout -> transport-unhealthy behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from diploid_agent.acp_client import AcpClient, AcpPromptResult


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> AcpClient:
    """Return a minimally initialized AcpClient with a fake devin binary."""
    monkeypatch.setenv("WINDSURF_API_KEY", "test-key")
    fake_devin = tmp_path / "devin"
    fake_devin.write_text("#!/bin/sh\n")
    fake_devin.chmod(0o755)
    c = AcpClient(
        model="swe-1-7",
        agent_bin=str(fake_devin),
        api_key="test-key",
        timeout=18120.0,
    )
    # Start un-initialized so _ensure_started runs _start_transport.
    c._transport_healthy = False
    c._transport._initialized = False
    return c


def _make_result(stop_reason: str | None, timed_out: bool = False) -> AcpPromptResult:
    return AcpPromptResult(
        reply="partial",
        session_id="s1",
        stop_reason=stop_reason,
        partial=stop_reason is not None or timed_out,
        timed_out=timed_out,
    )


def test_create_session_marks_transport_unhealthy_on_timeout(client, monkeypatch) -> None:
    """A hard timeout (stop_reason='timeout') makes the transport unhealthy."""
    monkeypatch.setattr(client, "_ensure_started", lambda *a, **k: None)
    monkeypatch.setattr(
        client, "_run", lambda coro, timeout=None, **kw: _make_result("timeout", True)
    )

    client.create_session("test prompt")

    assert client._transport_healthy is False
    assert client._transport._is_transport_healthy() is False


def test_send_message_marks_transport_unhealthy_on_timeout(client, monkeypatch) -> None:
    """Same for follow-up messages."""
    monkeypatch.setattr(client, "_ensure_started", lambda *a, **k: None)
    monkeypatch.setattr(
        client, "_run", lambda coro, timeout=None, **kw: _make_result("timeout", True)
    )

    client.send_message("s1", "continue")

    assert client._transport_healthy is False


def test_create_session_keeps_transport_healthy_on_cancelled(client, monkeypatch) -> None:
    """A soft cancel (stop_reason='cancelled') should not kill the transport."""
    monkeypatch.setattr(client, "_ensure_started", lambda *a, **k: None)
    client._transport_healthy = True
    # timed_out is True because _soft_timeout_canceller sets prompt.timed_out,
    # but the stop_reason is cancelled.
    monkeypatch.setattr(
        client, "_run", lambda coro, timeout=None, **kw: _make_result("cancelled", True)
    )

    client.create_session("test prompt")

    assert client._transport_healthy is True


def test_create_session_keeps_transport_healthy_on_completed(client, monkeypatch) -> None:
    """A completed result should not kill the transport."""
    monkeypatch.setattr(client, "_ensure_started", lambda *a, **k: None)
    client._transport_healthy = True
    monkeypatch.setattr(
        client, "_run", lambda coro, timeout=None, **kw: _make_result(None, False)
    )

    client.create_session("test prompt")

    assert client._transport_healthy is True


def test_is_transport_healthy_respects_flag(client) -> None:
    """_is_transport_healthy must return False when _transport_healthy is False."""
    client._transport_healthy = False
    assert client._transport._is_transport_healthy() is False


def test_create_session_restarts_transport_after_timeout(client, monkeypatch) -> None:
    """After a hard timeout, the next create_session must restart the child."""
    starts = []

    async def fake_start_transport(mcp_servers: Any = None) -> None:
        starts.append(1)
        client._transport._initialized = True
        client._transport._transport_healthy = True
        client._transport._loop = mock.Mock(is_closed=lambda: False, is_running=lambda: True)
        client._transport._proc = mock.Mock(returncode=None)

    monkeypatch.setattr(client._transport, "_start_transport", fake_start_transport)
    monkeypatch.setattr(client._transport, "_kill_process_group", lambda proc: None)

    results = iter(
        [
            _make_result("timeout", True),  # first turn hard times out
            _make_result(None, False),       # second turn completes
        ]
    )

    def fake_run(coro, timeout=None, **kw):
        name = getattr(coro, "__name__", None)
        if name == "_start_transport":
            starts.append(1)
            client._transport._initialized = True
            client._transport._transport_healthy = True
            client._transport._loop = mock.Mock(is_closed=lambda: False, is_running=lambda: True)
            client._transport._proc = mock.Mock(returncode=None)
            return None
        if name in ("_create_session", "_send_message"):
            return next(results)
        raise NotImplementedError(name)

    monkeypatch.setattr(client, "_run", fake_run)

    # First call: start transport, run, timeout -> mark unhealthy.
    client.create_session("first prompt", cwd=Path("/tmp"))
    assert client._transport_healthy is False

    # Second call: _ensure_started sees unhealthy, restarts, runs, completes.
    client.create_session("second prompt", cwd=Path("/tmp"))
    assert client._transport_healthy is True
    assert len(starts) == 2


class _FakeLoop:
    def __init__(self) -> None:
        self.running = True
        self.stop_calls = 0

    def is_running(self) -> bool:
        return self.running

    def is_closed(self) -> bool:
        return not self.running

    def call_soon_threadsafe(self, cb: Any, *args: Any) -> None:
        cb()

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


class _FakeThread:
    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        pass


class _FakeProc:
    def __init__(self) -> None:
        self.pid = 111
        self.returncode: int | None = None

    def kill(self) -> None:
        self.returncode = -9


class _LateFuture:
    """Completes only when told; models a close coroutine that finishes
    after close() already gave up waiting on it."""

    def __init__(self) -> None:
        self._callbacks: list[Any] = []

    def add_done_callback(self, cb: Any) -> None:
        self._callbacks.append(cb)

    def result(self, timeout: float | None = None) -> None:
        return None

    def complete(self) -> None:
        for cb in list(self._callbacks):
            cb(self)


def test_close_does_not_stop_or_clear_next_generation(client, monkeypatch) -> None:
    """Regression (2026-09-08 ghost generation): close() teardown must operate
    on the loop/thread/proc it captured at entry.

    The _close_transport done-callback fires on whichever thread completes the
    future -- potentially after a follow-on _ensure_started installed the next
    generation's loop.  Dereferencing ``self._loop`` at fire time stopped the
    NEW loop and parked its _start_transport at the first await for the full
    _run timeout; unconditional slot clears then rolled the replacements back.
    """
    client._initialized = True
    client._proc = _FakeProc()

    old_loop = _FakeLoop()
    new_loop = _FakeLoop()
    new_thread = _FakeThread()
    new_proc = _FakeProc()

    class _SwappingThread(_FakeThread):
        def join(self, timeout: float | None = None) -> None:
            # A follow-on _ensure_started installs the next generation's
            # handles while the old teardown is still finishing.
            client._loop = new_loop
            client._thread = new_thread
            client._proc = new_proc

    old_thread = _SwappingThread()
    client._loop = old_loop
    client._thread = old_thread

    future = _LateFuture()

    def fake_run_coroutine_threadsafe(coro: Any, loop: Any) -> Any:
        assert loop is old_loop
        coro.close()  # never awaited on a fake loop; silence the warning
        return future

    monkeypatch.setattr(
        asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe
    )
    monkeypatch.setattr(client._sandbox, "cleanup", lambda: None)
    monkeypatch.setattr(client._control, "close", lambda: None)

    client.close()

    # The replacement handles installed mid-teardown survive the slot clears.
    assert client._loop is new_loop
    assert client._thread is new_thread
    assert client._proc is new_proc

    # The abandoned close coroutine finishes later; its done-callback must
    # stop only the captured old loop, never the replacement.
    future.complete()
    assert old_loop.stop_calls == 1
    assert old_loop.running is False
    assert new_loop.stop_calls == 0
    assert new_loop.running is True

    # Leave no fake running loop for the atexit-registered close().
    client._loop = None
    client._thread = None
    client._proc = None
    client._initialized = False


def test_close_stops_captured_loop_on_timeout(client, monkeypatch) -> None:
    """Timeout path also uses the captured loop, not self._loop."""
    client._initialized = True
    old_loop = _FakeLoop()
    old_proc = _FakeProc()
    client._loop = old_loop
    client._thread = _FakeThread()
    client._proc = old_proc

    class _TimeoutFuture(_LateFuture):
        def result(self, timeout: float | None = None) -> None:
            raise TimeoutError()

    future = _TimeoutFuture()

    def fake_run_coroutine_threadsafe(coro: Any, loop: Any) -> Any:
        coro.close()
        return future

    monkeypatch.setattr(
        asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe
    )
    monkeypatch.setattr(client._sandbox, "cleanup", lambda: None)
    monkeypatch.setattr(client._control, "close", lambda: None)

    client.close()

    assert old_proc.returncode == -9  # captured proc killed
    assert old_loop.stop_calls == 1  # captured loop stopped via the except path
    assert client._loop is None
    assert client._thread is None
    assert client._proc is None
