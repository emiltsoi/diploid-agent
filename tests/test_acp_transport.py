"""Focused unit tests for AcpTransport."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from typing import Any

import pytest

from diploid_agent.acp_client.errors import AcpTransportError
from diploid_agent.acp_client.lifecycle import AcpRestartHistory
from diploid_agent.acp_client.transport import AcpTransport


class FakeMetrics:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {}

    def inc(self, name: str, *, reason: str | None = None) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1


class FakeSandbox:
    def __init__(self) -> None:
        self.prepared: list[Any] = []
        self.devin_home: Any = None

    def prepare(self, mcp_servers: Any) -> None:
        self.prepared.append(mcp_servers)


class FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.pid = 12345

    def kill(self) -> None:
        pass

    def terminate(self) -> None:
        pass


class FakeWatchdog:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FakeClient:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.metrics = FakeMetrics()
        self._sandbox = FakeSandbox()
        self._active_prompts: dict[str, Any] = {}
        self._pending_cancels: set[str] = set()
        self._proc: Any = None
        self._loop: Any = None
        self._max_restarts = 3
        self._restart_backoff_window = 300.0
        self._restart_history_store = AcpRestartHistory(None, 300.0)
        self._api_key = "test"
        self._startup_timeout = 30.0
        self.agent_bin = "/fake/devin"
        self.start_args: list[str] = []
        self._watchdog = FakeWatchdog()

    def _check_restart_backoff(self) -> None:
        self._restart_history_store.check(self._max_restarts)

    def _record_restart_attempt(self) -> None:
        self._restart_history_store.record()


def _make_transport() -> AcpTransport:
    client = FakeClient()
    return AcpTransport(client)


def test_healthy_when_not_initialized() -> None:
    transport = _make_transport()
    assert transport.healthy() is False


def test_healthy_with_running_transport() -> None:
    transport = _make_transport()
    transport._initialized = True
    transport._transport_healthy = True
    transport._proc = FakeProcess(returncode=None)
    assert transport.healthy() is True


def test_healthy_when_process_exited() -> None:
    transport = _make_transport()
    transport._initialized = True
    transport._transport_healthy = True
    transport._proc = FakeProcess(returncode=1)
    assert transport.healthy() is False


def test_healthy_with_inflight_future_within_deadline() -> None:
    transport = _make_transport()
    transport._initialized = True
    transport._proc = FakeProcess(returncode=None)
    transport._inflight_future = concurrent.futures.Future()
    transport._inflight_deadline = time.monotonic() + 60.0
    assert transport.healthy() is True


def test_healthy_uses_transport_healthy_when_no_inflight() -> None:
    transport = _make_transport()
    transport._initialized = True
    transport._proc = FakeProcess(returncode=None)
    transport._transport_healthy = True
    assert transport.healthy() is True


def test_is_transport_healthy_false_when_not_initialized() -> None:
    transport = _make_transport()
    assert transport._is_transport_healthy() is False


def test_record_and_check_restart_backoff() -> None:
    client = FakeClient()
    client._max_restarts = 2
    client._restart_backoff_window = 10.0
    transport = AcpTransport(client)
    transport._record_restart_attempt()
    transport._record_restart_attempt()
    with pytest.raises(AcpTransportError):
        transport._check_restart_backoff()


def test_unblock_inflight_sets_exception_and_aborts_prompts() -> None:
    transport = _make_transport()
    future = concurrent.futures.Future()
    transport._inflight_future = future
    transport._pending[1] = asyncio.Future()
    prompt: Any = type(
        "Prompt", (), {"cancelled": False, "timed_out": False, "cancel_done": asyncio.Future()}
    )()
    transport._client._active_prompts["s-1"] = prompt

    transport._unblock_inflight("test reason")

    assert future.done()
    assert future.exception() is not None
    assert transport._pending == {}
    assert prompt.cancelled is True
    assert prompt.timed_out is True
    assert prompt.cancel_done.done()


def test_unblock_inflight_clears_pending() -> None:
    transport = _make_transport()
    transport._pending[1] = asyncio.Future()
    transport._unblock_inflight("test reason")
    assert transport._pending == {}


def test_kill_process_group_swallows_errors() -> None:
    transport = _make_transport()
    proc = FakeProcess()
    # Should not raise even though this is a fake process.
    transport._kill_process_group(proc)


def test_route_update_routes_to_active_prompt() -> None:
    transport = _make_transport()
    chunks: list[str] = []
    updates: list[dict[str, Any]] = []
    prompt: Any = type(
        "Prompt",
        (),
        {
            "session_id": "s-1",
            "chunks": chunks,
            "updates": updates,
            "on_chunk": lambda text: chunks.append(text),
            "on_update": lambda update: updates.append(update),
        },
    )()
    prompt.chunks = chunks
    prompt.updates = updates
    transport._client._active_prompts["s-1"] = prompt

    msg = {
        "method": "session/update",
        "params": {
            "sessionId": "s-1",
            "update": {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "hello"},
            },
        },
    }
    transport._route_update(msg)
    assert "hello" in prompt.chunks
    assert len(prompt.updates) == 1


def test_route_update_falls_back_to_single_active_prompt() -> None:
    transport = _make_transport()
    prompt = type(
        "Prompt",
        (),
        {"session_id": "s-1", "chunks": [], "updates": [], "on_chunk": None, "on_update": None},
    )()
    prompt.chunks = []
    prompt.updates = []
    transport._client._active_prompts["s-1"] = prompt

    msg = {
        "method": "session/update",
        "params": {
            "sessionId": "s-2",
            "update": {"sessionUpdate": "agent_message", "content": {"type": "text", "text": "hi"}},
        },
    }
    transport._route_update(msg)
    assert "hi" in prompt.chunks


async def _noop() -> None:
    """No-op coroutine for monkey-patching transport background tasks."""
    return


def test_start_transport_calls_initialize(monkeypatch: Any) -> None:
    """Regression: _start_transport must call self.call('initialize', ...).

    Before the fix it referenced the non-existent AcpTransport._call,
    which was a left-over from when the startup code lived on AcpClient.
    """
    transport = _make_transport()
    transport._loop = asyncio.new_event_loop()

    called: dict[str, Any] = {}

    async def fake_call(method: str, params: Any, timeout: float | None = None) -> Any:
        called["method"] = method
        called["params"] = params
        called["timeout"] = timeout
        return {"agentInfo": {"title": "Test", "version": "0.0"}}

    monkeypatch.setattr(transport, "call", fake_call)
    monkeypatch.setattr(transport, "_reader", _noop)
    monkeypatch.setattr(transport, "_stderr_drain", _noop)

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> Any:
        return FakeProcess(returncode=None)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    asyncio.run(transport._start_transport())

    assert called.get("method") == "initialize"
    assert called["params"]["protocolVersion"] == 1
    assert called["params"]["clientInfo"]["name"] == "diploid-agent"
    assert called["timeout"] == transport._client._startup_timeout
