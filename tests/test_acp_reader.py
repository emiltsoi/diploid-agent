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
import os
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
UPDATES = int(os.environ.get("FAKE_ACP_UPDATES", "0"))
HANG_METHODS = set(
    m for m in os.environ.get("FAKE_ACP_HANG_METHODS", "").split(",") if m
)


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
    if method in HANG_METHODS:
        continue  # never respond: exercises caller-side timeout paths
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
        for i in range(UPDATES):
            _send({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": sid,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t-1",
                    "i": i,
                },
            }})
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


def _wait_for(predicate: Any, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """Poll until predicate() is true; prompt callbacks run on a worker thread
    and may land after send_message returns."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


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
        assert _wait_for(
            lambda: any(
                u.get("rawOutput") and len(u["rawOutput"]) == 200 * 1024 for u in updates
            )
        )
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


def test_blocking_on_chunk_does_not_stall_reader(fake_acp: Path) -> None:
    """A slow on_chunk must not starve the stdout reader.

    Prompt callbacks run harness code that takes ``runtime._lock`` and can
    block on plugin or memory work.  If they ran on the ACP loop, the reader
    would stall, the stdout pipe would fill, and the child would wedge on
    write -- the same hang signature as the oversized-line bug.
    """
    client = _make_client(fake_acp)
    entered = threading.Event()
    release = threading.Event()

    def slow_chunk(_text: str) -> None:
        entered.set()
        release.wait(timeout=30)

    try:
        # The prompt must complete without waiting for the blocked callback:
        # on the old code path (callback on the ACP loop) the response could
        # never be routed and send_message would hang until timeout.
        result = client.send_message("s-1", "hi", on_chunk=slow_chunk, timeout=30.0)
        assert result.reply == "ok"
        assert result.stop_reason == "end_turn"
        assert client.health()
        # The callback still runs, just on the worker thread.
        assert entered.wait(timeout=10)
    finally:
        release.set()
        client.close()


def test_call_unlocked_releases_all_rlock_levels() -> None:
    """_call_unlocked must drop every held level, not just one.

    Nested @_locked paths (e.g. runtime.dispatch -> controller.dispatch ->
    dispatch.dispatch) stack acquisitions on the same runtime RLock.  A
    single release would leave the lock held for the whole engine call and
    block the ACP prompt-callback worker for its duration.
    """
    from diploid_agent.runtime.agent_runtime import AgentRuntime

    rt = AgentRuntime.__new__(AgentRuntime)
    rt._lock = threading.RLock()
    rt._lock.acquire()
    rt._lock.acquire()

    observed: dict[str, bool] = {}

    def probe() -> None:
        observed["owned"] = rt._lock._is_owned()

    rt._call_unlocked(probe)

    assert observed["owned"] is False
    # Both levels were reacquired: releasing twice succeeds, a third raises.
    rt._lock.release()
    rt._lock.release()
    with pytest.raises(RuntimeError):
        rt._lock.release()


def test_background_timeout_does_not_poison_transport(
    fake_acp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A background call that times out must not mark the transport unhealthy.

    Memory summaries and other best-effort calls share the chat's transport.
    If a background timeout forced a restart, the next foreground turn would
    lose its session for no fault of its own.
    """
    monkeypatch.setenv("FAKE_ACP_HANG_METHODS", "session/prompt")
    client = _make_client(fake_acp)
    try:
        result = client.send_message("s-1", "hi", timeout=1.0, background=True)
        assert result.timed_out
        assert result.stop_reason == "timeout"
        # The transport stays healthy and remains usable for real work.
        assert client._transport_healthy is True
        assert client.health()
        assert client.session_alive("s-1")
    finally:
        client.close()


def test_foreground_timeout_marks_transport_unhealthy(
    fake_acp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Foreground timeouts keep the eager unhealthy marking."""
    monkeypatch.setenv("FAKE_ACP_HANG_METHODS", "session/prompt")
    client = _make_client(fake_acp)
    try:
        result = client.send_message("s-1", "hi", timeout=1.0)
        assert result.timed_out
        assert client._transport_healthy is False
    finally:
        client.close()


def test_resume_budget_bounds_hanging_resume(
    fake_acp: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stalled session/resume must give up inside its budget, not linger.

    ``acp_resume_timeout`` is a real end-to-end budget shared by the
    resume/load attempts and the config re-apply: an opportunistic resume
    should fail fast so the caller can fall back to prompt rehydration.
    """
    from diploid_agent.acp_client.lifecycle import AcpLifecycleLog

    monkeypatch.setenv("FAKE_ACP_HANG_METHODS", "session/resume,session/load")
    log = AcpLifecycleLog(tmp_path / "acp-lifecycle.jsonl")
    client = _make_client(fake_acp)
    client._lifecycle_log = log
    try:
        start = time.monotonic()
        with pytest.raises(AcpTransportError):
            client.resume_session("s-1", timeout=1.5)
        elapsed = time.monotonic() - start
        # Budget 1.5s plus the _run headroom; must be far below the old
        # control-timeout-scale wait.
        assert elapsed < 20.0
        events = log.recent_events()
        failures = [e for e in events if e["event"] == "session.resume.failure"]
        assert failures
    finally:
        client.close()


def test_prompt_updates_buffer_is_bounded(
    fake_acp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retained prompt updates are capped; the live callback still sees all."""
    from diploid_agent.acp_client.types import _PROMPT_UPDATES_MAXLEN

    monkeypatch.setenv("FAKE_ACP_UPDATES", str(_PROMPT_UPDATES_MAXLEN + 50))
    client = _make_client(fake_acp)
    try:
        seen: list[dict[str, Any]] = []
        result = client.send_message("s-1", "hi", on_update=seen.append, timeout=30.0)
        assert result.reply == "ok"
        # The retained tail is bounded while the live callback saw everything.
        # The callback worker drains asynchronously; give it a moment.
        expected = _PROMPT_UPDATES_MAXLEN + 50
        assert _wait_for(lambda: len(seen) >= expected)
        assert len(result.updates) == _PROMPT_UPDATES_MAXLEN
        # Drop-oldest semantics: the most recent updates are retained.
        assert result.updates[-1].get("sessionUpdate") == "agent_message_chunk"
    finally:
        client.close()


def test_cb_queue_overflow_drops_without_blocking() -> None:
    """A full callback queue drops work instead of backpressuring the reader."""
    import queue as queue_mod

    client = _FakeClient()
    transport = AcpTransport(client)
    # A non-None _cb_thread routes through the queue; a tiny queue fills fast.
    transport._cb_thread = threading.Thread(target=lambda: None)
    transport._cb_queue = queue_mod.Queue(maxsize=2)

    ran: list[str] = []
    transport._dispatch_cb(lambda a: ran.append(a), "a")
    transport._dispatch_cb(lambda a: ran.append(a), "b")
    start = time.monotonic()
    transport._dispatch_cb(lambda a: ran.append(a), "c")
    assert time.monotonic() - start < 1.0
    assert transport._cb_dropped == 1


def test_lifecycle_events_carry_pid_and_generation(tmp_path: Path) -> None:
    """Lifecycle entries are stamped with pid + transport_gen for postmortems."""
    from diploid_agent.acp_client.lifecycle import AcpLifecycleLog

    log = AcpLifecycleLog(tmp_path / "acp-lifecycle.jsonl")
    log.context = lambda: {"pid": 4242, "transport_gen": 7}
    log.write("transport.start", detail={"child_pid": 999})
    events = log.recent_events()
    assert events[0]["pid"] == 4242
    assert events[0]["transport_gen"] == 7
    assert events[0]["detail"]["child_pid"] == 999
    # Explicit kwargs override the provider.
    log.write("transport.restart", pid=1, transport_gen=2)
    assert log.recent_events()[-1]["pid"] == 1


def test_close_logs_transport_stop_once_per_generation(tmp_path: Path) -> None:
    """close() records transport.stop for the current generation, once."""
    from diploid_agent.acp_client.lifecycle import AcpLifecycleLog

    log = AcpLifecycleLog(tmp_path / "acp-lifecycle.jsonl")
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", lifecycle_log=log)
    client._transport.generation = 3
    client._proc = _FakeProcForStart()  # type: ignore[assignment]

    client.close()
    client.close()

    stops = [e for e in log.recent_events() if e["event"] == "transport.stop"]
    assert len(stops) == 1
    assert stops[0]["transport_gen"] == 3
    assert stops[0]["pid"] == os.getpid()


def test_watchdog_emits_prompt_silence_telemetry(tmp_path: Path) -> None:
    """An in-flight prompt with a live child and no stdout gets flagged."""
    from diploid_agent.acp_client.lifecycle import AcpLifecycleLog
    from diploid_agent.acp_client.watchdog import PromptWatchdog

    log = AcpLifecycleLog(tmp_path / "acp-lifecycle.jsonl")
    client = _FakeClient()
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() + 3600.0
    client._last_request_at = time.monotonic()
    client._last_stdout_at = time.monotonic() - 700.0
    client._last_control_call_deadline = 0.0
    client._active_prompts = {"s-1": object()}
    client._pending = {1: object()}
    client._proc = _FakeProc()
    client._silence_warn_after = 100.0
    client._lifecycle_log = log
    client.metrics = None

    watchdog = PromptWatchdog(client)
    watchdog._running = True
    watchdog._last_silence_warn = -1e9  # defeat throttle on freshly booted hosts
    watchdog.check()

    events = [e for e in log.recent_events() if e["event"] == "prompt.silence"]
    assert len(events) == 1
    assert events[0]["session_id"] == "s-1"
    assert events[0]["detail"]["silence_s"] >= 700.0
    # Throttled: a second check inside the same interval does not repeat.
    watchdog.check()
    events = [e for e in log.recent_events() if e["event"] == "prompt.silence"]
    assert len(events) == 1
