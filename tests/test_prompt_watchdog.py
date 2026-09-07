"""Focused unit tests for PromptWatchdog."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any

from diploid_agent.acp_client.types import _Prompt
from diploid_agent.acp_client.watchdog import PromptWatchdog


class FakeProcess:
    pid = 12345
    returncode: int | None = None


class FakeLoop:
    def is_running(self) -> bool:
        return True

    def call_soon_threadsafe(self, cb: Any, *args: Any) -> None:
        pass


class FakeMetrics:
    def __init__(self) -> None:
        self.counters: dict[str, int] = {}

    def inc(self, name: str, *, reason: str | None = None) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1


class FakeLifecycleLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, dict[str, Any] | None]] = []

    def write(
        self,
        event: str,
        *,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.events.append((event, reason, detail))


class FakeTransport:
    def __init__(self) -> None:
        self.generation = 1


class FakeClient:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._watchdog_interval = 0.01
        self._watchdog_timeout = 1.0
        self._proc: FakeProcess | None = FakeProcess()
        self._inflight_future: concurrent.futures.Future[Any] | None = None
        self._inflight_deadline = time.monotonic() + 60.0
        self._last_request_at = time.monotonic()
        self._last_control_call_deadline = 0.0
        self._active_prompts: dict[str, _Prompt] = {}
        self._pending: dict[int, Any] = {}
        self.metrics = FakeMetrics()
        self._transport_healthy = True
        self._initialized = True
        self._unblock_reason: str | None = None
        self._killed: list[FakeProcess] = []
        self._loop: Any = FakeLoop()
        self._transport = FakeTransport()
        self._lifecycle_log = FakeLifecycleLog()

    def _unblock_inflight(self, reason: str) -> None:
        self._unblock_reason = reason

    def _kill_process_group(self, proc: FakeProcess) -> None:
        self._killed.append(proc)


def _make_prompt(soft_timeout: float = 600.0) -> _Prompt:
    return _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=concurrent.futures.Future(),
        cancel_done=concurrent.futures.Future(),
        soft_timeout=soft_timeout,
        started_at=time.monotonic(),
    )


def test_watchdog_start_stop() -> None:
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    watchdog.start()
    assert watchdog._thread is not None
    assert watchdog._thread.is_alive()
    watchdog.stop()
    assert watchdog._thread is None or not watchdog._thread.is_alive()


def test_watchdog_noop_when_not_running() -> None:
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() - 1.0
    # _running is False by default; check should bail.
    watchdog.check()
    assert client._unblock_reason is None


def test_watchdog_fires_on_deadline() -> None:
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() - 1.0
    client._pending[1] = None  # non-prompt control call
    client._active_prompts = {}
    watchdog._running = True
    watchdog.check()
    assert client._unblock_reason == "ACP transport watchdog detected a stall"
    assert client._transport_healthy is False
    assert client._initialized is False
    assert "acp_watchdog_fired_total" in client.metrics.counters


def test_watchdog_fires_on_exited_process() -> None:
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() + 60.0
    assert client._proc is not None
    client._proc.returncode = 1
    watchdog._running = True
    watchdog.check()
    assert client._unblock_reason == "ACP transport watchdog detected a stall"


def test_watchdog_noop_for_silent_prompt() -> None:
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() + 60.0
    client._active_prompts["s-1"] = _make_prompt()
    client._last_progress_at = time.monotonic() - 100.0
    client._last_stdout_at = time.monotonic()
    watchdog._running = True
    watchdog.check()
    assert client._unblock_reason is None


def test_watchdog_fires_on_control_call_timeout() -> None:
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() + 60.0
    client._last_request_at = time.monotonic() - 10.0
    client._last_control_call_deadline = time.monotonic() - 1.0
    client._pending[1] = None
    client._active_prompts = {}
    watchdog._running = True
    watchdog.check()
    assert client._unblock_reason == "ACP transport watchdog detected a stall"


def test_watchdog_uses_global_timeout_without_per_call_deadline() -> None:
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() + 60.0
    client._last_request_at = time.monotonic() - client._watchdog_timeout - 1.0
    client._last_control_call_deadline = 0.0
    client._pending[1] = None
    client._active_prompts = {}
    watchdog._running = True
    watchdog.check()
    assert client._unblock_reason == "ACP transport watchdog detected a stall"


def test_watchdog_does_not_kill_newer_generation() -> None:
    """Regression: a stall judged on gen N must not kill the gen N+1 child.

    ``check()`` observes under ``_lock`` but recovery runs later under
    ``_lifecycle_lock``; ``_ensure_started`` can complete a restart in between.
    Recovery must re-verify the generation/proc identity before killing.
    """
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() - 1.0
    client._pending[1] = None
    client._active_prompts = {}
    watchdog._running = True

    old_proc = client._proc
    new_proc = FakeProcess()
    inner = watchdog._stall_recovery_inner

    def _restart_then_recover(*args: Any) -> Any:
        # _ensure_started finishes a restart while recovery waits on
        # _lifecycle_lock: new generation, new child process.
        client._transport.generation = 2
        client._proc = new_proc
        return inner(*args)

    watchdog._stall_recovery_inner = _restart_then_recover  # type: ignore[assignment]
    watchdog.check()

    assert client._killed == []
    assert old_proc not in client._killed
    assert client._proc is new_proc
    assert client._unblock_reason is None
    assert client._transport_healthy is True
    assert client._initialized is True
    assert any(
        event == "watchdog.recovery.skipped" and reason == "stale_generation"
        for event, reason, _ in client._lifecycle_log.events
    )


def test_watchdog_dead_proc_recovery_skips_replaced_transport() -> None:
    """The proc_dead trigger must also not land on a replacement child."""
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() + 60.0
    assert client._proc is not None
    client._proc.returncode = 1
    watchdog._running = True

    new_proc = FakeProcess()
    inner = watchdog._stall_recovery_inner

    def _restart_then_recover(*args: Any) -> Any:
        client._transport.generation = 2
        client._proc = new_proc
        return inner(*args)

    watchdog._stall_recovery_inner = _restart_then_recover  # type: ignore[assignment]
    watchdog.check()

    assert client._killed == []
    assert client._proc is new_proc
    assert client._transport_healthy is True
    assert any(
        event == "watchdog.recovery.skipped" and reason == "stale_generation"
        for event, reason, _ in client._lifecycle_log.events
    )


def test_watchdog_skips_recovery_when_stall_cleared() -> None:
    """If the in-flight call finished while recovery waited, do not kill."""
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    inflight = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = time.monotonic() - 1.0
    client._pending[1] = None
    client._active_prompts = {}
    watchdog._running = True

    inner = watchdog._stall_recovery_inner

    def _complete_then_recover(*args: Any) -> Any:
        inflight.set_result(None)
        return inner(*args)

    watchdog._stall_recovery_inner = _complete_then_recover  # type: ignore[assignment]
    watchdog.check()

    assert client._killed == []
    assert client._unblock_reason is None
    assert client._transport_healthy is True
    assert any(
        event == "watchdog.recovery.skipped" and reason == "cleared"
        for event, reason, _ in client._lifecycle_log.events
    )


def test_watchdog_recovery_stamps_observed_identity() -> None:
    """The watchdog_stall event records which generation was judged stalled."""
    client = FakeClient()
    watchdog = PromptWatchdog(client)
    client._inflight_future = concurrent.futures.Future()
    client._inflight_deadline = time.monotonic() - 1.0
    client._pending[1] = None
    client._active_prompts = {}
    watchdog._running = True
    watchdog.check()

    restart_events = [
        detail
        for event, reason, detail in client._lifecycle_log.events
        if event == "transport.restart" and reason == "watchdog_stall"
    ]
    assert restart_events
    detail = restart_events[0]
    assert detail is not None
    assert detail["killed"] is True
    assert detail["trigger"] == "inflight_deadline"
    assert detail["stalled_gen"] == 1
    assert detail["stalled_pid"] == 12345
