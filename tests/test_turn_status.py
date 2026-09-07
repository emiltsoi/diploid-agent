"""Tests for the long-polling /turn status endpoint."""

import threading
import time
from types import SimpleNamespace

from diploid_agent.models import ActiveTurn
from diploid_agent.runtime.turn_controller import TurnController


def test_turn_status_returns_immediately_when_idle() -> None:
    runtime = _FakeRuntime()
    controller = TurnController(runtime)
    assert controller.turn_status("chat-1") == {"chat_id": "chat-1", "status": "idle"}


def test_turn_status_long_poll_wakes_on_message_update() -> None:
    runtime = _FakeRuntime()
    controller = TurnController(runtime)
    active = ActiveTurn(
        chat_id="chat-1",
        session_id="session-1",
        user_message="hello",
        start_time=time.time(),
    )
    runtime._active_turns["chat-1"] = active

    def updater() -> None:
        time.sleep(0.05)
        with runtime._lock:
            active.message_text = "world"
        with active._condition:
            active._condition.notify_all()

    threading.Thread(target=updater, daemon=True).start()

    start = time.perf_counter()
    status = controller.turn_status("chat-1", wait=5.0)
    elapsed = time.perf_counter() - start

    assert status["message_text"] == "world"
    assert status["status"] == "running"
    assert elapsed < 1.0


def test_turn_status_long_poll_returns_unchanged_after_timeout() -> None:
    runtime = _FakeRuntime()
    controller = TurnController(runtime)
    active = ActiveTurn(
        chat_id="chat-1",
        session_id="session-1",
        user_message="hello",
        start_time=time.time(),
    )
    runtime._active_turns["chat-1"] = active

    start = time.perf_counter()
    status = controller.turn_status("chat-1", wait=0.1)
    elapsed = time.perf_counter() - start

    assert status["status"] == "running"
    assert 0.05 <= elapsed <= 0.5


class _FakeRuntime:
    def __init__(self) -> None:
        self._active_turns: dict[str, ActiveTurn] = {}
        self._lock = threading.RLock()


class _FakeEngine:
    def __init__(self, live_session_id: str | None = None) -> None:
        self.live_session_id = live_session_id
        self.cancelled: list[str] = []

    def active_session_id(self) -> str | None:
        return self.live_session_id

    def cancel(self, session_id: str) -> None:
        self.cancelled.append(session_id)


def _attach_stop_fakes(
    runtime: _FakeRuntime,
    engine: _FakeEngine,
    record_session_id: str | None,
) -> None:
    runtime.engine = engine  # type: ignore[attr-defined]
    runtime.wake_queue = None  # type: ignore[attr-defined]
    record = (
        SimpleNamespace(session_id=record_session_id) if record_session_id else None
    )
    runtime._active_record = lambda chat_id: record  # type: ignore[attr-defined,method-assign]


def test_stop_cancels_live_session_when_recorded_id_is_stale() -> None:
    """A mid-turn session swap (force_new_session) leaves ActiveTurn.session_id
    stale while the in-flight prompt is registered under the new id. /stop must
    cancel the live session, not the recorded one."""
    runtime = _FakeRuntime()
    engine = _FakeEngine(live_session_id="new-session")
    _attach_stop_fakes(runtime, engine, record_session_id="old-session")
    runtime._active_turns["chat-1"] = ActiveTurn(
        "chat-1", "old-session", "hello", time.time()
    )
    controller = TurnController(runtime)

    result = controller.stop("chat-1")

    assert "stopping" in result.reply.lower()
    assert engine.cancelled[0] == "new-session"
    assert set(engine.cancelled) == {"new-session", "old-session"}
    assert runtime._active_turns["chat-1"].stopped is True


def test_stop_uses_recorded_session_when_no_live_prompt() -> None:
    runtime = _FakeRuntime()
    engine = _FakeEngine(live_session_id=None)
    _attach_stop_fakes(runtime, engine, record_session_id="old-session")
    runtime._active_turns["chat-1"] = ActiveTurn(
        "chat-1", "old-session", "hello", time.time()
    )
    controller = TurnController(runtime)

    result = controller.stop("chat-1")

    assert "stopping" in result.reply.lower()
    assert engine.cancelled == ["old-session"]
