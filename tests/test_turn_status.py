"""Tests for the long-polling /turn status endpoint."""

import threading
import time
from types import SimpleNamespace

from diploid_agent.models import ActiveTurn
from diploid_agent.runtime.turn_controller import TurnController
from diploid_agent.turn.stream import TurnStream


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


def test_turn_status_exposes_last_side_effect() -> None:
    runtime = _FakeRuntime()
    controller = TurnController(runtime)
    active = ActiveTurn(
        chat_id="chat-1",
        session_id="session-1",
        user_message="hello",
        start_time=time.time(),
    )
    active.last_side_effect = "exec (running)"
    active.last_side_effect_at = time.time()
    runtime._active_turns["chat-1"] = active

    status = controller.turn_status("chat-1")

    assert status["last_side_effect"] == "exec (running)"
    assert status["last_side_effect_at"] == active.last_side_effect_at


def test_turn_status_long_poll_wakes_on_side_effect() -> None:
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
            active.last_side_effect = "exec (running)"
            active.last_side_effect_at = time.time()
        with active._condition:
            active._condition.notify_all()

    threading.Thread(target=updater, daemon=True).start()

    start = time.perf_counter()
    status = controller.turn_status("chat-1", wait=5.0)
    elapsed = time.perf_counter() - start

    assert status["last_side_effect"] == "exec (running)"
    assert elapsed < 1.0


def test_turn_stream_dedups_identical_tool_updates() -> None:
    runtime = _FakeRuntime()
    active = ActiveTurn(
        chat_id="chat-1",
        session_id="session-1",
        user_message="hello",
        start_time=time.time(),
    )
    runtime._active_turns["chat-1"] = active
    stream = TurnStream(runtime, "chat-1")

    # Real ACP wire shape: title/status live at the top level of the update.
    update = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "call-1",
        "title": "exec",
        "status": "in_progress",
    }
    stream.on_update(update)
    first_at = active.last_side_effect_at
    assert active.last_side_effect == "exec (in_progress)"

    stream.on_update(update)
    # Identical title+status composes the same line: no stamp, no notify.
    assert active.last_side_effect_at == first_at
    # Breadcrumbs still record every event.
    assert len(active.side_effects) == 2

    stream.on_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call-1",
            "title": "exec",
            "status": "completed",
        }
    )
    assert active.last_side_effect == "exec (completed)"
    assert active.last_side_effect_at > first_at


def test_turn_stream_falls_back_to_nested_content_title() -> None:
    """Non-standard updates nesting title/status inside content still work."""
    runtime = _FakeRuntime()
    active = ActiveTurn(
        chat_id="chat-1",
        session_id="session-1",
        user_message="hello",
        start_time=time.time(),
    )
    runtime._active_turns["chat-1"] = active
    stream = TurnStream(runtime, "chat-1")

    stream.on_update(
        {
            "sessionUpdate": "tool_call",
            "content": {"title": "exec", "status": "running"},
        }
    )
    assert active.last_side_effect == "exec (running)"


def test_turn_stream_prefers_raw_input_command_over_terminal_title() -> None:
    """Exec terminal-id titles are replaced by the real command line."""
    runtime = _FakeRuntime()
    active = ActiveTurn(
        chat_id="chat-1",
        session_id="session-1",
        user_message="hello",
        start_time=time.time(),
    )
    runtime._active_turns["chat-1"] = active
    stream = TurnStream(runtime, "chat-1")

    stream.on_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "call-1",
            "kind": "execute",
            "title": "exec:0#8d26229e80f5426896d800c6b1e9fd15",
            "status": "in_progress",
            "rawInput": {"command": "sleep 8 && git log --oneline -1"},
        }
    )
    assert active.last_side_effect == "execute: sleep 8 && git log --oneline -1 (in_progress)"
    # rawInput keys are captured into the breadcrumb for inspection.
    assert active.side_effects[-1]["input"]["command"].startswith("sleep 8")


def test_turn_stream_remembers_command_across_updates() -> None:
    """tool_call_update chunks lack rawInput; the learned command persists."""
    runtime = _FakeRuntime()
    active = ActiveTurn(
        chat_id="chat-1",
        session_id="session-1",
        user_message="hello",
        start_time=time.time(),
    )
    runtime._active_turns["chat-1"] = active
    stream = TurnStream(runtime, "chat-1")

    stream.on_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "call-1",
            "kind": "execute",
            "title": "exec:0#8d26229e80f5426896d800c6b1e9fd15",
            "status": "in_progress",
            "rawInput": {"command": "sleep 8"},
        }
    )
    stream.on_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call-1",
            "title": "exec:0#8d26229e80f5426896d800c6b1e9fd15",
            "status": "completed",
        }
    )
    assert active.last_side_effect == "execute: sleep 8 (completed)"


class _FakeRuntime:
    def __init__(self) -> None:
        self._active_turns: dict[str, ActiveTurn] = {}
        self._lock = threading.RLock()
        self._plugins = SimpleNamespace(on_partial=lambda *a, **k: None)
        self.active_record = lambda chat_id: None  # type: ignore[method-assign]


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
    record = SimpleNamespace(session_id=record_session_id) if record_session_id else None
    runtime.active_record = lambda chat_id: record  # type: ignore[attr-defined,method-assign]


def test_stop_cancels_live_session_when_recorded_id_is_stale() -> None:
    """A mid-turn session swap (force_new_session) leaves ActiveTurn.session_id
    stale while the in-flight prompt is registered under the new id. /stop must
    cancel the live session, not the recorded one."""
    runtime = _FakeRuntime()
    engine = _FakeEngine(live_session_id="new-session")
    _attach_stop_fakes(runtime, engine, record_session_id="old-session")
    runtime._active_turns["chat-1"] = ActiveTurn("chat-1", "old-session", "hello", time.time())
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
    runtime._active_turns["chat-1"] = ActiveTurn("chat-1", "old-session", "hello", time.time())
    controller = TurnController(runtime)

    result = controller.stop("chat-1")

    assert "stopping" in result.reply.lower()
    assert engine.cancelled == ["old-session"]
