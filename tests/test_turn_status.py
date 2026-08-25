"""Tests for the long-polling /turn status endpoint."""

import threading
import time

from devin_fleet_harness.models import ActiveTurn
from devin_fleet_harness.runtime.turn_controller import TurnController


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
