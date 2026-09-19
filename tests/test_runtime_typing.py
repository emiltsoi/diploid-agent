"""Tests for RuntimeTyping wake-turn typing presence."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from diploid_agent.runtime.typing import RuntimeTyping


def _wait_calls(mock: MagicMock, minimum: int = 1, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while mock.call_count < minimum and time.monotonic() < deadline:
        time.sleep(0.01)


def test_turn_typing_heartbeats_until_finished() -> None:
    notifier = MagicMock()
    typing = RuntimeTyping(notifier_fn=lambda: notifier)
    typing.on_turn_started("chat-1")
    try:
        _wait_calls(notifier.typing)
        assert notifier.typing.call_count >= 1
        assert notifier.typing.call_args[0][0] == "chat-1"
    finally:
        typing.on_turn_finished("chat-1")
        typing.stop()


def test_turn_finish_without_start_is_noop() -> None:
    notifier = MagicMock()
    typing = RuntimeTyping(notifier_fn=lambda: notifier)
    typing.on_turn_finished("chat-1")
    typing.on_turn_finished("chat-1")
    assert "chat-1" not in typing._turn_chats
    assert typing._counts.get("chat-1", 0) == 0
    typing.stop()


def test_turn_finish_does_not_eat_task_count() -> None:
    """A user turn (never started) finishing mid-task must not kill typing."""
    notifier = MagicMock()
    typing = RuntimeTyping(notifier_fn=lambda: notifier)
    task = SimpleNamespace(chat_id="chat-1")
    typing.on_task_started("plan-1", task)
    typing.on_turn_finished("chat-1")  # user-turn cleanup, never started
    assert typing._counts["chat-1"] == 1
    assert "chat-1" in typing._threads
    typing.on_task_done("plan-1", task, ("done", "", 0))
    assert typing._counts["chat-1"] == 0
    typing.stop()


def test_turn_and_task_compose_on_one_heartbeat() -> None:
    """Turn + task share one heartbeat; typing survives until both end."""
    notifier = MagicMock()
    typing = RuntimeTyping(notifier_fn=lambda: notifier)
    task = SimpleNamespace(chat_id="chat-1")
    typing.on_turn_started("chat-1")
    typing.on_task_started("plan-1", task)
    assert typing._counts["chat-1"] == 2
    assert len(typing._threads) == 1
    typing.on_task_done("plan-1", task, ("done", "", 0))
    assert typing._counts["chat-1"] == 1
    assert "chat-1" in typing._threads
    typing.on_turn_finished("chat-1")
    assert typing._counts["chat-1"] == 0
    assert "chat-1" not in typing._threads
    typing.stop()


def test_task_without_chat_id_does_not_type() -> None:
    notifier = MagicMock()
    typing = RuntimeTyping(notifier_fn=lambda: notifier)
    task = SimpleNamespace(chat_id=None)
    typing.on_task_started("plan-1", task)
    typing.on_task_done("plan-1", task, ("done", "", 0))
    assert typing._counts == {}
    typing.stop()
