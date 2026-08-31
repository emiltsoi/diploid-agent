"""Tests for the Telegram long-polling ingress."""

import json
import time
from pathlib import Path
from typing import Any

import httpx

from diploid_agent.config import (
    NotificationsConfig,
    TaskConfig,
    TelegramConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.models import ChatResult
from diploid_agent.telegram_poll import ChatInput, TelegramPoller, TurnWorker
from diploid_agent.transport.telegram import DeliveryWorker, _format_thought


def _update(**kwargs) -> dict:
    """Build a minimal Telegram update with a message."""
    update = {"update_id": 1}
    update.update(kwargs)
    return update


def test_parse_update_text_message() -> None:
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "hello",
        }
    )
    parsed = TelegramPoller._parse_update(update)
    assert isinstance(parsed, ChatInput)
    assert parsed.chat_id == 12345
    assert parsed.message_id == 1
    assert parsed.text == "hello"
    assert parsed.reply_to is None
    assert parsed.reply_to_message_id is None


def test_parse_update_with_reply_to_user() -> None:
    update = _update(
        message={
            "message_id": 2,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "explain",
            "reply_to_message": {
                "message_id": 1,
                "from": {"id": 2, "is_bot": False},
                "text": "The previous user message.",
            },
        }
    )
    parsed = TelegramPoller._parse_update(update)
    assert parsed.reply_to == "The previous user message."
    assert parsed.reply_to_is_bot is False
    assert parsed.reply_to_message_id == 1


def test_parse_update_with_reply_to_bot() -> None:
    update = _update(
        message={
            "message_id": 3,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "expand",
            "reply_to_message": {
                "message_id": 2,
                "from": {"id": 0, "is_bot": True},
                "text": "The bot reply.",
            },
        }
    )
    parsed = TelegramPoller._parse_update(update)
    assert parsed.reply_to == "The bot reply."
    assert parsed.reply_to_is_bot is True
    assert parsed.reply_to_message_id == 2


def test_parse_update_uses_caption_when_no_text() -> None:
    update = _update(
        message={
            "message_id": 4,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "describe this",
            "reply_to_message": {
                "message_id": 3,
                "from": {"id": 2, "is_bot": False},
                "caption": "An image caption.",
            },
        }
    )
    parsed = TelegramPoller._parse_update(update)
    assert parsed.reply_to == "An image caption."
    assert parsed.reply_to_message_id == 3


def test_parse_update_skips_bot_messages() -> None:
    update = _update(
        message={
            "message_id": 5,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 0, "is_bot": True},
            "text": "bot echo",
        }
    )
    assert TelegramPoller._parse_update(update) is None


def test_parse_update_requires_text() -> None:
    update = _update(
        message={
            "message_id": 6,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
        }
    )
    assert TelegramPoller._parse_update(update) is None


def test_make_preview_respects_max_chars() -> None:
    text = "word " * 100  # 500 characters
    preview, original_length = TelegramPoller._make_preview(text, 120)
    assert len(preview) <= 120
    assert original_length == 500
    assert preview.startswith("word ")


def test_register_message_ids(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        reply_preview_chars=20,
    )
    poller._register_message_ids(
        chat_id=12345,
        message_ids=[101, 102],
        session_number=1,
        turn_number=3,
        text="This is the longer assistant reply.",
        kind="reply",
    )
    path = poller._message_registry_path(12345)
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert entry["chat_id"] == 12345
        assert entry["session_number"] == 1
        assert entry["turn_number"] == 3
        assert entry["kind"] == "reply"
        assert "preview" in entry
        assert entry["original_length"] == 35


class _FakeResponse:
    def __init__(self, data: Any) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._data


class _FakeClient:
    def __init__(self, url_data: dict[str, Any]) -> None:
        self._url_data = url_data

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._url_data.get(url, {}))

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(self._url_data.get(url, {}))


def test_harness_metrics_with_data() -> None:
    poller = TelegramPoller(token="dummy", harness_url="http://localhost")
    poller._local.client = _FakeClient(
        {
            "http://localhost/metrics/12345": {
                "cumulative": {
                    "turns": 3,
                    "input_tokens": 100,
                    "output_tokens": 200,
                    "total_tokens": 300,
                    "cached_tokens": 50,
                    "latency_seconds": 1.234,
                },
                "last_turn": {
                    "turn_number": 3,
                    "model": "glm-5-2",
                    "total_tokens": 120,
                    "latency_seconds": 0.456,
                },
            },
        }
    )
    result = poller._harness_metrics(12345)
    assert "Turns: 3" in result
    assert "Tokens: 300 total (100 in / 200 out)" in result
    assert "Cached tokens: 50" in result
    assert "Latency: 1.23s" in result
    assert "Last turn: #3 (glm-5-2) — 120 tokens in 0.46s" in result


def test_harness_metrics_empty() -> None:
    poller = TelegramPoller(token="dummy", harness_url="http://localhost")
    poller._local.client = _FakeClient(
        {
            "http://localhost/metrics/12345": {"cumulative": {}, "last_turn": None},
        }
    )
    result = poller._harness_metrics(12345)
    assert result == "No metrics for this chat yet."


def test_harness_metrics_error() -> None:
    class _FailingClient:
        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    poller = TelegramPoller(token="dummy", harness_url="http://localhost")
    poller._local.client = _FailingClient()
    result = poller._harness_metrics(12345)
    assert result == "Sorry, I could not fetch your chat metrics."


def test_handle_update_routes_metrics_command(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_metrics = lambda chat_id: "Metrics reply"
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/metrics",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Metrics reply", 1)]


def test_handle_update_routes_metrics_command_with_bot_username(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_metrics = lambda chat_id: "Metrics reply"
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/metrics@devin_pilot_bot",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Metrics reply", 1)]


def test_handle_update_routes_command_with_extra_whitespace(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_metrics = lambda chat_id: "Metrics reply"
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "  /metrics   ",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Metrics reply", 1)]


def test_handle_update_routes_recall_command_with_bot_username_and_args(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    calls: list[tuple[int, str]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        pass

    poller._send_text = fake_send
    poller._harness_recall = lambda chat_id, query: calls.append((chat_id, query)) or "recalled"
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/recall@devin_pilot_bot project setup",
        }
    )
    poller._handle_update(update)
    assert calls == [(12345, "project setup")]


def test_handle_update_routes_mcp_list(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_mcp_list = lambda chat_id: "MCP servers: github"
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/mcp list",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "MCP servers: github", 1)]


def test_handle_update_routes_mcp_enable(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_mcp_enable = lambda chat_id, name: f"enabled {name}"
    update = _update(
        message={
            "message_id": 2,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/mcp enable github",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "enabled github", 2)]


def test_handle_update_routes_skill_list(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_skill_list = lambda chat_id: "Skills: review"
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/skill list",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Skills: review", 1)]


def test_handle_update_routes_skill_disable(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_skill_disable = lambda chat_id, name: f"disabled {name}"
    update = _update(
        message={
            "message_id": 3,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/skill disable review",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "disabled review", 3)]


def test_handle_update_routes_state_command(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_state_event = lambda chat_id, plugin, event, raw_args: (
        f"Set {plugin} {event} with {raw_args!r}."
    )
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/state curriculum set_target_language Klingon",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Set curriculum set_target_language with 'Klingon'.", 1)]


def test_handle_update_routes_help_command(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/help",
        }
    )
    poller._handle_update(update)
    assert len(sent) == 1
    assert sent[0][0] == 12345
    assert "Available commands:" in sent[0][1]
    assert sent[0][2] == 1


def test_handle_update_routes_restart_command(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_restart = lambda chat_id: {"reply": "Restarted", "notice": None}
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/restart",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Restarted", 1)]


def test_stream_turn_finalizes_thought_before_final(tmp_path: Path) -> None:
    """When thinking is streamed, the thought block is finalized before the final placeholder is created."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        stream_chunk_interval=0.0,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    class FakeFuture:
        _done = False

        def done(self) -> bool:
            if not self._done:
                self._done = True
                return False
            return True

        def result(self) -> dict[str, Any]:
            return {"reply": "final reply", "notice": None}

        def cancel(self) -> None:
            pass

    next_id = iter([100])
    calls: list[tuple[str, Any]] = []

    worker._harness_turn_status = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "status": "running",
        "message_text": "",
        "thought_text": "some thought",
    }

    def fake_edit_message_text(chat_id: int, message_id: int, text: str) -> None:
        calls.append(("edit", chat_id, message_id, text))

    def fake_delete_message(chat_id: int, message_id: int) -> None:
        calls.append(("delete", chat_id, message_id))

    def fake_send_message(chat_id: int, text: str, **kwargs: Any) -> int:
        calls.append(("send_message", chat_id, text))
        return next(next_id)

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        calls.append(("send_text", chat_id, text, first_message_id))
        return [first_message_id or 100]

    poller._edit_message_text = fake_edit_message_text
    poller._delete_message = fake_delete_message
    poller._send_message = fake_send_message
    poller._send_text = fake_send_text

    worker._stream_turn(FakeFuture(), None, 50)

    send_text_calls = [c for c in calls if c[0] == "send_text"]
    assert len(send_text_calls) == 1
    # The thought placeholder is updated with _edit_message_text during streaming.
    edit_calls = [c for c in calls if c[0] == "edit"]
    assert len(edit_calls) == 1
    assert edit_calls[0][2] == 50
    assert "some thought" in edit_calls[0][3]
    # _send_text is only the final reply using the placeholder created after thinking.
    assert send_text_calls[0][3] == 100
    assert send_text_calls[0][2] == "final reply"


def test_stream_turn_keeps_thought_when_status_goes_idle(tmp_path: Path) -> None:
    """If the harness pops the active turn before the worker finalises,
    the last streamed thought is still preserved."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        stream_chunk_interval=0.0,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    class FakeFuture:
        _done = False

        def done(self) -> bool:
            if not self._done:
                self._done = True
                return False
            return True

        def result(self) -> dict[str, Any]:
            return {"reply": "final reply", "notice": None}

        def cancel(self) -> None:
            pass

    next_id = iter([100])
    calls: list[tuple[str, Any]] = []
    status_calls: list[bool] = []

    def fake_turn_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
        status_calls.append(True)
        if len(status_calls) == 1:
            return {
                "status": "running",
                "message_text": "",
                "thought_text": "some thought",
            }
        # On the final call after the future completes, the harness has
        # already popped the active turn and returns idle.
        return {"status": "idle"}

    def fake_send_message(chat_id: int, text: str, **kwargs: Any) -> int:
        calls.append(("send_message", chat_id, text))
        return next(next_id)

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        calls.append(("send_text", chat_id, text, first_message_id))
        return [first_message_id or 100]

    def fake_edit_message_text(chat_id: int, message_id: int, text: str) -> None:
        calls.append(("edit", chat_id, message_id, text))

    def fake_delete_message(chat_id: int, message_id: int) -> None:
        calls.append(("delete", chat_id, message_id))

    worker._harness_turn_status = fake_turn_status
    poller._edit_message_text = fake_edit_message_text
    poller._delete_message = fake_delete_message
    poller._send_message = fake_send_message
    poller._send_text = fake_send_text

    worker._stream_turn(FakeFuture(), None, 50)

    send_text_calls = [c for c in calls if c[0] == "send_text"]
    assert len(send_text_calls) == 1
    assert send_text_calls[0][3] == 100
    assert send_text_calls[0][2] == "final reply"

    edit_calls = [c for c in calls if c[0] == "edit"]
    assert len(edit_calls) == 1
    assert edit_calls[0][2] == 50
    assert "some thought" in edit_calls[0][3]


def test_format_thought_short_and_long() -> None:
    """Short thoughts keep the full text; long thoughts roll to the latest tail."""
    short = _format_thought("a quick thought")
    assert short == "Thinking...\na quick thought"

    long_text = "x" * 5000
    long = _format_thought(long_text)
    assert long.startswith("... (thinking continues)")
    assert long.endswith("x" * (4096 - len("... (thinking continues)") - 1))
    assert len(long) <= 4096


def test_stream_turn_thought_tail_updates(tmp_path: Path) -> None:
    """As the thought grows past the Telegram limit the placeholder keeps showing the tail."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        stream_chunk_interval=0.0,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    class FakeFuture:
        def __init__(self) -> None:
            self._ticks = 0

        def done(self) -> bool:
            self._ticks += 1
            return self._ticks > 3

        def result(self) -> dict[str, Any]:
            return {"reply": "final reply", "notice": None}

        def cancel(self) -> None:
            pass

    thoughts = ["a", "b" * 5000, "c" * 5000]
    edit_history: list[str] = []
    delete_history: list[int] = []

    def fake_turn_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if not thoughts:
            return {"status": "idle"}
        thought = thoughts.pop(0)
        return {
            "status": "running",
            "message_text": "",
            "thought_text": thought,
        }

    def fake_edit_message_text(chat_id: int, message_id: int, text: str) -> None:
        edit_history.append(text)

    def fake_delete_message(chat_id: int, message_id: int) -> None:
        delete_history.append(message_id)

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        return [first_message_id or 100]

    worker._harness_turn_status = fake_turn_status
    poller._edit_message_text = fake_edit_message_text
    poller._delete_message = fake_delete_message
    poller._send_message = lambda *args, **kwargs: 100
    poller._send_text = fake_send_text

    worker._stream_turn(FakeFuture(), None, 50)

    assert len(edit_history) == 3
    assert edit_history[0].endswith("a")
    assert edit_history[1].startswith("... (thinking continues)")
    assert edit_history[2].startswith("... (thinking continues)")
    assert delete_history == []


def test_stream_turn_empty_reply_deletes_placeholder(tmp_path: Path) -> None:
    """If the turn returns an empty reply, the placeholder is deleted rather than left hanging."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        stream_chunk_interval=0.0,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    class FakeFuture:
        def done(self) -> bool:
            return True

        def result(self) -> dict[str, Any]:
            return {"reply": "", "notice": "nothing here"}

        def cancel(self) -> None:
            pass

    deleted: list[int] = []
    sent: list[tuple[str, Any]] = []

    def fake_turn_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "idle"}

    def fake_delete_message(chat_id: int, message_id: int) -> None:
        deleted.append(message_id)

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        sent.append((text, reply_to_message_id))
        return [first_message_id or 100]

    worker._harness_turn_status = fake_turn_status
    poller._delete_message = fake_delete_message
    poller._send_text = fake_send_text
    poller._send_message = lambda *args, **kwargs: 100
    poller._edit_message_text = lambda *args, **kwargs: None

    worker._stream_turn(FakeFuture(), 42, None)

    assert deleted == [42]
    assert sent == [("System: nothing here", chat_input.message_id)]


def test_stream_turn_continuation_deletes_committed_message(tmp_path: Path) -> None:
    """If an intermediate message is committed and the turn continues, delete the
    committed message so the continuation does not leave a duplicate."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        stream_chunk_interval=0.0,
        intermediate_idle=0.0,
        intermediate_min_chars=1,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    class FakeFuture:
        calls = 0

        def done(self) -> bool:
            self.calls += 1
            return self.calls > 2

        def result(self) -> dict[str, Any]:
            return {"continuation": True, "reply": "first part then more."}

        def cancel(self) -> None:
            pass

    next_id = iter([100, 101])
    sent: list[tuple[str, Any]] = []
    deleted: list[int] = []
    statuses: list[dict[str, Any]] = [
        {"status": "running", "message_text": "first part.", "thought_text": ""},
        {"status": "running", "message_text": "first part.", "thought_text": ""},
    ]

    def fake_turn_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return statuses.pop(0)

    def fake_send_message(chat_id: int, text: str, **kwargs: Any) -> int:
        sent.append(("send_message", chat_id, text))
        return next(next_id)

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        sent.append(("send_text", chat_id, text, first_message_id))
        return [first_message_id or 100]

    def fake_delete_message(chat_id: int, message_id: int) -> None:
        deleted.append(message_id)

    worker._harness_turn_status = fake_turn_status
    poller._send_message = fake_send_message
    poller._send_text = fake_send_text
    poller._delete_message = fake_delete_message
    poller._edit_message_text = lambda *args, **kwargs: None
    poller._save_placeholder_state = lambda *args, **kwargs: None

    worker._stream_turn(FakeFuture(), None, None)

    # The first placeholder (100) is committed, then a second placeholder (101)
    # is created. On continuation the committed one is deleted, not the current
    # placeholder.
    assert 100 in deleted
    assert 101 not in deleted
    assert not any(c[0] == "send_text" for c in sent)


def _fake_response(status_code: int, body: dict[str, Any]) -> httpx.Response:
    request = httpx.Request("POST", "https://api.telegram.org/bottest/test")
    return httpx.Response(status_code, json=body, request=request)


def test_api_retries_429_and_respects_retry_after(tmp_path: Path, monkeypatch) -> None:
    """A 429 response with retry_after triggers a sleep and then a retry."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
        max_telegram_retries=2,
        min_telegram_interval=0.0,
    )
    now = [0.0]

    def fake_monotonic() -> float:
        return now[0]

    def fake_sleep(n: float) -> None:
        now[0] += n

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    calls: list[int] = []

    def fake_post(url: str, *, data: Any = None, **kwargs: Any) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return _fake_response(
                429,
                {
                    "ok": False,
                    "error_code": 429,
                    "description": "Too Many Requests: retry after 2",
                    "parameters": {"retry_after": 2.0},
                },
            )
        return _fake_response(200, {"ok": True, "result": {"message_id": 123}})

    poller.client.post = fake_post
    data = poller._api("sendMessage", chat_id=1, text="hi")

    assert data["ok"] is True
    assert data["result"]["message_id"] == 123
    assert len(calls) == 2
    assert now[0] == 2.0


def test_api_treats_message_not_modified_as_noop(tmp_path: Path) -> None:
    """A 400 MESSAGE_NOT_MODIFIED is treated as a no-op and returned as ok."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
        max_telegram_retries=1,
    )

    def fake_post(url: str, *, data: Any = None, **kwargs: Any) -> httpx.Response:
        return _fake_response(
            400,
            {
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: MESSAGE NOT MODIFIED",
            },
        )

    poller.client.post = fake_post
    data = poller._api("editMessageText", chat_id=1, message_id=2, text="same")

    assert data["ok"] is True
    assert data["result"] == {}


class _FakeConfigRuntime:
    def __init__(self) -> None:
        self.task = TaskConfig()
        self.waker = WakerConfig()
        self.timer = TimerConfig()
        self.notifications = NotificationsConfig()
        self.telegram = TelegramConfig()
        self.notifier = "noop"

    def _apply(self, current: Any, new: Any) -> None:
        for field in new.model_fields_set:
            setattr(current, field, getattr(new, field))

    def get_task_config(self) -> TaskConfig:
        return self.task

    def update_task_config(self, cfg: TaskConfig) -> str:
        self._apply(self.task, cfg)
        return "Task config updated"

    def get_waker_config(self) -> WakerConfig:
        return self.waker

    def update_waker_config(self, cfg: WakerConfig) -> str:
        self._apply(self.waker, cfg)
        return "Waker config updated"

    def get_timer_config(self) -> TimerConfig:
        return self.timer

    def update_timer_config(self, cfg: TimerConfig) -> str:
        self._apply(self.timer, cfg)
        return "Timer config updated"

    def get_notifications_config(self) -> NotificationsConfig:
        return self.notifications

    def update_notifications_config(self, cfg: NotificationsConfig) -> str:
        self._apply(self.notifications, cfg)
        self.notifier = f"notifier-{cfg.enabled}"
        return "Notifications config updated"

    def get_telegram_config(self) -> TelegramConfig:
        return self.telegram

    def update_telegram_config(self, cfg: TelegramConfig) -> str:
        self._apply(self.telegram, cfg)
        return "Telegram config updated"


def test_parse_config_value() -> None:
    assert TelegramPoller._parse_config_value("4") == 4
    assert TelegramPoller._parse_config_value("5.5") == 5.5
    assert TelegramPoller._parse_config_value("true") is True
    assert TelegramPoller._parse_config_value("false") is False
    assert TelegramPoller._parse_config_value("null") is None
    assert TelegramPoller._parse_config_value('["shell","noop"]') == ["shell", "noop"]
    assert TelegramPoller._parse_config_value("not-json") == "not-json"


def test_harness_config_direct_updates_task() -> None:
    runtime = _FakeConfigRuntime()
    poller = TelegramPoller(token="dummy", runtime=runtime)
    result = poller._harness_config(12345, "task workers=2 shell_timeout=120.0")
    assert "workers" in result
    assert runtime.task.workers == 2
    assert runtime.task.shell_timeout == 120.0


def test_harness_config_direct_partial_update() -> None:
    runtime = _FakeConfigRuntime()
    runtime.timer.enabled = False
    poller = TelegramPoller(token="dummy", runtime=runtime)
    poller._harness_config(12345, "timer interval_seconds=10.0")
    assert runtime.timer.enabled is False
    assert runtime.timer.interval_seconds == 10.0


def test_harness_config_direct_notifications_rebuilds_notifier() -> None:
    runtime = _FakeConfigRuntime()
    poller = TelegramPoller(token="dummy", runtime=runtime)
    poller._harness_config(
        12345, "notifications enabled=false webhook_url=https://example.com/hook"
    )
    assert runtime.notifications.enabled is False
    assert runtime.notifications.webhook_url == "https://example.com/hook"
    assert runtime.notifier == "notifier-False"


def test_harness_config_direct_updates_telegram() -> None:
    runtime = _FakeConfigRuntime()
    poller = TelegramPoller(token="dummy", runtime=runtime)
    result = poller._harness_config(12345, "telegram message_format=markdown_v2")
    assert "markdown_v2" in result
    assert runtime.telegram.message_format == "markdown_v2"


def test_harness_config_direct_invalid_section() -> None:
    runtime = _FakeConfigRuntime()
    poller = TelegramPoller(token="dummy", runtime=runtime)
    result = poller._harness_config(12345, "bad workers=2")
    assert "Unknown config section" in result


def test_harness_config_direct_invalid_pair() -> None:
    runtime = _FakeConfigRuntime()
    poller = TelegramPoller(token="dummy", runtime=runtime)
    result = poller._harness_config(12345, "task workers")
    assert "Invalid pair" in result


def test_harness_config_http_posts_to_endpoint() -> None:
    poller = TelegramPoller(token="dummy", harness_url="http://localhost")
    poller._local.client = _FakeClient(
        {
            "http://localhost/task/config": {
                "workers": 2,
                "shell_timeout": 120.0,
                "enabled_types": ["shell", "noop", "acp", "subagent"],
            },
        }
    )
    result = poller._harness_config(12345, "task workers=2")
    assert "workers" in result
    assert "120.0" in result


def test_stream_turn_heartbeat_wait_has_minimum_floor(tmp_path: Path, monkeypatch: Any) -> None:
    """The /turn long-poll wait must never drop below 5 s, even when a heartbeat is due."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        stream_chunk_interval=0.0,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    # Speed up the heartbeat interval so we hit the due path quickly.
    monkeypatch.setattr("diploid_agent.transport.telegram._HEARTBEAT_INTERVAL", 0.1)

    class FakeFuture:
        _ticks = 0

        def done(self) -> bool:
            self._ticks += 1
            return self._ticks > 4

        def result(self) -> dict[str, Any]:
            return {"reply": "", "notice": None}

        def cancel(self) -> None:
            pass

    waits: list[float] = []

    def fake_turn_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
        waits.append(kwargs.get("wait", 0.0))
        return {"status": "running", "message_text": "", "thought_text": ""}

    worker._harness_turn_status = fake_turn_status  # type: ignore[method-assign]
    poller._edit_message_text = lambda *args, **kwargs: None
    poller._delete_message = lambda *args, **kwargs: None
    poller._send_text = lambda *args, **kwargs: []
    poller._send_message = lambda *args, **kwargs: 100

    worker._stream_turn(FakeFuture(), 42, None)

    assert all(w >= 5.0 for w in waits), waits


def test_stream_turn_splits_intermediate_messages(tmp_path: Path, monkeypatch: Any) -> None:
    """When the streamed reply pauses after a complete sentence, it is committed
    as its own message and the final reply is sent below it without duplicating
    the committed text.
    """
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        intermediate_messages=True,
        intermediate_idle=0.0,
        intermediate_min_chars=1,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    statuses = [
        {"status": "running", "message_text": "I’ll check.", "thought_text": ""},
        {"status": "running", "message_text": "I’ll check.", "thought_text": ""},
        {
            "status": "running",
            "message_text": "I’ll check.\n\nDone, love.",
            "thought_text": "",
        },
    ]
    status_iter = iter(statuses)

    class FakeFuture:
        ticks = 0

        def done(self) -> bool:
            self.ticks += 1
            return self.ticks > len(statuses)

        def result(self) -> dict[str, Any]:
            return {"reply": "I’ll check.\n\nDone, love.", "notice": None}

        def cancel(self) -> None:
            pass

    next_ids = iter([101, 102, 103])
    sent_messages: list[tuple[int, str, dict[str, Any], int]] = []
    send_text_calls: list[tuple[int, str, int | None]] = []
    edit_history: list[tuple[int, str]] = []

    tick = [0.0]

    def fake_monotonic() -> float:
        tick[0] += 0.1
        return tick[0]

    def fake_send_message(chat_id: int, text: str, **kwargs: Any) -> int:
        message_id = next(next_ids)
        sent_messages.append((chat_id, text, kwargs, message_id))
        return message_id

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        send_text_calls.append((chat_id, text, first_message_id))
        return [first_message_id or 100]

    def fake_edit_message_text(chat_id: int, message_id: int, text: str) -> None:
        edit_history.append((message_id, text))

    monkeypatch.setattr("diploid_agent.transport.telegram.time.monotonic", fake_monotonic)
    worker._harness_turn_status = lambda *args, **kwargs: next(status_iter)  # type: ignore[method-assign]
    poller._send_message = fake_send_message  # type: ignore[method-assign]
    poller._send_text = fake_send_text  # type: ignore[method-assign]
    poller._edit_message_text = fake_edit_message_text  # type: ignore[method-assign]

    worker._stream_turn(FakeFuture(), 42, None)

    # The first chunk was committed as message 42, then a new placeholder (101)
    # was started below it.
    assert any(mid == 42 and "I’ll check." in txt for mid, txt in edit_history)
    assert any(item[3] == 101 and item[1] == "..." for item in sent_messages)
    reply_kwargs = next(item[2] for item in sent_messages if item[3] == 101)
    assert reply_kwargs.get("reply_to_message_id") == chat_input.message_id

    # The new placeholder was edited with the full text.
    assert any(
        mid == 101 and "I’ll check." in txt and "Done, love." in txt for mid, txt in edit_history
    )

    # The final reply was sliced to avoid duplicating the committed text.
    assert len(send_text_calls) == 1
    assert send_text_calls[0][1] == "Done, love."
    assert send_text_calls[0][2] == 101


def test_stream_turn_no_split_when_intermediate_messages_disabled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """With intermediate_messages disabled, the full reply edits the original
    placeholder even when the text pauses.
    """
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        intermediate_messages=False,
        intermediate_idle=0.0,
        intermediate_min_chars=1,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    statuses = [
        {"status": "running", "message_text": "I’ll check.", "thought_text": ""},
        {"status": "running", "message_text": "I’ll check.", "thought_text": ""},
        {
            "status": "running",
            "message_text": "I’ll check.\n\nDone, love.",
            "thought_text": "",
        },
    ]
    status_iter = iter(statuses)

    class FakeFuture:
        ticks = 0

        def done(self) -> bool:
            self.ticks += 1
            return self.ticks > len(statuses)

        def result(self) -> dict[str, Any]:
            return {"reply": "I’ll check.\n\nDone, love.", "notice": None}

        def cancel(self) -> None:
            pass

    send_text_calls: list[tuple[int, str, int | None]] = []
    sent_messages: list[tuple[int, str, dict[str, Any], int]] = []

    tick = [0.0]

    def fake_monotonic() -> float:
        tick[0] += 0.1
        return tick[0]

    def fake_send_message(chat_id: int, text: str, **kwargs: Any) -> int:
        sent_messages.append((chat_id, text, kwargs, 999))
        return 999

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        send_text_calls.append((chat_id, text, first_message_id))
        return [first_message_id or 100]

    monkeypatch.setattr("diploid_agent.transport.telegram.time.monotonic", fake_monotonic)
    worker._harness_turn_status = lambda *args, **kwargs: next(status_iter)  # type: ignore[method-assign]
    poller._send_message = fake_send_message  # type: ignore[method-assign]
    poller._send_text = fake_send_text  # type: ignore[method-assign]
    poller._edit_message_text = lambda *args, **kwargs: None  # type: ignore[method-assign]

    worker._stream_turn(FakeFuture(), 42, None)

    # No extra placeholder was sent and the full reply replaced message 42.
    assert not sent_messages
    assert len(send_text_calls) == 1
    assert send_text_calls[0][1] == "I’ll check.\n\nDone, love."
    assert send_text_calls[0][2] == 42


def test_stream_turn_no_duplicate_when_ask_block_added_after_commit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """If the model commits an intermediate sentence and then appends an ask
    block, the final reply must not duplicate the committed visible text."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        intermediate_messages=True,
        intermediate_idle=0.0,
        intermediate_min_chars=1,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    ask_block = '\n\n```ask\n{"question": "Shall I start?", "options": ["yes"]}\n```'
    final_raw = f"I’ll check.{ask_block}"

    statuses = [
        {"status": "running", "message_text": "I’ll check.", "thought_text": ""},
        {"status": "running", "message_text": "I’ll check.", "thought_text": ""},
        {"status": "running", "message_text": final_raw, "thought_text": ""},
    ]
    status_iter = iter(statuses)

    class FakeFuture:
        ticks = 0

        def done(self) -> bool:
            self.ticks += 1
            return self.ticks > len(statuses)

        def result(self) -> dict[str, Any]:
            return {"reply": final_raw, "notice": None}

        def cancel(self) -> None:
            pass

    sent_messages: list[tuple[int, str, dict[str, Any], int]] = []
    send_text_calls: list[tuple[int, str, int | None]] = []
    edit_history: list[tuple[int, str]] = []

    tick = [0.0]

    def fake_monotonic() -> float:
        tick[0] += 0.1
        return tick[0]

    def fake_send_message(chat_id: int, text: str, **kwargs: Any) -> int:
        sent_messages.append((chat_id, text, kwargs, 101))
        return 101

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        send_text_calls.append((chat_id, text, first_message_id))
        return [first_message_id or 100]

    def fake_edit_message_text(chat_id: int, message_id: int, text: str) -> None:
        edit_history.append((message_id, text))

    monkeypatch.setattr("diploid_agent.transport.telegram.time.monotonic", fake_monotonic)
    worker._harness_turn_status = lambda *args, **kwargs: next(status_iter)  # type: ignore[method-assign]
    poller._send_message = fake_send_message  # type: ignore[method-assign]
    poller._send_text = fake_send_text  # type: ignore[method-assign]
    poller._edit_message_text = fake_edit_message_text  # type: ignore[method-assign]

    worker._stream_turn(FakeFuture(), 42, None)

    # The first chunk was committed as message 42 and a new placeholder (101)
    # started below it.
    assert any(mid == 42 and "I’ll check." in txt for mid, txt in edit_history)
    assert any(item[1] == "..." for item in sent_messages)

    # No second commit happened: the ask block added no new *visible* content,
    # so only one new placeholder was sent.
    assert len([m for m in sent_messages if m[1] == "..."]) == 1

    # The final reply is the ask block suffix, not the duplicated visible text.
    assert len(send_text_calls) == 1
    assert "I’ll check." not in send_text_calls[0][1]
    assert send_text_calls[0][1].startswith("```ask")
    assert send_text_calls[0][2] == 101


def test_stream_turn_no_duplicate_when_final_reply_stripped_of_ask_block(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """If the committed text already included an ask block and the final
    raw reply was stripped to the visible text, the placeholder is deleted
    instead of re-sending the same visible content."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        intermediate_messages=True,
        intermediate_idle=0.0,
        intermediate_min_chars=1,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    ask_block = '\n\n```ask\n{"question": "Shall I start?", "options": ["yes"]}\n```'
    raw_text = f"I’ll check.{ask_block}"

    statuses = [
        {"status": "running", "message_text": raw_text, "thought_text": ""},
        {"status": "running", "message_text": raw_text, "thought_text": ""},
    ]
    status_iter = iter(statuses)

    class FakeFuture:
        ticks = 0

        def done(self) -> bool:
            self.ticks += 1
            return self.ticks > len(statuses)

        def result(self) -> dict[str, Any]:
            # The runtime returned only the visible text (ask block stripped).
            return {"reply": "I’ll check.", "notice": None}

        def cancel(self) -> None:
            pass

    sent_messages: list[tuple[int, str, dict[str, Any], int]] = []
    send_text_calls: list[tuple[int, str, int | None]] = []
    delete_history: list[int] = []

    tick = [0.0]

    def fake_monotonic() -> float:
        tick[0] += 0.1
        return tick[0]

    def fake_send_message(chat_id: int, text: str, **kwargs: Any) -> int:
        sent_messages.append((chat_id, text, kwargs, 101))
        return 101

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        send_text_calls.append((chat_id, text, first_message_id))
        return [first_message_id or 100]

    def fake_delete_message(chat_id: int, message_id: int) -> None:
        delete_history.append(message_id)

    monkeypatch.setattr("diploid_agent.transport.telegram.time.monotonic", fake_monotonic)
    worker._harness_turn_status = lambda *args, **kwargs: next(status_iter)  # type: ignore[method-assign]
    poller._send_message = fake_send_message  # type: ignore[method-assign]
    poller._send_text = fake_send_text  # type: ignore[method-assign]
    poller._edit_message_text = lambda *args, **kwargs: None  # type: ignore[method-assign]
    poller._delete_message = fake_delete_message  # type: ignore[method-assign]

    worker._stream_turn(FakeFuture(), 42, None)

    # The commit created one new placeholder below message 42.
    assert any(item[1] == "..." for item in sent_messages)

    # The final visible text is exactly what was already committed, so no
    # final sendMessage is issued; the dangling placeholder is deleted.
    assert not send_text_calls
    assert delete_history == [101]


def test_stream_turn_no_duplicate_when_stream_text_has_trailing_whitespace(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """If the streamed text has a trailing space at commit but the final
    reply does not, the final placeholder must be deleted instead of
    re-sending the same visible text."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        intermediate_messages=True,
        intermediate_idle=0.0,
        intermediate_min_chars=1,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    # Streaming includes a trailing space; final reply is the same but trimmed.
    statuses = [
        {"status": "running", "message_text": "I’ll check. ", "thought_text": ""},
        {"status": "running", "message_text": "I’ll check. ", "thought_text": ""},
    ]
    status_iter = iter(statuses)

    class FakeFuture:
        ticks = 0

        def done(self) -> bool:
            self.ticks += 1
            return self.ticks > len(statuses)

        def result(self) -> dict[str, Any]:
            return {"reply": "I’ll check.", "notice": None}

        def cancel(self) -> None:
            pass

    sent_messages: list[tuple[int, str, dict[str, Any], int]] = []
    send_text_calls: list[tuple[int, str, int | None]] = []
    delete_history: list[int] = []

    tick = [0.0]

    def fake_monotonic() -> float:
        tick[0] += 0.1
        return tick[0]

    def fake_send_message(chat_id: int, text: str, **kwargs: Any) -> int:
        sent_messages.append((chat_id, text, kwargs, 101))
        return 101

    def fake_send_text(
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        send_text_calls.append((chat_id, text, first_message_id))
        return [first_message_id or 100]

    def fake_delete_message(chat_id: int, message_id: int) -> None:
        delete_history.append(message_id)

    monkeypatch.setattr("diploid_agent.transport.telegram.time.monotonic", fake_monotonic)
    worker._harness_turn_status = lambda *args, **kwargs: next(status_iter)  # type: ignore[method-assign]
    poller._send_message = fake_send_message  # type: ignore[method-assign]
    poller._send_text = fake_send_text  # type: ignore[method-assign]
    poller._edit_message_text = lambda *args, **kwargs: None  # type: ignore[method-assign]
    poller._delete_message = fake_delete_message  # type: ignore[method-assign]

    worker._stream_turn(FakeFuture(), 42, None)

    # The streaming text was committed as message 42 and a new placeholder (101)
    # started below it.
    assert any(item[1] == "..." for item in sent_messages)

    # The final reply matches the committed visible text, so the final
    # placeholder is deleted and no second message is sent.
    assert not send_text_calls
    assert delete_history == [101]


def test_send_message_forwards_parse_mode(tmp_path: Path) -> None:
    """_send_message should pass parse_mode to the Telegram API."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, *, data: Any = None, **kwargs: Any) -> httpx.Response:
        calls.append(data or {})
        return _fake_response(200, {"ok": True, "result": {"message_id": 42}})

    poller.client.post = fake_post  # type: ignore[method-assign]
    msg_id = poller._send_message(123, "*bold*", parse_mode="MarkdownV2")

    assert msg_id == 42
    assert calls[0].get("parse_mode") == "MarkdownV2"


def test_send_message_fallback_on_parse_error(tmp_path: Path) -> None:
    """A 400 parse/markdown error should fall back to plain text."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, *, data: Any = None, **kwargs: Any) -> httpx.Response:
        calls.append(data or {})
        if len(calls) == 1:
            return _fake_response(
                400,
                {
                    "ok": False,
                    "error_code": 400,
                    "description": "Bad Request: can't parse message text",
                },
            )
        return _fake_response(200, {"ok": True, "result": {"message_id": 42}})

    poller.client.post = fake_post  # type: ignore[method-assign]
    msg_id = poller._send_message(123, "*bold*", parse_mode="MarkdownV2")

    assert msg_id == 42
    assert len(calls) == 2
    assert calls[0].get("parse_mode") == "MarkdownV2"
    assert calls[1].get("parse_mode") is None


def test_send_text_uses_markdown_v2_when_configured(tmp_path: Path, monkeypatch: Any) -> None:
    """_send_text should format replies as MarkdownV2 when configured."""
    runtime = _FakeConfigRuntime()
    runtime.telegram.message_format = "markdown_v2"
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
        runtime=runtime,
    )

    calls: list[dict[str, Any]] = []

    def fake_post(url: str, *, data: Any = None, **kwargs: Any) -> httpx.Response:
        calls.append(data or {})
        return _fake_response(200, {"ok": True, "result": {"message_id": 42}})

    poller.client.post = fake_post  # type: ignore[method-assign]

    sent = poller._send_text(123, "**bold**")

    assert sent == [42]
    assert calls[0].get("parse_mode") == "MarkdownV2"
    assert calls[0].get("text") == "*bold*"


def test_send_message_forwards_reply_markup(tmp_path: Path) -> None:
    """_send_message should forward a reply_markup JSON payload."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, *, data: Any = None, **kwargs: Any) -> httpx.Response:
        calls.append(data or {})
        return _fake_response(200, {"ok": True, "result": {"message_id": 42}})

    poller.client.post = fake_post  # type: ignore[method-assign]
    markup = {"keyboard": [[{"text": "A"}]], "resize_keyboard": True}
    msg_id = poller._send_message(123, "Pick one", reply_markup=markup)

    assert msg_id == 42
    assert calls[0].get("reply_markup") == json.dumps(markup)


def test_send_text_extracts_ask_block_and_sends_keyboard(tmp_path: Path) -> None:
    """A reply with a ```ask block should be sent as a keyboard question."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, *, data: Any = None, **kwargs: Any) -> httpx.Response:
        calls.append(data or {})
        return _fake_response(200, {"ok": True, "result": {"message_id": 42}})

    poller.client.post = fake_post  # type: ignore[method-assign]

    text = (
        "Which file should I edit?\n\n"
        "```ask\n"
        '{"question": "Which file should I edit?", "options": ["a.py", "b.py"]}\n'
        "```"
    )
    sent = poller._send_text(123, text)

    assert sent == [42]
    assert "Which file should I edit?" in calls[0].get("text", "")
    assert "```ask" not in calls[0].get("text", "")
    assert "a.py" not in calls[0].get("text", "")
    reply_markup = json.loads(calls[0].get("reply_markup", "{}"))
    assert reply_markup["keyboard"] == [[{"text": "a.py"}], [{"text": "b.py"}]]


def test_save_and_load_pending_question(tmp_path: Path) -> None:
    """Pending questions can be saved, loaded, and removed."""
    from diploid_agent.transport.interactive import AskBlock

    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    ask = AskBlock(question="Which file?", options=["a.py", "b.py"])
    poller._save_pending_question(123, ask, 42)

    loaded = poller._load_pending_question(123)
    assert loaded is not None
    assert loaded["question"] == "Which file?"
    assert loaded["options"] == ["a.py", "b.py"]
    assert loaded["message_id"] == 42

    poller._remove_pending_question(123)
    assert poller._load_pending_question(123) is None


def test_maybe_answer_pending_question(tmp_path: Path) -> None:
    """A button-press answer is rewritten into a contextual message."""
    from diploid_agent.transport.interactive import AskBlock

    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    poller._save_pending_question(
        123,
        AskBlock(question="Which file?", options=["a.py", "b.py"]),
        42,
    )

    chat_input = ChatInput(chat_id=123, message_id=2, text="a.py")
    answered = poller._maybe_answer_pending_question(chat_input)
    assert "Which file?" in answered.text
    assert "a.py" in answered.text
    assert answered.reply_to == "Which file?"
    assert answered.reply_to_is_bot is True
    assert answered.reply_to_message_id == 42

    # A non-option should clear the pending question and not rewrite.
    poller._save_pending_question(
        123,
        AskBlock(question="Which file?", options=["a.py", "b.py"]),
        42,
    )
    chat_input = ChatInput(chat_id=123, message_id=3, text="something else")
    unchanged = poller._maybe_answer_pending_question(chat_input)
    assert unchanged.text == "something else"
    assert poller._load_pending_question(123) is None


def test_stream_turn_strips_ask_block(tmp_path: Path, monkeypatch: Any) -> None:
    """The streaming placeholder text does not contain the ```ask fence."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    poller._stream_thoughts[123] = False

    edited: list[str] = []

    def fake_edit(chat_id: int, message_id: int, text: str, *, parse_mode: Any = None) -> None:
        edited.append(text)

    def fake_delete(chat_id: int, message_id: int) -> bool:
        return True

    poller._edit_message_text = fake_edit  # type: ignore[method-assign]
    poller._delete_message = fake_delete  # type: ignore[method-assign]

    chat_input = ChatInput(chat_id=123, message_id=1, text="hi")
    worker = TurnWorker(poller, chat_input)

    status_calls = 0

    def fake_turn_status(wait: float = 0.0) -> dict[str, Any]:
        nonlocal status_calls
        status_calls += 1
        if status_calls == 1:
            return {
                "status": "running",
                "message_text": (
                    'Which file?\n\n```ask\n{"question": "Which file?", "options": ["a.py"]}\n```'
                ),
            }
        return {"status": "idle"}

    worker._harness_turn_status = fake_turn_status  # type: ignore[method-assign]

    class FakeFuture:
        def __init__(self) -> None:
            self._checks = 0
            self._result: dict[str, Any] = {"reply": "", "notice": None}

        def done(self) -> bool:
            self._checks += 1
            return self._checks > 1

        def result(self) -> dict[str, Any]:
            return self._result

        def cancel(self) -> None:
            pass

    worker._stream_turn(FakeFuture(), 1, None)

    assert edited
    assert "```ask" not in edited[0]
    assert "a.py" not in edited[0]
    assert "Which file?" in edited[0]


class _FailingClient:
    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    def post(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


def test_harness_subagent_status_with_subagents() -> None:
    poller = TelegramPoller(token="dummy", harness_url="http://localhost")
    poller._local.client = _FakeClient(
        {
            "http://localhost/subagents/12345": {
                "chat_id": "12345",
                "subagents": [
                    {
                        "dispatch_id": "d-1",
                        "status": "running",
                        "summary": "Working on it",
                        "started_at": 0.0,
                        "finished_at": None,
                    },
                    {
                        "dispatch_id": "d-2",
                        "status": "completed",
                        "summary": "Done",
                        "started_at": 0.0,
                        "finished_at": 1.0,
                    },
                ],
            },
        }
    )
    result = poller._harness_subagent_status(12345)
    assert "running: d-1" in result
    assert "completed: d-2" in result
    assert "Working on it" in result
    assert "Done" in result
    assert "00:00:00" in result


def test_harness_subagent_status_empty() -> None:
    poller = TelegramPoller(token="dummy", harness_url="http://localhost")
    poller._local.client = _FakeClient(
        {
            "http://localhost/subagents/12345": {
                "chat_id": "12345",
                "subagents": [],
            },
        }
    )
    result = poller._harness_subagent_status(12345)
    assert result == "No background subagents for this chat."


def test_harness_subagent_status_error() -> None:
    poller = TelegramPoller(token="dummy", harness_url="http://localhost")
    poller._local.client = _FailingClient()
    result = poller._harness_subagent_status(12345)
    assert "Sorry" in result


def test_handle_update_routes_subagents_command(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> None:
        sent.append((chat_id, text, reply_to_message_id))

    poller._send_text = fake_send
    poller._harness_subagent_status = lambda chat_id: "Subagent status"
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/subagents",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Subagent status", 1)]


class _FakeDeliveryRuntime:
    """Runtime stub for delivery and queue tests."""

    def __init__(self, outbox: list[ChatResult] | None = None) -> None:
        self._outbox = outbox or []
        self.outbox_calls: list[tuple[str, float]] = []
        self.process_calls: list[ChatInput] = []
        self.config = _FakeConfigRuntime()

    def get_config(self) -> dict[str, Any]:
        return {
            "harness": {
                "notifications": {"enabled": True, "outbox_delivery": True},
                "telegram": self.config.telegram.model_dump(mode="json"),
            }
        }

    def outbox_pop(self, chat_id: str | None = None, wait: float = 0.0) -> ChatResult | None:
        self.outbox_calls.append((chat_id or "", wait))
        if self._outbox:
            return self._outbox.pop(0)
        return None

    def process(
        self,
        chat_id: str,
        message: str,
        *,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        notify: bool = False,
    ) -> ChatResult:
        self.process_calls.append(
            ChatInput(
                chat_id=int(chat_id),
                message_id=0,
                text=message,
                reply_to=reply_to,
                reply_to_is_bot=reply_to_is_bot,
                reply_to_message_id=reply_to_message_id,
            )
        )
        return ChatResult(reply=f"reply: {message}", turn_number=1)

    def stop(self, chat_id: str) -> None:
        pass


def test_delivery_worker_sends_outbox_result(tmp_path: Path) -> None:
    """A DeliveryWorker long-polls the outbox and sends new results."""
    runtime = _FakeDeliveryRuntime(
        outbox=[
            ChatResult(reply="outbox reply", turn_number=1),
        ]
    )
    poller = TelegramPoller(
        token="dummy",
        runtime=runtime,  # type: ignore[arg-type]
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[tuple[int, str, int | None]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> list[int]:
        sent.append((chat_id, text, reply_to_message_id))
        return [100]

    poller._send_text = fake_send  # type: ignore[method-assign]
    poller._last_user_message_ids[12345] = 50

    worker = DeliveryWorker(poller, 12345)
    worker.start()
    time.sleep(0.2)
    worker.stop()
    worker.join(timeout=2.0)

    assert sent == [(12345, "outbox reply", 50)]


def test_turn_worker_queued_input_is_processed(tmp_path: Path) -> None:
    """A second message sent while a turn is running is queued and processed next."""
    slow_runtime = _FakeDeliveryRuntime()
    slow_calls: list[str] = []

    def slow_process(chat_id: str, message: str, **kwargs: Any) -> ChatResult:
        slow_calls.append(message)
        time.sleep(0.2)
        return ChatResult(reply=f"reply: {message}", turn_number=1)

    slow_runtime.process = slow_process  # type: ignore[method-assign]
    slow_poller = TelegramPoller(
        token="dummy",
        runtime=slow_runtime,  # type: ignore[arg-type]
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[str] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> list[int]:
        sent.append(text)
        return [100]

    slow_poller._send_text = fake_send  # type: ignore[method-assign]
    slow_poller._send_message = lambda *args, **kwargs: 100  # type: ignore[method-assign]
    slow_poller._edit_message_text = lambda *args, **kwargs: None  # type: ignore[method-assign]
    slow_poller._delete_message = lambda *args, **kwargs: None  # type: ignore[method-assign]
    slow_poller._register_message_ids = lambda *args, **kwargs: None  # type: ignore[method-assign]

    TurnWorker._harness_turn_status = (  # type: ignore[method-assign]
        lambda self, *args, **kwargs: {"status": "idle"}
    )

    worker = TurnWorker(slow_poller, ChatInput(chat_id=123, message_id=1, text="first"))
    worker.start()
    time.sleep(0.05)

    # Queue a second message while the first is running.
    second = ChatInput(chat_id=123, message_id=2, text="second")
    worker.steer(second)

    worker.join(timeout=2.0)

    assert slow_calls == ["first", "second"]


def test_handle_update_starts_worker_and_queues_messages(tmp_path: Path) -> None:
    """Two messages arriving in quick succession are queued and processed in order."""
    runtime = _FakeDeliveryRuntime()
    poller = TelegramPoller(
        token="dummy",
        runtime=runtime,  # type: ignore[arg-type]
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[str] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        first_message_id: int | None = None,
    ) -> list[int]:
        sent.append(text)
        return [100]

    poller._send_text = fake_send  # type: ignore[method-assign]
    poller._send_message = lambda *args, **kwargs: 100  # type: ignore[method-assign]
    poller._edit_message_text = lambda *args, **kwargs: None  # type: ignore[method-assign]
    poller._delete_message = lambda *args, **kwargs: None  # type: ignore[method-assign]
    poller._register_message_ids = lambda *args, **kwargs: None  # type: ignore[method-assign]

    # Patch the worker harness to be deterministic and fast.
    def fake_harness_chat(self: TurnWorker, chat_input: ChatInput) -> dict[str, Any]:
        time.sleep(0.05)
        return {"reply": f"reply: {chat_input.text}", "turn_number": 1}

    def fake_harness_turn_status(self: TurnWorker, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "idle"}

    TurnWorker._harness_turn_status = fake_harness_turn_status  # type: ignore[method-assign]
    TurnWorker._harness_chat = fake_harness_chat  # type: ignore[method-assign]

    poller._handle_update(
        _update(
            message={
                "message_id": 1,
                "chat": {"id": 123, "type": "private"},
                "from": {"id": 1, "is_bot": False},
                "text": "first",
            }
        )
    )

    # Give the worker a chance to start, then send a second message.
    time.sleep(0.02)
    poller._handle_update(
        _update(
            message={
                "message_id": 2,
                "chat": {"id": 123, "type": "private"},
                "from": {"id": 1, "is_bot": False},
                "text": "second",
            }
        )
    )

    # Wait for both to be processed and the worker to finish.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with poller._worker_lock:
            if not poller._active_workers.get(123) and not poller._pending_inputs.get(123):
                break
        time.sleep(0.05)

    assert "reply: first" in sent
    assert "reply: second" in sent
