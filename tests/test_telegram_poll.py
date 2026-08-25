"""Tests for the Telegram long-polling ingress."""

import json
import time
from pathlib import Path
from typing import Any

import httpx

from devin_fleet_harness.config import (
    NotificationsConfig,
    TaskConfig,
    TimerConfig,
    WakerConfig,
)
from devin_fleet_harness.telegram_poll import ChatInput, TelegramPoller, TurnWorker
from devin_fleet_harness.transport.telegram import _format_thought


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
                "enabled_types": ["shell", "noop", "acp"],
            },
        }
    )
    result = poller._harness_config(12345, "task workers=2")
    assert "workers" in result
    assert "120.0" in result
