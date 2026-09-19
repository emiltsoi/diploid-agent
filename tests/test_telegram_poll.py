"""Tests for the Telegram long-polling ingress."""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import httpx

from diploid_agent.config import (
    NotificationsConfig,
    TaskConfig,
    TelegramConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.models import ChatResult
from diploid_agent.telegram_poll import (
    ChatInput,
    TelegramAttachment,
    TelegramPoller,
    TurnWorker,
)
from diploid_agent.transport.interactive import AskBlock, FileRef
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

    def patch(self, url: str, **kwargs: Any) -> _FakeResponse:
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


def test_handle_update_sub_command_arity_and_usage(tmp_path: Path) -> None:
    """Sub-command table: arity checks, usage fallbacks, first-token vs whole-rest."""
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
    poller._harness_plugin_enable = lambda chat_id, name, enabled: f"plugin {name} {enabled}"
    poller._harness_plugin_list = lambda chat_id: "plugins"
    poller._harness_mcp_enable = lambda chat_id, name: f"mcp {name}"
    poller._harness_skill_create = lambda chat_id, name, content: f"skill {name}: {content}"

    def send(text: str) -> None:
        update = _update(
            message={
                "message_id": 7,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 1, "is_bot": False},
                "text": text,
            }
        )
        poller._handle_update(update)

    send("/plugin enable x trailing words")
    send("/plugin enable")
    send("/plugin")
    send("/mcp enable two words")
    send("/mcp list extra")
    send("/skill create name markdown body here")
    send("/skill create")
    assert sent == [
        # plugin takes the first token only; extras are ignored.
        (12345, "plugin x True", 7),
        (12345, "Usage: /plugin list | /plugin enable <name> | /plugin disable <name> | /plugin reload <name>", 7),
        (12345, "plugins", 7),
        # mcp passes the rest through as one name, like the old split(None, 1).
        (12345, "mcp two words", 7),
        # extra tokens on a zero-arg sub-command fall back to usage.
        (12345, "Usage: /mcp list | /mcp enable <name> | /mcp disable <name>", 7),
        # create consumes name + remaining content.
        (12345, "skill name: markdown body here", 7),
        (12345, "Usage: /skill list | /skill enable <name> | /skill disable <name> | /skill create <name> <markdown>", 7),
    ]


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
    assert len(send_text_calls) == 2
    # The thought placeholder is updated with _edit_message_text during streaming.
    edit_calls = [c for c in calls if c[0] == "edit"]
    assert len(edit_calls) == 1
    assert edit_calls[0][2] == 50
    assert "some thought" in edit_calls[0][3]
    # The full thought is sent as multi-part messages, then the final reply.
    assert send_text_calls[0][2] == "Thinking...\nsome thought"
    assert send_text_calls[1][2] == "final reply"


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
    assert len(send_text_calls) == 2
    assert send_text_calls[0][2] == "Thinking...\nsome thought"
    assert send_text_calls[1][2] == "final reply"

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
    # The live placeholder is deleted and the full thought is sent in multi-part
    # messages once the turn completes.
    assert delete_history == [50]


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
    from diploid_agent.transport.telegram import sender as sender_module

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

    fake_time = type(time)("fake_time", "time stub for tests")
    fake_time.monotonic = fake_monotonic
    fake_time.sleep = fake_sleep
    monkeypatch.setattr(sender_module, "time", fake_time)

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


def test_harness_config_http_telegram_uses_patch_config() -> None:
    """No /telegram/config route exists — telegram goes through PATCH /config."""
    poller = TelegramPoller(token="dummy", harness_url="http://localhost")
    calls: list[tuple[str, str, dict[str, Any]]] = []

    class _PatchRecorder(_FakeClient):
        def patch(self, url: str, **kwargs: Any) -> _FakeResponse:
            calls.append(("PATCH", url, kwargs.get("json", {})))
            return _FakeResponse({"ok": True})

        def post(self, url: str, **kwargs: Any) -> _FakeResponse:
            calls.append(("POST", url, kwargs.get("json", {})))
            return _FakeResponse({})

    poller._local.client = _PatchRecorder({})
    result = poller._harness_config(12345, "telegram message_format=markdown_v2")
    assert calls == [
        ("PATCH", "http://localhost/config", {"telegram": {"message_format": "markdown_v2"}})
    ]
    assert "restart" in result


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


def test_stream_turn_floors_poll_rate_on_instant_replies(tmp_path: Path, monkeypatch: Any) -> None:
    """A stopped turn makes the harness return /turn instantly forever; the
    worker must pace itself instead of spinning into a hot poll loop, and
    must exit on _should_stop instead of waiting for chat_future forever."""
    from diploid_agent.transport.telegram import workers as workers_mod

    monkeypatch.setattr(workers_mod, "_MIN_POLL_INTERVAL", 0.05)
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        stream_chunk_interval=0.0,
    )
    chat_input = ChatInput(chat_id=12345, message_id=1, text="hello")
    worker = TurnWorker(poller, chat_input)

    calls = 0

    def fake_turn_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls >= 5:
            worker._should_stop.set()
        return {
            "status": "running",
            "stopped": True,
            "message_text": "",
            "thought_text": "",
        }

    class PendingFuture:
        def done(self) -> bool:
            return False

        def result(self, timeout: float | None = None) -> dict[str, Any]:
            raise TimeoutError()

        def cancel(self) -> None:
            pass

    worker._harness_turn_status = fake_turn_status  # type: ignore[method-assign]
    poller._edit_message_text = lambda *args, **kwargs: None
    poller._delete_message = lambda *args, **kwargs: None
    poller._send_text = lambda *args, **kwargs: []
    poller._send_message = lambda *args, **kwargs: 100

    start = time.monotonic()
    result = worker._stream_turn(PendingFuture(), None, None)
    elapsed = time.monotonic() - start

    assert calls == 5
    # Without the floor this would be hundreds of calls per second.
    assert elapsed >= 4 * 0.05
    assert result["notice"].startswith("Turn stopped")


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
            "message_text": "I’ll check.\n\nDone, thanks.",
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
            return {"reply": "I’ll check.\n\nDone, thanks.", "notice": None}

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

    # The new placeholder was edited with only the uncommitted tail — the
    # committed prefix is not duplicated into the second message.
    assert any(
        mid == 101 and "Done, thanks." in txt and "I’ll check." not in txt
        for mid, txt in edit_history
    )

    # The final reply was sliced to avoid duplicating the committed text.
    assert len(send_text_calls) == 1
    assert send_text_calls[0][1] == "Done, thanks."
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
            "message_text": "I’ll check.\n\nDone, thanks.",
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
            return {"reply": "I’ll check.\n\nDone, thanks.", "notice": None}

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
    assert send_text_calls[0][1] == "I’ll check.\n\nDone, thanks."
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
    """A reply with a ```ask block should be sent as an inline keyboard question."""
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
    assert reply_markup["inline_keyboard"] == [
        [{"text": "a.py", "callback_data": "ask_0"}],
        [{"text": "b.py", "callback_data": "ask_1"}],
        [{"text": "Cancel", "callback_data": "ask_cancel"}],
    ]


def test_send_text_cancellable_ask_block(tmp_path: Path) -> None:
    """A cancellable ask block appends a cancel row to the inline keyboard."""
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
        "Which file?\n\n"
        "```ask\n"
        '{"question": "Which file?", "options": ["a.py", "b.py"], "cancellable": true}\n'
        "```"
    )
    sent = poller._send_text(123, text)

    assert sent == [42]
    reply_markup = json.loads(calls[0].get("reply_markup", "{}"))
    assert reply_markup["inline_keyboard"] == [
        [{"text": "a.py", "callback_data": "ask_0"}],
        [{"text": "b.py", "callback_data": "ask_1"}],
        [{"text": "Cancel", "callback_data": "ask_cancel"}],
    ]


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


def test_cancel_pending_question(tmp_path: Path) -> None:
    """A cancellable question's cancel button is swallowed and removes the keyboard."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )

    sent: list[tuple[int, str, dict[str, Any] | None]] = []
    deleted: list[tuple[int, int]] = []

    def fake_send(
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> int | None:
        sent.append((chat_id, text, reply_markup))
        return 100

    def fake_delete(chat_id: int, message_id: int) -> None:
        deleted.append((chat_id, message_id))

    poller._send_message = fake_send  # type: ignore[method-assign]
    poller._delete_message = fake_delete  # type: ignore[method-assign]

    poller._save_pending_question(
        123,
        AskBlock(
            question="Which file?",
            options=["a.py", "b.py"],
            cancellable=True,
        ),
        42,
    )

    chat_input = ChatInput(chat_id=123, message_id=2, text="Cancel")
    result = poller._maybe_answer_pending_question(chat_input)
    assert result is None
    assert poller._load_pending_question(123) is None
    assert len(sent) == 1
    assert sent[0][0] == 123
    assert sent[0][1] == "Cancelled."
    assert sent[0][2] == {"remove_keyboard": True}
    assert deleted == [(123, 2)]


def test_parse_update_callback_query() -> None:
    """A callback query from an inline keyboard is parsed as a ChatInput."""
    update = _update(
        callback_query={
            "id": "cq1",
            "from": {"id": 1, "is_bot": False},
            "message": {
                "message_id": 42,
                "chat": {"id": 12345, "type": "private"},
                "text": "Which file?",
            },
            "data": "ask_0",
        }
    )
    parsed = TelegramPoller._parse_update(update)
    assert parsed is not None
    assert parsed.chat_id == 12345
    assert parsed.message_id == 42
    assert parsed.text == "ask_0"
    assert parsed.callback_query_id == "cq1"
    assert parsed.reply_to == "Which file?"
    assert parsed.reply_to_is_bot is True
    assert parsed.reply_to_message_id == 42


def test_parse_update_skips_bot_callback_query() -> None:
    """Callback queries from other bots are ignored."""
    update = _update(
        callback_query={
            "id": "cq1",
            "from": {"id": 0, "is_bot": True},
            "message": {
                "message_id": 42,
                "chat": {"id": 12345, "type": "private"},
                "text": "Which file?",
            },
            "data": "ask_0",
        }
    )
    assert TelegramPoller._parse_update(update) is None


def test_answer_ask_callback(tmp_path: Path) -> None:
    """A valid inline option callback is translated back to the option text."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    answered: list[str] = []
    cleared: list[tuple[int, int]] = []

    def fake_answer(callback_query_id: str) -> None:
        answered.append(callback_query_id)

    def fake_clear(chat_id: int, message_id: int) -> None:
        cleared.append((chat_id, message_id))

    poller._answer_callback_query = fake_answer  # type: ignore[method-assign]
    poller._clear_inline_keyboard = fake_clear  # type: ignore[method-assign]

    poller._save_pending_question(
        123,
        AskBlock(question="Which file?", options=["a.py", "b.py"]),
        42,
    )

    chat_input = ChatInput(
        chat_id=123,
        message_id=42,
        text="ask_0",
        callback_query_id="cq1",
    )
    result = poller._maybe_answer_pending_question(chat_input)

    assert result is not None
    assert "Which file?" in result.text
    assert "a.py" in result.text
    assert result.reply_to == "Which file?"
    assert result.reply_to_message_id == 42
    assert result.callback_query_id is None
    assert poller._load_pending_question(123) is None
    assert answered == ["cq1"]
    assert cleared == [(123, 42)]


def test_cancel_ask_callback(tmp_path: Path) -> None:
    """A cancellable inline cancel callback edits the question and starts no turn."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    answered: list[str] = []
    cleared: list[tuple[int, int]] = []
    edited: list[tuple[int, int, str]] = []

    def fake_answer(callback_query_id: str) -> None:
        answered.append(callback_query_id)

    def fake_clear(chat_id: int, message_id: int) -> None:
        cleared.append((chat_id, message_id))

    def fake_edit(chat_id: int, message_id: int, text: str, *, parse_mode: Any = None) -> bool:
        edited.append((chat_id, message_id, text))
        return True

    poller._answer_callback_query = fake_answer  # type: ignore[method-assign]
    poller._clear_inline_keyboard = fake_clear  # type: ignore[method-assign]
    poller._edit_message_text = fake_edit  # type: ignore[method-assign]

    poller._save_pending_question(
        123,
        AskBlock(
            question="Which file?",
            options=["a.py", "b.py"],
            cancellable=True,
        ),
        42,
    )

    chat_input = ChatInput(
        chat_id=123,
        message_id=42,
        text="ask_cancel",
        callback_query_id="cq1",
    )
    result = poller._maybe_answer_pending_question(chat_input)

    assert result is None
    assert poller._load_pending_question(123) is None
    assert answered == ["cq1"]
    assert edited == [(123, 42, "Cancelled.")]
    assert cleared == [(123, 42)]


def test_stale_ask_callback_ignored(tmp_path: Path) -> None:
    """A callback for an unknown question is ignored and the keyboard is cleared."""
    poller = TelegramPoller(
        token="dummy",
        state_dir=tmp_path / ".poller-placeholders",
    )
    answered: list[str] = []
    cleared: list[tuple[int, int]] = []

    def fake_answer(callback_query_id: str) -> None:
        answered.append(callback_query_id)

    def fake_clear(chat_id: int, message_id: int) -> None:
        cleared.append((chat_id, message_id))

    poller._answer_callback_query = fake_answer  # type: ignore[method-assign]
    poller._clear_inline_keyboard = fake_clear  # type: ignore[method-assign]

    chat_input = ChatInput(
        chat_id=123,
        message_id=42,
        text="ask_0",
        callback_query_id="cq1",
    )
    result = poller._maybe_answer_pending_question(chat_input)

    assert result is None
    assert answered == ["cq1"]
    assert cleared == [(123, 42)]


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

    def outbox_pop(
        self,
        chat_id: str | None = None,
        wait: float = 0.0,
        return_chat_id: bool = False,
    ) -> ChatResult | tuple[str, ChatResult] | None:
        self.outbox_calls.append((chat_id or "", wait, return_chat_id))
        if not self._outbox:
            return None
        result = self._outbox.pop(0)
        if return_chat_id:
            return (chat_id or "12345", result)
        return result

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


def test_handle_update_routes_graceful_restart_command(tmp_path: Path) -> None:
    """/graceful-restart without an argument uses the runtime persona name."""
    runtime = _FakeConfigRuntime()
    runtime.config = SimpleNamespace(persona=SimpleNamespace(name="test-pilot"))
    poller = TelegramPoller(
        token="dummy",
        runtime=runtime,
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
    poller._harness_graceful_restart = lambda chat_id, service: {
        "reply": f"Restarting {service}",
        "notice": None,
    }
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/graceful-restart",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Restarting test-pilot.service", 1)]


def test_handle_update_routes_graceful_restart_with_explicit_service(tmp_path: Path) -> None:
    """/graceful-restart with an explicit service name uses that name."""
    runtime = _FakeConfigRuntime()
    runtime.config = SimpleNamespace(persona=SimpleNamespace(name="test-pilot"))
    poller = TelegramPoller(
        token="dummy",
        runtime=runtime,
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
    poller._harness_graceful_restart = lambda chat_id, service: {
        "reply": f"Restarting {service}",
        "notice": None,
    }
    update = _update(
        message={
            "message_id": 1,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "text": "/graceful-restart my-service.service",
        }
    )
    poller._handle_update(update)
    assert sent == [(12345, "Restarting my-service.service", 1)]


# ---------------------------------------------------------------------------
# Attachment transfer
# ---------------------------------------------------------------------------


def test_parse_update_photo_uses_largest_variant() -> None:
    update = _update(
        message={
            "message_id": 7,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "caption": "look at this",
            "photo": [
                {"file_id": "small", "width": 90, "height": 90, "file_size": 100},
                {"file_id": "big", "width": 800, "height": 800, "file_size": 5000},
                {"file_id": "mid", "width": 320, "height": 320, "file_size": 800},
            ],
        }
    )
    parsed = TelegramPoller._parse_update(update)
    assert isinstance(parsed, ChatInput)
    assert parsed.text == "look at this"
    assert len(parsed.attachments) == 1
    att = parsed.attachments[0]
    assert att.kind == "photo"
    assert att.file_id == "big"
    assert att.file_size == 5000


def test_parse_update_captionless_media_still_parsed() -> None:
    update = _update(
        message={
            "message_id": 8,
            "chat": {"id": 12345, "type": "private"},
            "from": {"id": 1, "is_bot": False},
            "document": {
                "file_id": "doc1",
                "file_name": "notes.txt",
                "mime_type": "text/plain",
                "file_size": 42,
            },
        }
    )
    parsed = TelegramPoller._parse_update(update)
    assert isinstance(parsed, ChatInput)
    assert parsed.text == ""
    assert len(parsed.attachments) == 1
    att = parsed.attachments[0]
    assert att.kind == "document"
    assert att.file_name == "notes.txt"
    assert att.mime_type == "text/plain"


def test_parse_update_all_media_keys() -> None:
    for key in ("voice", "video", "video_note", "sticker", "animation", "audio"):
        update = _update(
            message={
                "message_id": 9,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 1, "is_bot": False},
                key: {"file_id": f"{key}-id"},
            }
        )
        parsed = TelegramPoller._parse_update(update)
        assert parsed is not None, key
        assert parsed.attachments[0].kind == key


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, size: int) -> Any:
        yield from self._chunks


class _AttachmentClient:
    """Fake httpx client: post() answers getFile, stream() serves file bytes."""

    def __init__(
        self,
        file_path: str = "photos/f.jpg",
        file_size: int = 4,
        chunks: list[bytes] | None = None,
    ) -> None:
        self._file_path = file_path
        self._file_size = file_size
        self._chunks = chunks if chunks is not None else [b"data"]
        self.streamed_urls: list[str] = []

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {
                "ok": True,
                "result": {"file_path": self._file_path, "file_size": self._file_size},
            }
        )

    def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        self.streamed_urls.append(url)
        return _FakeStreamResponse(self._chunks)


def _attachment_poller(tmp_path: Path, client: Any) -> TelegramPoller:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        sessions_root=tmp_path / "sessions",
    )
    poller._local.client = client
    return poller


def test_ingest_saves_attachment_and_annotates(tmp_path: Path) -> None:
    client = _AttachmentClient(file_path="documents/notes.txt", chunks=[b"hello file"])
    poller = _attachment_poller(tmp_path, client)
    ci = ChatInput(
        chat_id=12345,
        message_id=11,
        text="read this",
        attachments=(
            TelegramAttachment(
                kind="document",
                file_id="doc1",
                file_name="notes.txt",
                mime_type="text/plain",
            ),
        ),
    )
    out = poller._ingest_attachments(ci)
    dest = tmp_path / "sessions" / "12345" / "inbox" / "11-notes.txt"
    assert dest.read_bytes() == b"hello file"
    assert "read this\n[attachment saved: inbox/11-notes.txt (document, text/plain)]" == out.text


def test_ingest_captionless_media_text_is_annotation(tmp_path: Path) -> None:
    client = _AttachmentClient(file_path="photos/f.jpg", chunks=[b"\xff"])
    poller = _attachment_poller(tmp_path, client)
    ci = ChatInput(
        chat_id=12345,
        message_id=12,
        text="",
        attachments=(TelegramAttachment(kind="photo", file_id="p1"),),
    )
    out = poller._ingest_attachments(ci)
    dest = tmp_path / "sessions" / "12345" / "inbox" / "12-f.jpg"
    assert dest.read_bytes() == b"\xff"
    assert out.text == "[attachment saved: inbox/12-f.jpg (photo)]"


def test_ingest_disabled_leaves_text_alone(tmp_path: Path) -> None:
    client = _AttachmentClient()
    poller = _attachment_poller(tmp_path, client)
    poller._static_telegram_config = TelegramConfig(attachments_enabled=False)
    ci = ChatInput(
        chat_id=12345,
        message_id=13,
        text="cap",
        attachments=(TelegramAttachment(kind="document", file_id="d"),),
    )
    out = poller._ingest_attachments(ci)
    assert out.text == "cap"
    assert client.streamed_urls == []
    assert not (tmp_path / "sessions").exists()


def test_ingest_oversized_annotates_and_writes_nothing(tmp_path: Path) -> None:
    client = _AttachmentClient(file_size=50_000_000)
    poller = _attachment_poller(tmp_path, client)
    ci = ChatInput(
        chat_id=12345,
        message_id=14,
        text="",
        attachments=(TelegramAttachment(kind="video", file_id="v", file_size=50_000_000),),
    )
    out = poller._ingest_attachments(ci)
    assert "could not be saved" in out.text
    assert client.streamed_urls == []


def test_ingest_stream_overrun_annotates_and_removes_partial(tmp_path: Path) -> None:
    client = _AttachmentClient(
        file_path="documents/big.bin",
        file_size=None,
        chunks=[b"x" * 65_536] * 400,
    )
    client._file_size = None
    poller = _attachment_poller(tmp_path, client)
    poller._static_telegram_config = TelegramConfig(attachments_max_bytes=100_000)
    ci = ChatInput(
        chat_id=12345,
        message_id=15,
        text="",
        attachments=(TelegramAttachment(kind="document", file_id="d", file_name="big.bin"),),
    )
    out = poller._ingest_attachments(ci)
    assert "could not be saved" in out.text
    assert not (tmp_path / "sessions" / "12345" / "inbox" / "15-big.bin").exists()


def test_ingest_getfile_failure_annotates(tmp_path: Path) -> None:
    class _NoFile:
        def post(self, url: str, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse({"ok": True, "result": {}})

    poller = _attachment_poller(tmp_path, _NoFile())
    ci = ChatInput(
        chat_id=12345,
        message_id=16,
        text="",
        attachments=(TelegramAttachment(kind="photo", file_id="p"),),
    )
    out = poller._ingest_attachments(ci)
    assert "could not be saved" in out.text


def test_safe_filename_strips_path_components() -> None:
    from diploid_agent.transport.telegram.poller import _safe_filename

    assert _safe_filename("../../etc/passwd", fallback="f") == "passwd"
    assert _safe_filename("..\\win\\evil.exe", fallback="f") == "win_evil.exe"
    assert _safe_filename(".../..", fallback="f") == "f"
    assert _safe_filename("normal-name_v2.jpg", fallback="f") == "normal-name_v2.jpg"


# ---------------------------------------------------------------------------
# STT (voice transcription)
# ---------------------------------------------------------------------------


def test_ingest_voice_transcribes_with_command_provider(tmp_path: Path) -> None:
    client = _AttachmentClient(file_path="voice/v.oga", chunks=[b"ogg-bytes"])
    poller = _attachment_poller(tmp_path, client)
    poller._static_telegram_config = TelegramConfig(
        stt_provider="command", stt_command="echo heard:"
    )
    ci = ChatInput(
        chat_id=12345,
        message_id=20,
        text="",
        attachments=(TelegramAttachment(kind="voice", file_id="v1", mime_type="audio/ogg"),),
    )
    out = poller._ingest_attachments(ci)
    assert "[attachment saved: inbox/20-v.oga (voice, audio/ogg)]" in out.text
    assert '[transcript: "heard:' in out.text


def test_ingest_provider_none_adds_no_transcript(tmp_path: Path) -> None:
    client = _AttachmentClient(file_path="voice/v.oga", chunks=[b"ogg"])
    poller = _attachment_poller(tmp_path, client)
    ci = ChatInput(
        chat_id=12345,
        message_id=21,
        text="",
        attachments=(TelegramAttachment(kind="voice", file_id="v"),),
    )
    out = poller._ingest_attachments(ci)
    assert "transcript" not in out.text


def test_ingest_photo_never_transcribes(tmp_path: Path) -> None:
    client = _AttachmentClient(file_path="photos/p.jpg", chunks=[b"\xff"])
    poller = _attachment_poller(tmp_path, client)
    poller._static_telegram_config = TelegramConfig(
        stt_provider="command", stt_command="echo should-not-run"
    )
    ci = ChatInput(
        chat_id=12345,
        message_id=22,
        text="",
        attachments=(TelegramAttachment(kind="photo", file_id="p"),),
    )
    out = poller._ingest_attachments(ci)
    assert "transcript" not in out.text


def test_ingest_failing_stt_command_annotates_unavailable(tmp_path: Path) -> None:
    client = _AttachmentClient(file_path="voice/v.oga", chunks=[b"ogg"])
    poller = _attachment_poller(tmp_path, client)
    poller._static_telegram_config = TelegramConfig(stt_provider="command", stt_command="false")
    ci = ChatInput(
        chat_id=12345,
        message_id=23,
        text="",
        attachments=(TelegramAttachment(kind="voice", file_id="v"),),
    )
    out = poller._ingest_attachments(ci)
    assert "[transcript unavailable]" in out.text


def test_transcribe_faster_whisper_dispatch(tmp_path: Path, monkeypatch: Any) -> None:
    import sys
    import types

    import diploid_agent.transport.telegram.voice as voice_mod

    class _Seg:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Model:
        def __init__(self, name: str, **kwargs: Any) -> None:
            assert name == "tiny"

        def transcribe(self, path: str) -> tuple[list[_Seg], Any]:
            return [_Seg(" hello "), _Seg(" world ")], SimpleNamespace()

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _Model
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    voice_mod._model_cache.clear()

    cfg = TelegramConfig(stt_provider="faster-whisper", stt_model="tiny")
    assert voice_mod.transcribe(tmp_path / "x.oga", cfg) == "hello world"
    # Second call reuses the cached model (no second __init__ assertion needed —
    # a re-instantiation would just re-run the same fake; the cache key is
    # pinned by the model name).
    assert "tiny" in voice_mod._model_cache


def test_transcribe_unknown_provider_returns_none(tmp_path: Path) -> None:
    from diploid_agent.transport.telegram.voice import transcribe

    cfg = TelegramConfig()
    object.__setattr__(cfg, "stt_provider", "bogus")
    assert transcribe(tmp_path / "x.oga", cfg) is None


def test_poller_stt_kwargs_reach_static_config() -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        stt_provider="command",
        stt_command="whisper-cli",
        stt_model="base",
    )
    cfg = poller._live_telegram_config
    assert cfg.stt_provider == "command"
    assert cfg.stt_command == "whisper-cli"
    assert cfg.stt_model == "base"


# ---------------------------------------------------------------------------
# TTS (say blocks → voice notes)
# ---------------------------------------------------------------------------


def test_extract_say_block() -> None:
    from diploid_agent.transport.interactive import extract_say_block

    text, say = extract_say_block("hello there\n```say\nGood night, love.\n```")
    assert text == "hello there"
    assert say == "Good night, love."
    assert extract_say_block("no block") == ("no block", None)
    assert extract_say_block("```say\n\n```") == ("```say\n\n```", None)


def test_say_block_sends_voice_via_command_provider(tmp_path: Path) -> None:
    ogg = tmp_path / "voice.ogg"
    ogg.write_bytes(b"OggS" + b"\x00" * 64)
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        tts_provider="command",
        tts_command=f"cat {ogg}",
    )
    calls: list[tuple[str, dict, dict]] = []

    def fake_api(method: str, *, files: Any = None, **params: Any) -> dict:
        calls.append((method, params, files or {}))
        return {"ok": True, "result": {"message_id": 99}}

    poller._api = fake_api
    poller._send_text(12345, "text reply\n```say\nGood night, love.\n```")

    send = [c for c in calls if c[0] == "sendMessage"]
    voice = [c for c in calls if c[0] == "sendVoice"]
    assert len(send) == 1 and send[0][1]["text"] == "text reply"
    assert len(voice) == 1
    assert voice[0][1]["chat_id"] == 12345
    field = voice[0][2]["voice"]
    assert field[1][:4] == b"OggS"


def test_say_block_non_ogg_uses_sendaudio(tmp_path: Path) -> None:
    wav = tmp_path / "clip.bin"
    wav.write_bytes(b"RIFF" + b"\x00" * 32)
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        tts_provider="command",
        tts_command=f"cat {wav}",
    )
    calls: list[str] = []
    poller._api = lambda method, **kw: (
        calls.append(method) or {"ok": True, "result": {"message_id": 1}}
    )
    poller._send_text(12345, "```say\nhi\n```")
    assert "sendAudio" in calls


def test_say_block_provider_none_falls_back_to_text(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent: list[str] = []
    poller._api = lambda method, **kw: (
        sent.append(kw["text"]) or {"ok": True, "result": {"message_id": 1}}
    )
    poller._send_text(12345, "main text\n```say\nspoken words\n```")
    assert sent == ["main text", "[say] spoken words"]


def test_say_block_over_max_chars_falls_back_to_text(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        tts_provider="command",
        tts_command="false",
        tts_max_chars=5,
    )
    sent: list[str] = []
    poller._api = lambda method, **kw: (
        sent.append(kw["text"]) or {"ok": True, "result": {"message_id": 1}}
    )
    poller._send_text(12345, "```say\nthis is far too long to speak\n```")
    assert sent == ["[say] this is far too long to speak"]


def test_say_block_synth_failure_falls_back_to_text(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        tts_provider="command",
        tts_command="false",
    )
    sent: list[str] = []
    poller._api = lambda method, **kw: (
        sent.append(kw["text"]) or {"ok": True, "result": {"message_id": 1}}
    )
    poller._send_text(12345, "```say\nsomething\n```")
    assert sent == ["[say] something"]


# ---------------------------------------------------------------------------
# Outbound files (```file blocks → workspace uploads)
# ---------------------------------------------------------------------------


def test_extract_file_blocks() -> None:
    from diploid_agent.transport.interactive import extract_file_blocks

    text, refs = extract_file_blocks(
        "here you go\n```file\noutbox/a.pdf\nreport for you\n```\ntail"
    )
    assert text == "here you go\n\ntail"
    assert refs == [FileRef(path="outbox/a.pdf", caption="report for you")]

    text2, refs2 = extract_file_blocks("```file\na.txt\n```\nmiddle\n```file\nb.jpg\n```")
    assert [r.path for r in refs2] == ["a.txt", "b.jpg"]
    assert refs2[0].caption == ""
    assert "file" not in text2

    assert extract_file_blocks("no block") == ("no block", [])
    assert extract_file_blocks("```file\n\n```") == ("", [])


def test_file_block_sends_document(tmp_path: Path) -> None:
    workspace = tmp_path / "12345"
    (workspace / "outbox").mkdir(parents=True)
    (workspace / "outbox" / "report.pdf").write_bytes(b"%PDF-fake")
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        sessions_root=tmp_path,
    )
    calls: list[tuple[str, dict, dict]] = []

    def fake_api(method: str, *, files: Any = None, **params: Any) -> dict:
        calls.append((method, params, files or {}))
        return {"ok": True, "result": {"message_id": 42}}

    poller._api = fake_api
    poller._send_text(12345, "made this\n```file\noutbox/report.pdf\nSeptember notes\n```")

    send = [c for c in calls if c[0] == "sendMessage"]
    doc = [c for c in calls if c[0] == "sendDocument"]
    assert len(send) == 1 and send[0][1]["text"] == "made this"
    assert len(doc) == 1
    assert doc[0][1]["caption"] == "September notes"
    field = doc[0][2]["document"]
    assert field[0] == "report.pdf" and field[1] == b"%PDF-fake"


def test_file_block_image_uses_sendphoto(tmp_path: Path) -> None:
    workspace = tmp_path / "7"
    workspace.mkdir(parents=True)
    (workspace / "pic.png").write_bytes(b"\x89PNG" + b"\x00" * 16)
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        sessions_root=tmp_path,
    )
    calls: list[str] = []
    poller._api = lambda method, **kw: (
        calls.append(method) or {"ok": True, "result": {"message_id": 1}}
    )
    poller._send_text(7, "```file\npic.png\n```")
    assert "sendPhoto" in calls


def test_file_block_escape_falls_back_to_text(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("nope")
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        sessions_root=tmp_path,
    )
    sent: list[str] = []
    poller._api = lambda method, **kw: (
        sent.append(kw.get("text", method)) or {"ok": True, "result": {"message_id": 1}}
    )
    poller._send_text(12345, "```file\n../secret.txt\n```")
    poller._send_text(12345, f"```file\n{tmp_path}/secret.txt\n```")
    poller._send_text(12345, "```file\nmissing.txt\n```")
    assert sent == [
        "[file] ../secret.txt",
        f"[file] {tmp_path}/secret.txt",
        "[file] missing.txt",
    ]


def test_file_block_oversize_falls_back(tmp_path: Path) -> None:
    workspace = tmp_path / "9"
    workspace.mkdir(parents=True)
    (workspace / "big.bin").write_bytes(b"\x00" * 64)
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        sessions_root=tmp_path,
        attachments_max_bytes=8,
    )
    sent: list[str] = []
    poller._api = lambda method, **kw: (
        sent.append(kw.get("text", method)) or {"ok": True, "result": {"message_id": 1}}
    )
    poller._send_text(9, "```file\nbig.bin\n```")
    assert sent == ["[file] big.bin"]


def test_file_block_caption_preserved_on_failure(tmp_path: Path) -> None:
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        sessions_root=tmp_path,
    )
    sent: list[str] = []
    poller._api = lambda method, **kw: (
        sent.append(kw.get("text", method)) or {"ok": True, "result": {"message_id": 1}}
    )
    poller._send_text(5, "```file\ngone.txt\nkeep these words\n```")
    assert sent == ["[file] gone.txt\nkeep these words"]

# ---------------------------------------------------------------------------
# P3 review fixes — turn-worker ordering, delivery backoff, voice, stream caps
# ---------------------------------------------------------------------------

def test_placeholder_sent_before_attachment_ingest(tmp_path: Path) -> None:
    """The "..." placeholder is visible before any download/STT work begins."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    chat_input = ChatInput(
        chat_id=12345,
        message_id=1,
        text="hi",
        attachments=[TelegramAttachment(kind="document", file_id="f1")],
    )
    worker = TurnWorker(poller, chat_input)
    order: list[str] = []

    poller._send_message = lambda chat_id, text, **kw: order.append("placeholder") or 100

    def fake_ingest(ci: ChatInput) -> ChatInput:
        order.append("ingest")
        return ci

    poller._ingest_attachments = fake_ingest
    poller._api = lambda method, **kw: {"ok": True, "result": {}}
    poller._save_placeholder_state = lambda *a, **kw: None
    poller._remove_placeholder_state = lambda *a, **kw: None
    worker._harness_chat = lambda ci: {"reply": "ok"}
    worker._stream_turn = lambda future, message_id, thought_id: {"reply": "ok"}

    worker._run_turn(chat_input)

    assert order[:2] == ["placeholder", "ingest"]

def test_delivery_worker_empty_backoff_is_interruptible(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An empty outbox must not park the worker on an uninterruptible sleep."""
    monkeypatch.setattr(DeliveryWorker, "_EMPTY_BACKOFF", 60.0)
    runtime = _FakeDeliveryRuntime()
    poller = TelegramPoller(
        token="dummy",
        runtime=runtime,  # type: ignore[arg-type]
        state_dir=tmp_path / ".poller-placeholders",
    )
    worker = DeliveryWorker(poller, 12345)
    worker.start()
    time.sleep(0.3)
    worker.stop()
    worker.join(timeout=5.0)
    # With Event.wait the worker exits promptly; time.sleep(60) would still be parked.
    assert not worker.is_alive()

def test_ask_keyboard_attached_to_last_chunk(tmp_path: Path) -> None:
    """A multi-chunk ask puts the keyboard on the final chunk, not nowhere."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    calls: list[tuple[str, dict]] = []
    next_id = iter(range(100, 110))

    def fake_api(method: str, **params: Any) -> dict:
        calls.append((method, params))
        return {"ok": True, "result": {"message_id": next(next_id)}}

    poller._api = fake_api  # type: ignore[method-assign]
    saved: list[int | None] = []
    poller._save_pending_question = lambda chat_id, block, msg_id: saved.append(msg_id)  # type: ignore[method-assign]

    text = ("word " * 900).strip() + '\n```ask\n{"options": ["a", "b"]}\n```'
    poller._send_text(12345, text)

    sends = [p for m, p in calls if m == "sendMessage"]
    assert len(sends) == 2
    assert "reply_markup" not in sends[0]
    markup = json.loads(sends[1]["reply_markup"])
    assert markup["inline_keyboard"]
    # The pending question is bound to the message that carries the keyboard.
    assert saved == [101]


# ---------------------------------------------------------------------------
# Voice synthesis bounds and piper cache/compat
# ---------------------------------------------------------------------------

def test_voice_synthesis_runs_outside_send_lock(tmp_path: Path) -> None:
    """A slow TTS call must not serialize every outbound message for the chat."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
        tts_provider="command",
        tts_command="cat /dev/null",
    )
    poller._api = lambda method, **kw: {"ok": True, "result": {"message_id": 1}}
    held: list[bool] = []

    def spy(chat_id: int, say_text: str, *, reply_to_message_id: int | None = None) -> None:
        held.append(poller._send_locks[chat_id]._is_owned())

    poller._maybe_send_voice = spy  # type: ignore[method-assign]
    poller._send_text(12345, "text reply\n```say\nhi there\n```")

    assert held == [False]

def test_synthesize_bounded_times_out(tmp_path: Path, monkeypatch: Any) -> None:
    """A wedged provider surfaces as None after the join deadline."""
    from diploid_agent.transport.telegram import voice as voice_mod

    monkeypatch.setattr(voice_mod, "synthesize", lambda *a, **kw: time.sleep(5))
    config = TelegramConfig(tts_provider="command", tts_command="cat")
    assert voice_mod.synthesize_bounded("hi", config, tmp_path, timeout=0.1) is None

def test_synthesize_bounded_passthrough_and_raise(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from diploid_agent.transport.telegram import voice as voice_mod

    out = tmp_path / "say.ogg"
    monkeypatch.setattr(voice_mod, "synthesize", lambda *a, **kw: out)
    config = TelegramConfig()
    assert voice_mod.synthesize_bounded("hi", config, tmp_path) is out

    def boom(*a: Any, **kw: Any) -> None:
        raise RuntimeError("wedged")

    monkeypatch.setattr(voice_mod, "synthesize", boom)
    try:
        voice_mod.synthesize_bounded("hi", config, tmp_path, timeout=1.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("synthesize exception should propagate")


class _FakeAudioChunk:
    sample_channels = 1
    sample_width = 2
    sample_rate = 22050
    audio_int16_bytes = b"\x00\x01" * 64

def _fake_ffmpeg(monkeypatch: Any, voice_mod: Any) -> None:
    def run(cmd: list[str], **kw: Any) -> SimpleNamespace:
        Path(cmd[-1]).write_bytes(b"OggS" + b"\x00" * 32)
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(voice_mod, "subprocess", SimpleNamespace(run=run))

def test_piper_voice_cache_idle_eviction(tmp_path: Path, monkeypatch: Any) -> None:
    """Loaded piper voices are evicted after the idle TTL."""
    import sys

    from diploid_agent.transport.telegram import voice as voice_mod

    loads: list[str] = []

    class FakePiperVoice:
        @classmethod
        def load(cls, path: str) -> Self:
            loads.append(path)
            return cls()

        def synthesize(self, text: str) -> Any:
            return iter([_FakeAudioChunk()])

    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=FakePiperVoice))
    _fake_ffmpeg(monkeypatch, voice_mod)

    stale_voice = object()
    voice_mod._voice_cache["/stale.onnx"] = (
        stale_voice,
        time.monotonic() - voice_mod._VOICE_IDLE_TTL - 1.0,
    )
    try:
        out = voice_mod._synthesize_piper("hi", "/new.onnx", tmp_path)
        assert out is not None
        assert "/stale.onnx" not in voice_mod._voice_cache
        assert loads == ["/new.onnx"]
    finally:
        voice_mod._voice_cache.clear()

def test_piper_typeerror_mid_iteration_is_real_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A TypeError while consuming synthesis output must not retry the legacy API."""
    import sys

    from diploid_agent.transport.telegram import voice as voice_mod

    calls: list[int] = []

    class FakePiperVoice:
        @classmethod
        def load(cls, path: str) -> Self:
            return cls()

        def synthesize(self, *args: Any) -> Any:
            calls.append(len(args))
            if len(args) != 1:
                raise AssertionError("legacy path must not be used")

            def gen() -> Any:
                yield _FakeAudioChunk()
                raise TypeError("mid-stream failure")

            return gen()

    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=FakePiperVoice))
    _fake_ffmpeg(monkeypatch, voice_mod)

    try:
        assert voice_mod._synthesize_piper("hi", "/m.onnx", tmp_path) is None
        assert calls == [1]
    finally:
        voice_mod._voice_cache.clear()

def test_piper_legacy_signature_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    """Old piper releases (synthesize(text, wav)) still work via the TypeError shim."""
    import sys

    from diploid_agent.transport.telegram import voice as voice_mod

    calls: list[int] = []

    class FakePiperVoice:
        @classmethod
        def load(cls, path: str) -> Self:
            return cls()

        def synthesize(self, *args: Any) -> Any:
            calls.append(len(args))
            if len(args) == 1:
                raise TypeError("synthesize() missing wav arg")
            args[1].setnchannels(1)
            args[1].setsampwidth(2)
            args[1].setframerate(22050)
            args[1].writeframes(b"\x00\x01" * 64)
            return None

    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=FakePiperVoice))
    _fake_ffmpeg(monkeypatch, voice_mod)

    try:
        out = voice_mod._synthesize_piper("hi", "/m.onnx", tmp_path)
        assert out is not None
        assert calls == [1, 2]
    finally:
        voice_mod._voice_cache.clear()


# ---------------------------------------------------------------------------
# StreamDisplay — heartbeat cap and continuation commit lifecycle
# ---------------------------------------------------------------------------

def _stream_display(poller: TelegramPoller, **overrides: Any) -> Any:
    from diploid_agent.transport.telegram.stream_display import StreamDisplay

    config = SimpleNamespace(
        intermediate_messages=True,
        intermediate_idle=0.0,
        intermediate_min_chars=10,
    )
    return StreamDisplay(
        poller=poller,
        chat_id=12345,
        reply_to_message_id=None,
        config=config,
        message_id=overrides.get("message_id"),
        thought_id=None,
    )

def test_next_wait_derives_cap_from_heartbeat_interval(tmp_path: Path) -> None:
    """The long-poll cap follows _HEARTBEAT_INTERVAL, not a stale constant."""
    from diploid_agent.transport.telegram.formatting import _HEARTBEAT_INTERVAL

    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    display = _stream_display(poller)
    wait = display.next_wait()
    # With the old 25.0 constant this would return exactly 25.0.
    assert 25.0 < wait <= _HEARTBEAT_INTERVAL

def test_continuation_deletes_committed_and_restreams(tmp_path: Path) -> None:
    """On continuation the committed intermediate is deleted because the next
    turn's fresh display re-streams the cumulative text — the live placeholder
    survives so the continuing stream has a message to fill."""
    poller = TelegramPoller(
        token="dummy",
        harness_url="http://localhost",
        state_dir=tmp_path / ".poller-placeholders",
    )
    sent_ids = iter(range(100, 110))
    edits: list[tuple[int, str]] = []
    deletes: list[int] = []

    poller._send_message = lambda chat_id, text, **kw: next(sent_ids)
    poller._edit_message_text = lambda chat_id, mid, text, **kw: edits.append(
        (mid, text)
    ) or True
    poller._delete_message = lambda chat_id, mid: deletes.append(mid)
    poller._save_placeholder_state = lambda *a, **kw: None

    display = _stream_display(poller)
    status = {"status": "running", "message_text": "First paragraph ends here. "}
    display.update(status)  # creates placeholder 100, shows the paragraph
    display.update(status)  # idle tail → commits 100, opens placeholder 101
    assert display.committed_message_id == 100
    assert display.message_id == 101

    display.finalize({"continuation": True})
    # The already-visible committed message is deleted; the live placeholder
    # is kept for the continuing stream.
    assert deletes == [100]

    # The continuation turn builds a fresh display with no committed baseline,
    # so the cumulative message_text re-streams the deleted paragraph.
    cont = _stream_display(poller)
    cont.update(
        {
            "status": "running",
            "message_text": "First paragraph ends here. Second part arrives.",
        }
    )
    assert edits[-1][1].startswith("First paragraph ends here.")
