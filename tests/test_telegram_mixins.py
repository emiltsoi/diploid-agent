"""Focused unit tests for the extracted Telegram mixins."""

from __future__ import annotations

from diploid_agent.models import ChatResult
from diploid_agent.transport.telegram.commands import TelegramCommandMixin
from diploid_agent.transport.telegram.sender import TelegramSenderMixin
from diploid_agent.transport.telegram.state import TelegramStateMixin


class _Commands(TelegramCommandMixin):
    pass


def test_split_telegram_text_empty() -> None:
    assert TelegramSenderMixin._split_telegram_text("") == [""]


def test_split_telegram_text_short() -> None:
    text = "short"
    assert TelegramSenderMixin._split_telegram_text(text) == [text]


def test_split_telegram_text_splits_by_words() -> None:
    text = "word " * 5000
    chunks = TelegramSenderMixin._split_telegram_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 4096 - 16


def test_make_preview_short_text() -> None:
    preview, length = TelegramStateMixin._make_preview("hello", max_chars=240)
    assert preview == "hello"
    assert length == 5


def test_make_preview_long_text() -> None:
    text = "a" * 300
    preview, length = TelegramStateMixin._make_preview(text, max_chars=10)
    assert len(preview) <= 10
    assert length == 300


def test_harness_help_returns_string() -> None:
    commands = _Commands()
    help_text = commands._harness_help(123)
    assert isinstance(help_text, str)
    assert "/help" in help_text


class _StubHandler:
    """Stands in for CommandHandler in embedded mode: returns raw values."""

    def __init__(self, result):
        self._result = result

    def call(self, **_kwargs):
        return self._result


def test_harness_call_reply_coerces_chat_result() -> None:
    """Embedded-mode calls can return ChatResult — not a sorry message."""
    commands = _Commands()
    commands.command_handler = _StubHandler(ChatResult(reply="reloaded ok"))
    reply = commands._harness_call_reply(sorry="Sorry, I could not reload x.")
    assert reply == "reloaded ok"


def test_harness_call_reply_error_dict_returns_sorry() -> None:
    commands = _Commands()
    commands.command_handler = _StubHandler({"error": "boom"})
    reply = commands._harness_call_reply(sorry="Sorry, I could not reload x.")
    assert reply == "Sorry, I could not reload x."
