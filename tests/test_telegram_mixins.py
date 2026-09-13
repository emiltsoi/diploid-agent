"""Focused unit tests for the extracted Telegram mixins."""

from __future__ import annotations

from diploid_agent.models import ChatResult
from diploid_agent.transport.telegram.commands import TelegramCommandMixin
from diploid_agent.transport.telegram.models import ChatInput
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


class _RecordingHandler:
    """Stands in for CommandHandler: records call kwargs, returns a result."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return ChatResult(reply="switched")


def _commands_recording() -> tuple[_Commands, _RecordingHandler, list[str]]:
    commands = _Commands()
    handler = _RecordingHandler()
    sent_texts: list[str] = []
    commands.command_handler = handler
    commands._send_text = lambda chat_id, text, **kw: sent_texts.append(text)
    commands._send_result = lambda chat_id, result, **kw: None
    return commands, handler, sent_texts


def _chat_input(text: str) -> ChatInput:
    return ChatInput(chat_id=1, message_id=2, text=text)


def test_model_command_in_place_flag() -> None:
    commands, handler, _ = _commands_recording()
    ci = _chat_input("/model --in-place glm-5.2")
    assert commands._handle_command(ci, "/model", "--in-place glm-5.2") is True
    call = handler.calls[-1]
    assert call["method"] == "switch_model"
    assert call["model"] == "glm-5.2"
    assert call["in_place"] is True
    assert call["http_body"] == {"model": "glm-5.2", "in_place": True}


def test_model_command_in_place_flag_trailing() -> None:
    """`/model <name> --in-place` is accepted — the flag may appear anywhere."""
    commands, handler, _ = _commands_recording()
    ci = _chat_input("/model glm-5.2 --in-place")
    assert commands._handle_command(ci, "/model", "glm-5.2 --in-place") is True
    call = handler.calls[-1]
    assert call["model"] == "glm-5.2"
    assert call["in_place"] is True


def test_model_command_plain_omits_in_place() -> None:
    """`/model <name>` forwards no `in_place` kwarg — older runtimes keep working."""
    commands, handler, _ = _commands_recording()
    ci = _chat_input("/model glm-5.2")
    assert commands._handle_command(ci, "/model", "glm-5.2") is True
    call = handler.calls[-1]
    assert call["model"] == "glm-5.2"
    assert "in_place" not in call
    assert call["http_body"] == {"model": "glm-5.2"}


def test_model_command_unknown_flag_shows_usage() -> None:
    commands, handler, sent_texts = _commands_recording()
    ci = _chat_input("/model --bogus glm-5.2")
    assert commands._handle_command(ci, "/model", "--bogus glm-5.2") is True
    assert handler.calls == []
    assert sent_texts == ["Usage: /model [--in-place] <name>"]


def test_model_command_no_args_shows_usage() -> None:
    commands, handler, sent_texts = _commands_recording()
    ci = _chat_input("/model --in-place")
    assert commands._handle_command(ci, "/model", "--in-place") is True
    assert handler.calls == []
    assert sent_texts == ["Usage: /model [--in-place] <name>"]
