"""Dataclasses for the Telegram long-polling transport."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelegramAttachment:
    """A file-bearing Telegram message part awaiting download."""

    kind: str
    file_id: str
    file_name: str | None = None
    mime_type: str | None = None
    file_size: int | None = None


@dataclass
class WakeTombstone:
    """Record of a wake display that finalized without its routed result.

    Left behind when ``WakeDisplayWorker`` exits on the grace-miss path so a
    late outbox result can fold into the already-sent bubbles instead of
    double-posting. ``last_bubble_content`` is the rendered text of
    ``last_message_id`` — ``None`` when unknown, in which case the delivery
    side sends the delta standalone rather than editing.
    """

    full_text: str
    last_message_id: int | None
    last_bubble_content: str | None
    session_number: int | None
    turn_number: int | None
    finalized_at: float


@dataclass(frozen=True)
class ChatInput:
    """A normalized user message from Telegram, including any reply-to context."""

    chat_id: int
    message_id: int
    text: str
    reply_to: str | None = None
    reply_to_is_bot: bool | None = None
    reply_to_message_id: int | None = None
    callback_query_id: str | None = None
    attachments: tuple[TelegramAttachment, ...] = ()
