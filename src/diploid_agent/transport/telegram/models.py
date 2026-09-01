"""Dataclasses for the Telegram long-polling transport."""

from __future__ import annotations

from dataclasses import dataclass


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
