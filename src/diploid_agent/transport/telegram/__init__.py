"""Telegram long-polling transport."""

import time  # noqa: F401

from diploid_agent.transport.telegram.poller import (
    _HEARTBEAT_INTERVAL,
    _REPLY_PLACEHOLDER,
    _THINKING_CONTINUED,
    _THINKING_PREFIX,
    ChatInput,
    DeliveryWorker,
    TelegramPoller,
    TelegramTransport,
    TurnWorker,
    _build_heartbeat_text,
    _format_elapsed,
    _format_subagent_time,
    _format_thought,
    main,
)

__all__ = [
    "_HEARTBEAT_INTERVAL",
    "_REPLY_PLACEHOLDER",
    "_THINKING_CONTINUED",
    "_THINKING_PREFIX",
    "ChatInput",
    "DeliveryWorker",
    "TelegramPoller",
    "TelegramTransport",
    "TurnWorker",
    "_build_heartbeat_text",
    "_format_elapsed",
    "_format_subagent_time",
    "_format_thought",
    "main",
]
