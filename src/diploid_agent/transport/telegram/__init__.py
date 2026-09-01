"""Telegram long-polling transport."""

import time  # noqa: F401

from diploid_agent.transport.telegram.commands import TelegramCommandMixin
from diploid_agent.transport.telegram.formatting import (
    _HEARTBEAT_INTERVAL,
    _REPLY_PLACEHOLDER,
    _THINKING_CONTINUED,
    _THINKING_PREFIX,
    _build_heartbeat_text,
    _format_elapsed,
    _format_subagent_time,
    _format_thought,
)
from diploid_agent.transport.telegram.models import ChatInput
from diploid_agent.transport.telegram.poller import (
    DeliveryWorker,
    TelegramPoller,
    TurnWorker,
)
from diploid_agent.transport.telegram.sender import TelegramSenderMixin
from diploid_agent.transport.telegram.state import TelegramStateMixin
from diploid_agent.transport.telegram.transport import TelegramTransport, main

__all__ = [
    "_HEARTBEAT_INTERVAL",
    "_REPLY_PLACEHOLDER",
    "_THINKING_CONTINUED",
    "_THINKING_PREFIX",
    "ChatInput",
    "DeliveryWorker",
    "TelegramCommandMixin",
    "TelegramPoller",
    "TelegramSenderMixin",
    "TelegramStateMixin",
    "TelegramTransport",
    "TurnWorker",
    "_build_heartbeat_text",
    "_format_elapsed",
    "_format_subagent_time",
    "_format_thought",
    "main",
]
