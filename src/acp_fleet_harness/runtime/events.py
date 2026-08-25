"""Typed event dataclasses used by the runtime event bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserMessageEvent:
    chat_id: str
    message: str
    model: str | None = None
    reply_to: str | None = None
    reply_to_is_bot: bool | None = None
    reply_to_message_id: int | None = None


@dataclass
class TimerFiredEvent:
    event_id: str
    chat_id: str
    reason: str
    silent: bool = True
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DispatchCompletedEvent:
    dispatch_id: str
    result: str


@dataclass
class TaskCompletedEvent:
    plan_id: str
    task_id: str
    result: str
    log: str = ""
