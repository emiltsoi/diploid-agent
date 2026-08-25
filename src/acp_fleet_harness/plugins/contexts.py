"""Typed contexts for plugin lifecycle hooks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acp_fleet_harness.dispatch import Dispatch
from acp_fleet_harness.engine.base import TurnRequest, TurnResult
from acp_fleet_harness.models import ChatResult, SessionRecord


@dataclass
class PromptContext:
    """The result of assembling a prompt for a turn."""

    prompt: str
    notice: str | None
    memory_flags: dict[str, bool]
    slots: dict[str, list[str]]
    model: str | None = None


@dataclass
class TurnStartContext:
    """Gate hook context for the start of a turn."""

    chat_id: str
    user_message: str
    model: str | None
    record: SessionRecord | None
    reply_to: str | None = None
    reply_to_is_bot: bool | None = None
    reply_to_message_id: int | None = None
    now: float = 0.0


@dataclass
class UserMessageContext:
    """Consult hook context for user-message formatting."""

    chat_id: str
    raw_message: str
    formatted_message: str | None
    reply_to: str | None = None
    reply_to_is_bot: bool | None = None
    reply_to_message_id: int | None = None


@dataclass
class PromptBuildContext:
    """Consult hook context before a prompt is built."""

    chat_id: str
    record: SessionRecord | None
    model: str | None
    is_first: bool
    continuation_anchor: str | None = None


@dataclass
class EngineCallContext:
    """Gate hook context immediately before the ACP engine is called."""

    chat_id: str
    request: TurnRequest
    session_id: str | None
    record: SessionRecord | None
    on_chunk: Callable[[str], None] | None = None
    on_update: Callable[[dict[str, Any]], None] | None = None


@dataclass
class EngineResultContext:
    """Consult hook context after the ACP engine returns a result."""

    chat_id: str
    record: SessionRecord | None
    result: TurnResult
    reply: str
    usage: dict[str, Any] | None = None
    stop_reason: str | None = None


@dataclass
class RecordTurnContext:
    """Consult hook context before a turn is recorded to the session store."""

    chat_id: str
    record: SessionRecord
    turn_number: int
    reply: str
    notice: str | None = None
    memory_flags: dict[str, bool] | None = None
    metrics: dict[str, Any] | None = None


@dataclass
class TurnErrorContext:
    """Notify hook context when a turn raises an unhandled exception."""

    chat_id: str
    record: SessionRecord | None
    user_message: str
    exception: BaseException
    now: float = 0.0


@dataclass
class SessionArchiveContext:
    """Consult hook context before the active session is archived."""

    chat_id: str
    old_record: SessionRecord


@dataclass
class SessionClearContext:
    """Consult hook context before the active session directory is cleared."""

    chat_id: str
    old_record: SessionRecord


@dataclass
class SessionStartContext:
    """Gate hook context for starting a new active session."""

    chat_id: str
    kind: str  # "new", "resume", "branch", "switch_model"
    user_message: str
    model: str
    session_number: int
    old_model: str | None = None
    skill_names: set[str] | None = None
    mcp_servers: list[dict[str, Any]] | None = None


@dataclass
class SessionActiveContext:
    """Consult/notify hook context after a new active record is stored."""

    chat_id: str
    record: SessionRecord


@dataclass
class DispatchCreateContext:
    """Consult hook context when a dispatch is being created."""

    chat_id: str
    record: SessionRecord
    context: str | None
    dispatch: Dispatch | None = None


@dataclass
class DispatchContinueContext:
    """Consult hook context when a dispatch result is being resumed."""

    chat_id: str
    dispatch: Dispatch
    result: str


@dataclass
class DispatchCompleteContext:
    """Notify hook context after a dispatch continuation completes."""

    chat_id: str
    dispatch: Dispatch
    record: SessionRecord
    result: ChatResult


@dataclass
class MemoryTransitionContext:
    """Consult hook context for chat or persona memory cap transitions."""

    chat_id: str
    record: SessionRecord
    kind: str  # "chat" or "persona"
    path: Path
    total: int
    cap: int
    notice: str | None = None
    suppress_default: bool = False


@dataclass
class SkillCommandContext:
    """Consult/notify hook context for /skill enable or disable."""

    chat_id: str
    skill_name: str
    enabled: bool
    record: SessionRecord | None


@dataclass
class McpCommandContext:
    """Consult/notify hook context for /mcp enable or disable."""

    chat_id: str
    server_name: str
    enabled: bool
    record: SessionRecord | None


@dataclass
class RetainContext:
    """Consult/notify hook context for retaining a memory item."""

    chat_id: str
    content: str
    tags: list[str]
    context: str | None


@dataclass
class PromoteContext:
    """Consult/notify hook context for promoting a fact to persona memory."""

    chat_id: str
    fact: str
    record: SessionRecord | None


@dataclass
class ShutdownContext:
    """Notify hook context when the harness is shutting down."""

    chat_id: str
    record: SessionRecord | None
    reason: str
    now: float
    instance_id: str
    instance_started_at: float


@dataclass
class IdleContext:
    """Notify hook context for idle time between turns."""

    chat_id: str
    now: float
    instance_id: str
    record: SessionRecord | None = None
