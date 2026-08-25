"""Shared dataclasses used across the harness and plugins."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class SessionRecord:
    """On-disk record for one chat session."""

    chat_id: str
    session_number: int
    session_id: str
    model: str
    persona: str
    cwd: str
    created_at: float
    updated_at: float
    turn_number: int = 0
    label: str | None = None
    parent: int | None = None
    last_stop_reason: str | None = None
    persona_memory_exceeded: bool = False
    chat_memory_exceeded: bool = False
    cumulative_metrics: dict[str, Any] | None = None
    last_turn_metrics: dict[str, Any] | None = None
    enabled_mcp_servers: list[str] | None = None
    enabled_skills: list[str] | None = None
    disabled_skills: list[str] | None = None
    plugin_overrides: dict[str, bool] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionRecord:
        return cls(
            chat_id=data["chat_id"],
            session_number=data.get("session_number", 1),
            session_id=data["session_id"],
            model=data["model"],
            persona=data.get("persona", "default-persona"),
            cwd=data["cwd"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            turn_number=data.get("turn_number", 0),
            label=data.get("label"),
            parent=data.get("parent"),
            last_stop_reason=data.get("last_stop_reason"),
            persona_memory_exceeded=data.get("persona_memory_exceeded", False),
            chat_memory_exceeded=data.get("chat_memory_exceeded", False),
            cumulative_metrics=data.get("cumulative_metrics"),
            last_turn_metrics=data.get("last_turn_metrics"),
            enabled_mcp_servers=data.get("enabled_mcp_servers"),
            enabled_skills=data.get("enabled_skills"),
            disabled_skills=data.get("disabled_skills"),
            plugin_overrides=data.get("plugin_overrides"),
        )


@dataclass
class ChatState:
    """All sessions for one chat."""

    sessions: dict[int, SessionRecord] = field(default_factory=dict)
    next_session_number: int = 1


@dataclass
class ChatResult:
    """Result of a harness turn."""

    reply: str
    notice: str | None = None
    dispatch_id: str | None = None
    session_id: str | None = None
    session_number: int | None = None
    turn_number: int | None = None
    metrics: dict[str, Any] | None = None


@dataclass
class ActiveTurn:
    """Track a turn that is currently running against the ACP server."""

    chat_id: str
    session_id: str | None
    user_message: str
    start_time: float
    message_text: str = ""
    thought_text: str = ""
    stopped: bool = False
    _condition: threading.Condition = field(default_factory=threading.Condition, repr=False)


@dataclass
class WakeEvent:
    """A pending wake event for a chat."""

    id: str
    chat_id: str
    reason: str
    priority: int
    scheduled_at: float
    payload: dict[str, Any] = field(default_factory=dict)
    silent: bool = True
    created_at: float = 0.0
    ready: bool = False
    attempts: int = 0
    leased_until: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WakeEvent:
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


@dataclass
class PartialTurn:
    """Snapshot of a turn while it is still streaming."""

    chat_id: str
    session_number: int
    turn_number: int
    user_message: str
    message_text: str = ""
    thought_text: str = ""
    updated_at: float = 0.0


class RuntimeStatus(BaseModel):
    instance_id: str
    started_at: float
    uptime_seconds: float
    event_bus_running: bool
    timer_running: bool
    task_engine_active: bool
    plan_count: int
    pending_wake_count: int
    active_chat_count: int
    plan_active: bool = False
    active_plans: list[str] = Field(default_factory=list)
