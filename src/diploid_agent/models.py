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
    # Turn number reserved for the in-progress turn. Persisted so a killed
    # turn's number is not reused by the next wake.
    pending_turn_number: int | None = None
    label: str | None = None
    parent: int | None = None
    last_stop_reason: str | None = None
    persona_memory_exceeded: bool = False
    chat_memory_exceeded: bool = False
    cumulative_metrics: dict[str, Any] | None = None
    last_turn_metrics: dict[str, Any] | None = None
    first_turn_metrics: dict[str, Any] | None = None
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
            pending_turn_number=data.get("pending_turn_number"),
            label=data.get("label"),
            parent=data.get("parent"),
            last_stop_reason=data.get("last_stop_reason"),
            persona_memory_exceeded=data.get("persona_memory_exceeded", False),
            chat_memory_exceeded=data.get("chat_memory_exceeded", False),
            cumulative_metrics=data.get("cumulative_metrics"),
            last_turn_metrics=data.get("last_turn_metrics"),
            first_turn_metrics=data.get("first_turn_metrics"),
            enabled_mcp_servers=data.get("enabled_mcp_servers"),
            enabled_skills=data.get("enabled_skills"),
            disabled_skills=data.get("disabled_skills"),
            plugin_overrides=data.get("plugin_overrides"),
        )

    def next_turn_number(self) -> int:
        """Return the turn number for an in-progress turn.

        If a pending number is already reserved (because a turn is underway or
        was killed), that number is the current turn. Otherwise it is one past
        the last completed turn.
        """
        if self.pending_turn_number is not None:
            return self.pending_turn_number
        return self.turn_number + 1

    def reserve_turn_number(self) -> int:
        """Reserve the next turn number, skipping any pending killed turn."""
        if self.pending_turn_number is not None:
            self.pending_turn_number += 1
        else:
            self.pending_turn_number = self.turn_number + 1
        return self.pending_turn_number

    def consume_turn_number(self) -> int:
        """Promote the pending turn number to completed and return it."""
        if self.pending_turn_number is not None:
            self.turn_number = self.pending_turn_number
            self.pending_turn_number = None
        else:
            self.turn_number += 1
        return self.turn_number


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
    continuation: bool = False
    dispatch_id: str | None = None
    session_id: str | None = None
    session_number: int | None = None
    turn_number: int | None = None
    metrics: dict[str, Any] | None = None
    reply_to_message_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatResult:
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


# Cap the in-memory thought/message stream so a runaway model cannot exhaust
# memory or produce giant wake/auto-continue payloads and status messages.
MAX_THOUGHT_TEXT_CHARS = 20000


@dataclass
class ActiveTurn:
    """Track a turn that is currently running against the ACP server."""

    chat_id: str
    session_id: str | None
    user_message: str
    start_time: float
    message_text: str = ""
    thought_text: str = ""
    thought_prefix: str = ""
    full_text: str = ""
    thought_total: int = 0
    full_text_offset: int = 0
    stopped: bool = False
    current_intent: str = ""
    last_side_effect: str = ""
    last_side_effect_at: float = 0.0
    side_effects: list[dict[str, Any]] = field(default_factory=list)
    _condition: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def __post_init__(self) -> None:
        if not self.current_intent:
            first_line = (
                (self.user_message or "").strip().splitlines()[0] if self.user_message else ""
            )
            self.current_intent = first_line[:200]

    def seed_from_wake(self, wake_event: WakeEvent | None) -> None:
        """Pre-populate the turn with content from a previous partial turn."""
        if not wake_event or not isinstance(wake_event.payload, dict):
            return
        message_text = wake_event.payload.get("message_text") or ""
        thought_text = wake_event.payload.get("thought_text") or ""
        thought_prefix = wake_event.payload.get("thought_prefix") or ""
        self.thought_text = thought_text[-MAX_THOUGHT_TEXT_CHARS:]
        self.thought_prefix = thought_prefix or thought_text[:MAX_THOUGHT_TEXT_CHARS]
        self.full_text = self.thought_text + message_text
        self.thought_total = wake_event.payload.get("thought_total") or len(self.thought_text)
        self.full_text_offset = wake_event.payload.get("full_text_offset") or 0
        self.recompute_message_text()

    def append_full_text(self, text: str) -> None:
        """Append an agent_message chunk, keeping full_text as a rolling window.

        When the buffer grows past MAX_THOUGHT_TEXT_CHARS we drop the oldest
        prefix (which is part of the thought) and adjust full_text_offset so
        message_text can still be computed correctly.
        """
        self.full_text += text
        excess = len(self.full_text) - MAX_THOUGHT_TEXT_CHARS
        if excess > 0:
            dropped = self.full_text[:excess]
            self.full_text_offset += len(dropped)
            self.full_text = self.full_text[excess:]

    def full_text_has_thought_prefix(self) -> bool:
        """Return True if the retained full_text window starts within the thought.

        For short thoughts, full_text must simply start with the thought_text.
        For long (capped) thoughts, we compare the first MAX chars of the
        thought against the retained window starting at full_text_offset.
        """
        if not self.thought_total or not self.full_text:
            return False
        if self.thought_total <= MAX_THOUGHT_TEXT_CHARS:
            return self.full_text.startswith(self.thought_text)
        prefix_start = self.full_text_offset
        prefix_end = min(MAX_THOUGHT_TEXT_CHARS, prefix_start + len(self.full_text))
        if prefix_end <= prefix_start:
            return False
        return (
            self.full_text[: prefix_end - prefix_start]
            == self.thought_prefix[prefix_start:prefix_end]
        )

    def recompute_message_text(self) -> None:
        """Keep message_text as the part of full_text after the thought prefix.

        thought_total is the cumulative length of agent_thought updates.
        full_text_offset is how much of the agent_message stream has been
        discarded from the front. We only remove the thought prefix if we can
        verify that full_text actually contains it; otherwise the ACP subprocess
        streams the answer and thought on separate channels.
        """
        if not self.full_text:
            self.message_text = ""
            return
        if self.thought_total > 0 and self.full_text_has_thought_prefix():
            start = max(0, self.thought_total - self.full_text_offset)
            self.message_text = self.full_text[start:]
        else:
            self.message_text = self.full_text
        if len(self.message_text) > MAX_THOUGHT_TEXT_CHARS:
            self.message_text = self.message_text[-MAX_THOUGHT_TEXT_CHARS:]

    def append_thought_text(self, text: str) -> None:
        """Append an agent_thought update and bump the cumulative counter."""
        self.thought_text += text
        self.thought_prefix += text
        self.thought_total += len(text)
        if len(self.thought_text) > MAX_THOUGHT_TEXT_CHARS:
            self.thought_text = self.thought_text[-MAX_THOUGHT_TEXT_CHARS:]
        if len(self.thought_prefix) > MAX_THOUGHT_TEXT_CHARS:
            self.thought_prefix = self.thought_prefix[:MAX_THOUGHT_TEXT_CHARS]

    def final_reply_text(self, reply: str) -> str:
        """Return the final reply with any leading thought text removed."""
        if not self.thought_total:
            return reply
        # For short thoughts the full thought_text is still accurate; for long
        # (capped) thoughts we verify with the first MAX chars prefix.
        if self.thought_total <= MAX_THOUGHT_TEXT_CHARS:
            thought = self.thought_text
        else:
            thought = self.thought_prefix
        if thought and reply.startswith(thought):
            return reply[self.thought_total :].lstrip("\n")
        return reply


def final_segment_reply(result: Any, reply: str) -> str | None:
    """Return the reply text produced after the last tool-call update.

    ``result.updates`` is a bounded tail of the turn's session/update
    stream (oldest entries are dropped, newest kept), so the last
    ``tool_call*`` update visible in it is still the last boundary and the
    agent_message text after it is complete.  That suffix of the reply is
    the "final segment" — the post-investigation answer without the working
    narration between tool calls.  Returns None when no tool boundary is
    visible or nothing was said after it; callers then retain the full
    reply.
    """
    updates = list(getattr(result, "updates", None) or [])
    last_tool = -1
    for i, update in enumerate(updates):
        if update.get("sessionUpdate") in ("tool_call", "tool_call_update"):
            last_tool = i
    if last_tool < 0:
        return None
    seg_chars = 0
    for update in updates[last_tool + 1 :]:
        if update.get("sessionUpdate") not in ("agent_message", "agent_message_chunk"):
            continue
        content = update.get("content", {})
        if isinstance(content, list):
            seg_chars += sum(len(b.get("text", "")) for b in content if b.get("type") == "text")
        elif isinstance(content, dict) and content.get("type") == "text":
            seg_chars += len(content.get("text", ""))
    if seg_chars <= 0 or not reply:
        return None
    return reply[-seg_chars:] if seg_chars < len(reply) else reply


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
    thought_prefix: str = ""
    thought_total: int = 0
    full_text_offset: int = 0
    updated_at: float = 0.0
    current_intent: str = ""
    last_side_effect: str = ""
    last_side_effect_at: float = 0.0
    side_effects: list[dict[str, Any]] = field(default_factory=list)

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
