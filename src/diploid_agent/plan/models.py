"""Plan and task models for AgentOS Phase 1."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskType(str, Enum):
    SHELL = "shell"
    NOOP = "noop"
    ACP = "acp"
    SUBAGENT = "subagent"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """One unit of work inside a plan."""

    id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:12]}")
    name: str
    description: str = ""
    type: TaskType = TaskType.SHELL
    command: str = ""
    prompt: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    log: str = ""
    started_at: float | None = None
    completed_at: float | None = None
    created_at: float = Field(default_factory=time.time)
    cwd: Path | None = None
    stop_reason: str | None = None
    cancelled: bool = False
    partial: bool = False
    timed_out: bool = False
    chat_id: str | None = None
    acp_model: str | None = None
    acp_timeout: float | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    dispatch_id: str | None = None

    @field_validator("cwd")
    @classmethod
    def expand_cwd(cls, v: Path | None) -> Path | None:
        return v.expanduser() if v is not None else None

    @field_validator("acp_model")
    @classmethod
    def _strip_acp_model(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        return v if v else None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls.model_validate(data)


class Plan(BaseModel):
    """A living plan made of tasks."""

    id: str = Field(default_factory=lambda: f"plan-{uuid.uuid4().hex[:12]}")
    name: str
    description: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    chat_id: str | None = None
    tasks: list[Task] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        return cls.model_validate(data)
