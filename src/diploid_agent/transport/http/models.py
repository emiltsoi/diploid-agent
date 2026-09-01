"""Pydantic request/response models for the HTTP transport."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    chat_id: str = Field(..., description="Stable identifier for the conversation")
    message: str = Field(..., description="User message")
    model: str | None = Field(
        None,
        description="Override model; starts a new Devin session if changed",
    )
    persona: str | None = Field(None, description="Override persona; not yet supported")
    reply_to: str | None = Field(
        None,
        description="Text of the message this is a reply to",
    )
    reply_to_is_bot: bool | None = Field(
        None,
        description="Whether the replied-to message was sent by the assistant",
    )
    reply_to_message_id: int | None = Field(
        None,
        description="Telegram message_id of the replied-to message",
    )


__all__ = [
    "BranchRequest",
    "ChatRequest",
    "ChatResponse",
    "ContinueRequest",
    "DispatchRequest",
    "GracefulRestartRequest",
    "McpCommandRequest",
    "OutboxResponse",
    "PlanCreateRequest",
    "PlanCreateTask",
    "PlanResponse",
    "PlanTaskDoneRequest",
    "PlanTaskStartRequest",
    "PluginAddRequest",
    "PluginCommandRequest",
    "PluginCreateRequest",
    "PluginEnableRequest",
    "PluginIncidentCreateRequest",
    "PluginIncidentListResponse",
    "PluginListResponse",
    "PluginRollbackRequest",
    "PluginSandboxRequest",
    "PluginToggleRequest",
    "PluginUpdateRequest",
    "RecallRequest",
    "RestartRequest",
    "ResumeRequest",
    "RetainRequest",
    "RuntimeStatusResponse",
    "SkillCommandRequest",
    "StateEventRequest",
    "StopRequest",
    "SubagentRequest",
    "SwitchModelRequest",
    "TaskResponse",
    "TimerRequest",
    "WakeRequest",
]


class ChatResponse(BaseModel):
    reply: str
    notice: str | None = None
    continuation: bool = False
    dispatch_id: str | None = None
    session_id: str | None = None
    session_number: int | None = None
    turn_number: int | None = None
    metrics: dict[str, Any] | None = None


class OutboxResponse(BaseModel):
    chat_id: str | None = None
    result: ChatResponse | None = None


class DispatchRequest(BaseModel):
    chat_id: str
    context: str | None = None


class ContinueRequest(BaseModel):
    dispatch_id: str
    result: str


class WakeRequest(BaseModel):
    chat_id: str
    reason: str = "user_request"
    event_id: str | None = None
    silent: bool | None = None


class SwitchModelRequest(BaseModel):
    chat_id: str
    model: str


class ResumeRequest(BaseModel):
    chat_id: str
    session_number: int


class BranchRequest(BaseModel):
    chat_id: str
    session_number: int


class StopRequest(BaseModel):
    chat_id: str


class RestartRequest(BaseModel):
    chat_id: str


class GracefulRestartRequest(BaseModel):
    chat_id: str
    service: str | None = None
    reason: str = ""


class SubagentRequest(BaseModel):
    chat_id: str
    prompt: str
    context: str | None = None
    model: str | None = None
    cwd: str | None = None
    acp_timeout: float | None = None


class TimerRequest(BaseModel):
    chat_id: str
    reason: str
    scheduled_at: float
    payload: dict[str, Any] = Field(default_factory=dict)
    silent: bool = True
    priority: int = 1


class RuntimeStatusResponse(BaseModel):
    instance_id: str
    started_at: float
    uptime_seconds: float
    event_bus_running: bool
    timer_running: bool
    task_engine_active: bool
    plan_count: int
    pending_wake_count: int
    active_chat_count: int


class StateEventRequest(BaseModel):
    chat_id: str
    plugin: str
    event: str
    params: dict[str, Any] | None = None


class RecallRequest(BaseModel):
    chat_id: str
    query: str
    tags: list[str] = Field(default_factory=list)
    max_tokens: int | None = None


class RetainRequest(BaseModel):
    chat_id: str
    content: str
    tags: list[str] = Field(default_factory=list)
    context: str | None = None


class McpCommandRequest(BaseModel):
    chat_id: str
    command: str = "list"
    name: str | None = None


class SkillCommandRequest(BaseModel):
    chat_id: str
    command: str = "list"
    name: str | None = None
    content: str | None = None


class PluginListResponse(BaseModel):
    plugins: list[dict[str, Any]]


class PluginCommandRequest(BaseModel):
    chat_id: str
    name: str


class PluginEnableRequest(BaseModel):
    chat_id: str
    name: str
    enabled: bool


class PluginAddRequest(BaseModel):
    chat_id: str | None = None
    plugin: dict[str, Any]
    dry_run: bool = False


class PluginUpdateRequest(BaseModel):
    chat_id: str | None = None
    name: str
    plugin: dict[str, Any]
    dry_run: bool = False


class PluginToggleRequest(BaseModel):
    chat_id: str | None = None
    name: str
    enabled: bool


class PluginRollbackRequest(BaseModel):
    steps: int = Field(default=1, ge=1, description="Number of snapshots to roll back")


class PluginSandboxRequest(BaseModel):
    chat_id: str | None = None
    module: str
    plugin: dict[str, Any] | None = None


class PluginCreateRequest(BaseModel):
    name: str
    module: str | None = None
    prompt_slot: str = "self_state"
    state_file: str | None = None
    mcp_server: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


class PluginIncidentListResponse(BaseModel):
    incidents: list[dict[str, Any]]


class PluginIncidentCreateRequest(BaseModel):
    plugin: str
    phase: str
    error: str
    action: str = ""
    chat_id: str = ""


class PlanCreateTask(BaseModel):
    """A task definition used when creating a plan over HTTP."""

    name: str
    description: str = ""
    type: str = "shell"
    command: str = ""
    depends_on: list[str] = Field(default_factory=list)
    cwd: str | None = None
    acp_model: str | None = None


class PlanCreateRequest(BaseModel):
    name: str
    description: str = ""
    chat_id: str | None = None
    tasks: list[PlanCreateTask] = Field(default_factory=list)


class PlanTaskStartRequest(BaseModel):
    plan_id: str
    task_id: str | None = None


class PlanTaskDoneRequest(BaseModel):
    plan_id: str
    task_id: str
    result: str = ""
    log: str = ""


class TaskResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    result: str | None = None
    log: str = ""
    acp_model: str | None = None


class PlanResponse(BaseModel):
    id: str
    name: str
    status: str
    chat_id: str | None = None
    tasks: list[TaskResponse]
    created_at: float
    updated_at: float
