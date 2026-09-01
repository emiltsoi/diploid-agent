"""HTTP transport for the conversational harness.

Provides a generic `/chat` endpoint and a Telegram-compatible `/webhook`.
The generic endpoint lets any caller (Telegram bot, other clients, curl) send a
message and receive a reply.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import PlainTextResponse

from diploid_agent.config import (
    Config,
    ConfigPersistenceError,
    NotificationsConfig,
    PluginConfig,
    TaskConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.models import ChatResult, RuntimeStatus, WakeEvent
from diploid_agent.plan.models import Plan, Task
from diploid_agent.plugins import PluginManager
from diploid_agent.runtime.agent_runtime import AgentRuntime
from diploid_agent.transport.base import OutboundMessage, RuntimeAPI, Transport
from diploid_agent.transport.command_handler import CommandHandler


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


def _to_response(result: Any) -> ChatResponse:
    """Convert a ChatResult or bare reply string into a ChatResponse."""
    if isinstance(result, str):
        return ChatResponse(reply=result)
    return ChatResponse(
        reply=result.reply,
        notice=result.notice,
        continuation=result.continuation,
        dispatch_id=result.dispatch_id,
        session_id=result.session_id,
        session_number=result.session_number,
        turn_number=result.turn_number,
        metrics=result.metrics,
    )


def _task_to_response(task: Task) -> TaskResponse:
    return TaskResponse(
        id=task.id,
        name=task.name,
        type=task.type.value,
        status=task.status.value,
        result=task.result,
        log=task.log,
        acp_model=task.acp_model,
    )


def _plan_to_response(plan: Plan) -> PlanResponse:
    return PlanResponse(
        id=plan.id,
        name=plan.name,
        status=plan.status.value,
        chat_id=plan.chat_id,
        tasks=[_task_to_response(t) for t in plan.tasks],
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


def create_app(config: Config, runtime: RuntimeAPI | None = None) -> FastAPI:
    runtime = runtime or AgentRuntime(config)
    command_handler = CommandHandler(runtime=runtime)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            runtime.start()
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(title="diploid-agent", lifespan=lifespan)
    app.state.runtime = runtime
    # Backward-compatible alias used by existing tests.
    app.state.harness = runtime

    def _require_api_key(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> None:
        """Require X-API-Key on POST endpoints when HARNESS_API_KEY is configured."""
        token = config.secrets.harness_api_key if config.secrets else None
        if token is None:
            return
        if x_api_key != token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing X-API-Key",
            )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return command_handler.call(
            method="health",
            requires_chat_id=False,
            catch=False,
        )

    @app.post("/ingress/{protocol}")
    async def ingress_route(protocol: str, request: Request) -> Response:
        """Generic pluggable ingress route."""
        return await runtime.handle_ingress(protocol, request)

    @app.post("/mesh/receive")
    async def mesh_receive(request: Request) -> Response:
        """Hermes-compatible mesh receive endpoint."""
        return await runtime.handle_ingress("mesh", request)

    @app.post("/plugins/openclaw-mesh/webhook")
    async def openclaw_mesh_webhook(request: Request) -> Response:
        """OpenClaw-compatible mesh receive alias."""
        return await runtime.handle_ingress("mesh", request)

    @app.get("/prometheus")
    def prometheus() -> PlainTextResponse:
        raw = command_handler.call(
            method="get_prometheus_metrics",
            requires_chat_id=False,
            catch=False,
        )
        return PlainTextResponse(raw)

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def chat(req: ChatRequest) -> ChatResponse:
        # The caller (e.g. the Telegram long-polling bot) is responsible for
        # delivering the reply, so suppress the runtime's own outbound notifier.
        result = runtime.process(
            req.chat_id,
            req.message,
            model=req.model,
            reply_to=req.reply_to,
            reply_to_is_bot=req.reply_to_is_bot,
            reply_to_message_id=req.reply_to_message_id,
            notify=False,
        )
        return _to_response(result)

    @app.post("/dispatch", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def dispatch(req: DispatchRequest) -> ChatResponse:
        raw = command_handler.call(
            method="dispatch",
            chat_id=req.chat_id,
            context=req.context,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/continue", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def continue_turn(req: ContinueRequest) -> ChatResponse:
        raw = command_handler.call(
            method="continue_turn",
            dispatch_id=req.dispatch_id,
            result=req.result,
            requires_chat_id=False,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/wake", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def wake(req: WakeRequest) -> ChatResponse:
        raw = command_handler.call(
            method="wake",
            chat_id=req.chat_id,
            event_id=req.event_id,
            reason=req.reason,
            silent=req.silent,
            catch=False,
        )
        if raw.reply == "Chat is busy; wake re-enqueued.":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=raw.reply,
            )
        if raw.reply == "Unknown or already completed wake event.":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=raw.reply,
            )
        if req.event_id is not None:
            runtime.wake_queue.complete(req.event_id)
        return _to_response(raw)

    @app.get("/models")
    def models() -> dict[str, list[str]]:
        raw = command_handler.call(
            method="list_models",
            http_path="/models",
            http_method="GET",
            requires_chat_id=False,
            catch=False,
        )
        if isinstance(raw, list):
            return {"models": raw}
        return raw

    @app.post(
        "/switch-model", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def switch_model(req: SwitchModelRequest) -> ChatResponse:
        raw = command_handler.call(
            method="switch_model",
            chat_id=req.chat_id,
            model=req.model,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/new/{chat_id}", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def new_session(chat_id: str) -> ChatResponse:
        raw = command_handler.call(
            method="new_session",
            chat_id=chat_id,
            http_path="/new/{chat_id}",
            catch=False,
        )
        return _to_response(raw)

    @app.get("/sessions/{chat_id}")
    def sessions(chat_id: str) -> dict[str, Any]:
        return command_handler.call(
            method="list_sessions",
            chat_id=chat_id,
            http_path="/sessions/{chat_id}",
            http_method="GET",
            catch=False,
        )

    @app.get("/subagents/{chat_id}")
    def subagents(chat_id: str) -> dict[str, Any]:
        return command_handler.call(
            method="subagent_status",
            chat_id=chat_id,
            http_path="/subagents/{chat_id}",
            http_method="GET",
            catch=False,
        )

    @app.get("/outbox/{chat_id}", response_model=OutboxResponse)
    def outbox(chat_id: str, wait: float = Query(0.0, ge=0, le=60)) -> OutboxResponse:
        raw = command_handler.call(
            method="outbox_pop",
            chat_id=chat_id,
            http_path="/outbox/{chat_id}",
            http_method="GET",
            wait=wait,
            catch=False,
        )
        if raw is None:
            return OutboxResponse(chat_id=chat_id, result=None)
        return OutboxResponse(chat_id=chat_id, result=_to_response(raw))

    @app.post("/resume", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def resume(req: ResumeRequest) -> ChatResponse:
        raw = command_handler.call(
            method="resume_session",
            chat_id=req.chat_id,
            session_number=req.session_number,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/branch", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def branch(req: BranchRequest) -> ChatResponse:
        raw = command_handler.call(
            method="branch_session",
            chat_id=req.chat_id,
            session_number=req.session_number,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/stop", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def stop(req: StopRequest) -> ChatResponse:
        raw = command_handler.call(
            method="stop",
            chat_id=req.chat_id,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/restart", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def restart(req: RestartRequest) -> ChatResponse:
        raw = command_handler.call(
            method="restart",
            chat_id=req.chat_id,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/graceful-restart",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def graceful_restart(req: GracefulRestartRequest) -> ChatResponse:
        raw = command_handler.call(
            method="graceful_service_restart",
            chat_id=req.chat_id,
            service=req.service,
            reason=req.reason,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/subagent",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def subagent(req: SubagentRequest) -> ChatResponse:
        raw = command_handler.call(
            method="subagent_start",
            chat_id=req.chat_id,
            prompt=req.prompt,
            context=req.context,
            model=req.model,
            cwd=Path(req.cwd) if req.cwd else None,
            acp_timeout=req.acp_timeout,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/state", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def state_event(req: StateEventRequest) -> ChatResponse:
        raw = command_handler.call(
            method="plugin_event",
            chat_id=req.chat_id,
            plugin=req.plugin,
            event=req.event,
            catch=False,
            **(req.params or {}),
        )
        return _to_response(raw)

    @app.get("/turn/{chat_id}")
    def turn_status(
        chat_id: str, wait: float = Query(0.0, ge=0, le=60, description="Long-poll wait in seconds")
    ) -> dict[str, Any]:
        """Return the partial state of an active turn for streaming clients."""
        return command_handler.call(
            method="turn_status",
            chat_id=chat_id,
            wait=wait,
            http_path="/turn/{chat_id}",
            http_method="GET",
            catch=False,
        )

    @app.get("/status/{chat_id}")
    def chat_status(chat_id: str) -> dict[str, object]:
        return command_handler.call(
            method="status",
            chat_id=chat_id,
            http_path="/status/{chat_id}",
            http_method="GET",
            catch=False,
        )

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return command_handler.call(
            method="get_metrics",
            http_path="/metrics",
            http_method="GET",
            requires_chat_id=False,
            catch=False,
        )

    @app.get("/metrics/{chat_id}")
    def chat_metrics(chat_id: str) -> dict[str, Any]:
        return command_handler.call(
            method="get_metrics",
            chat_id=chat_id,
            http_path="/metrics/{chat_id}",
            http_method="GET",
            catch=False,
        )

    @app.get("/mcp/{chat_id}")
    def mcp_get(chat_id: str) -> ChatResponse:
        raw = command_handler.call(
            method="mcp_list",
            chat_id=chat_id,
            http_path="/mcp/{chat_id}",
            http_method="GET",
            catch=False,
        )
        if isinstance(raw, dict):
            return ChatResponse(reply=raw.get("reply", ""))
        return ChatResponse(reply=raw)

    @app.post("/mcp", dependencies=[Depends(_require_api_key)])
    def mcp_post(req: McpCommandRequest) -> ChatResponse:
        if req.command == "list":
            raw = command_handler.call(
                method="mcp_list",
                chat_id=req.chat_id,
                http_path="/mcp/{chat_id}",
                http_method="GET",
                catch=False,
            )
        elif req.command == "enable" and req.name:
            raw = command_handler.call(
                method="mcp_enable",
                chat_id=req.chat_id,
                name=req.name,
                catch=False,
            )
        elif req.command == "disable" and req.name:
            raw = command_handler.call(
                method="mcp_disable",
                chat_id=req.chat_id,
                name=req.name,
                catch=False,
            )
        else:
            return ChatResponse(
                reply="Usage: command=list|enable|disable, name required for enable/disable"
            )
        if isinstance(raw, ChatResult):
            return _to_response(raw)
        if isinstance(raw, dict):
            return ChatResponse(reply=raw.get("reply", ""))
        return ChatResponse(reply=str(raw))

    @app.get("/skill/{chat_id}")
    def skill_get(chat_id: str) -> ChatResponse:
        raw = command_handler.call(
            method="skill_list",
            chat_id=chat_id,
            http_path="/skill/{chat_id}",
            http_method="GET",
            catch=False,
        )
        if isinstance(raw, dict):
            return ChatResponse(reply=raw.get("reply", ""))
        return ChatResponse(reply=raw)

    @app.post("/skill", dependencies=[Depends(_require_api_key)])
    def skill_post(req: SkillCommandRequest) -> ChatResponse:
        if req.command == "list":
            raw = command_handler.call(
                method="skill_list",
                chat_id=req.chat_id,
                http_path="/skill/{chat_id}",
                http_method="GET",
                catch=False,
            )
        elif req.command == "enable" and req.name:
            raw = command_handler.call(
                method="skill_enable",
                chat_id=req.chat_id,
                name=req.name,
                catch=False,
            )
        elif req.command == "disable" and req.name:
            raw = command_handler.call(
                method="skill_disable",
                chat_id=req.chat_id,
                name=req.name,
                catch=False,
            )
        elif req.command == "create" and req.name and req.content:
            raw = command_handler.call(
                method="skill_create",
                chat_id=req.chat_id,
                name=req.name,
                content=req.content,
                catch=False,
            )
        else:
            return ChatResponse(
                reply="Usage: command=list|enable|disable|create, name and content required for create"
            )
        if isinstance(raw, ChatResult):
            return _to_response(raw)
        if isinstance(raw, dict):
            return ChatResponse(reply=raw.get("reply", ""))
        return ChatResponse(reply=str(raw))

    @app.get("/plugins/{chat_id}", response_model=PluginListResponse)
    def plugin_list(chat_id: str) -> PluginListResponse:
        raw = command_handler.call(
            method="plugin_list",
            chat_id=chat_id,
            http_path="/plugins/{chat_id}",
            http_method="GET",
            catch=False,
        )
        return PluginListResponse(plugins=raw if isinstance(raw, list) else [])

    @app.post(
        "/plugin/enable", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def plugin_enable(req: PluginEnableRequest) -> ChatResponse:
        raw = command_handler.call(
            method="plugin_set_enabled",
            chat_id=req.chat_id,
            name=req.name,
            enabled=req.enabled,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/plugin/reload", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def plugin_reload(req: PluginCommandRequest) -> ChatResponse:
        raw = command_handler.call(
            method="plugin_reload",
            chat_id=req.chat_id,
            name=req.name,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/plugins",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_add(req: PluginAddRequest) -> ChatResponse:
        config = PluginConfig(**req.plugin)
        if req.dry_run:
            try:
                PluginManager.validate_module(config.module)
                return _to_response(ChatResult(reply=f"Dry run OK for {config.name}"))
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                ) from exc
        try:
            raw = command_handler.call(
                method="plugin_add",
                config=config,
                requires_chat_id=False,
                catch=False,
            )
            return _to_response(raw)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post(
        "/plugin/sandbox",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_sandbox(req: PluginSandboxRequest) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_sandbox",
                module=req.module,
                plugin=req.plugin or {},
                requires_chat_id=False,
                catch=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return _to_response(ChatResult(reply=json.dumps(raw, ensure_ascii=False)))

    @app.post(
        "/plugins/create",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_create(req: PluginCreateRequest) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_create",
                name=req.name,
                module=req.module,
                prompt_slot=req.prompt_slot,
                state_file=req.state_file,
                mcp_server=req.mcp_server,
                config=req.config,
                requires_chat_id=False,
                catch=False,
            )
            return _to_response(ChatResult(reply=json.dumps(raw, ensure_ascii=False)))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.delete(
        "/plugins/{name}",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_remove(name: str) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_remove",
                name=name,
                requires_chat_id=False,
                catch=False,
            )
            return _to_response(raw)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.patch(
        "/plugins/{name}",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_update(name: str, req: PluginUpdateRequest) -> ChatResponse:
        if req.name != name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name mismatch")
        try:
            # Validate the merged config before mutating runtime state.
            merged = dict(req.plugin)
            merged["name"] = name
            config = PluginConfig(**merged)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        if req.dry_run:
            try:
                PluginManager.validate_module(config.module)
                return _to_response(ChatResult(reply=f"Dry run OK for {name}"))
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
                ) from exc
        # Validate the module before applying the update.
        try:
            PluginManager.validate_module(config.module)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Merge via the existing update_plugins_config path, then reload.
        patch = {"plugins": [{"name": name, **req.plugin}]}
        command_handler.call(
            method="update_config",
            patch=patch,
            requires_chat_id=False,
            catch=False,
        )
        command_handler.call(
            method="plugin_reload",
            name=name,
            chat_id="0",  # chat_id is ignored by reload for modules
            catch=False,
        )
        return _to_response(ChatResult(reply=f"Plugin {name} updated"))

    @app.post(
        "/plugins/{name}/toggle",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_toggle(name: str, req: PluginToggleRequest) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_toggle",
                name=name,
                enabled=req.enabled,
                chat_id=req.chat_id,
                catch=False,
            )
            return _to_response(raw)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post(
        "/config/rollback",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def config_rollback(req: PluginRollbackRequest) -> ChatResponse:
        try:
            raw = command_handler.call(
                method="plugin_rollback",
                steps=req.steps,
                requires_chat_id=False,
                catch=False,
            )
            return _to_response(raw)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get(
        "/plugin-incidents",
        response_model=PluginIncidentListResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_incidents() -> PluginIncidentListResponse:
        raw = command_handler.call(
            method="incidents",
            requires_chat_id=False,
            catch=False,
        )
        return PluginIncidentListResponse(incidents=raw if isinstance(raw, list) else [])

    @app.get(
        "/plugin-incidents/{plugin_name}",
        response_model=PluginIncidentListResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_incidents_for(plugin_name: str) -> PluginIncidentListResponse:
        return PluginIncidentListResponse(incidents=runtime.incidents_for_plugin(plugin_name))

    @app.post(
        "/plugin-incidents",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_incident_create(req: PluginIncidentCreateRequest) -> ChatResponse:
        runtime.record_incident(
            plugin=req.plugin,
            phase=req.phase,
            error=req.error,
            action=req.action,
            chat_id=req.chat_id,
        )
        return _to_response(ChatResult(reply="incident recorded"))

    @app.get("/memory/{chat_id}")
    def memory(chat_id: str) -> dict[str, object]:
        raw = command_handler.call(
            method="memory",
            chat_id=chat_id,
            http_path="/memory/{chat_id}",
            http_method="GET",
            catch=False,
        )
        if isinstance(raw, str):
            return {"chat_id": chat_id, "memory": raw}
        if isinstance(raw, dict):
            return {"chat_id": chat_id, "memory": raw.get("memory", "")}
        return {"chat_id": chat_id, "memory": ""}

    @app.post("/summarize/{chat_id}", dependencies=[Depends(_require_api_key)])
    def summarize(chat_id: str) -> ChatResponse:
        raw = command_handler.call(
            method="summarize",
            chat_id=chat_id,
            http_path="/summarize/{chat_id}",
            catch=False,
        )
        return _to_response(raw)

    @app.post("/recall", dependencies=[Depends(_require_api_key)])
    def recall(req: RecallRequest) -> ChatResponse:
        raw = command_handler.call(
            method="recall",
            chat_id=req.chat_id,
            query=req.query,
            tags=req.tags,
            max_tokens=req.max_tokens,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/retain", dependencies=[Depends(_require_api_key)])
    def retain(req: RetainRequest) -> ChatResponse:
        raw = command_handler.call(
            method="retain",
            chat_id=req.chat_id,
            content=req.content,
            tags=req.tags,
            context=req.context,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/promote", dependencies=[Depends(_require_api_key)])
    def promote(req: ChatRequest) -> ChatResponse:
        raw = command_handler.call(
            method="promote",
            chat_id=req.chat_id,
            fact=req.message,
            http_body={"message": req.message},
            catch=False,
        )
        return _to_response(raw)

    @app.post("/webhook")
    async def telegram_webhook(request: Request) -> dict[str, object]:
        """Minimal Telegram webhook: extracts text and chat_id from update."""
        payload = await request.json()
        message = payload.get("message") or payload.get("edited_message") or {}
        chat = message.get("chat", {})
        chat_id = str(chat.get("id"))
        text = message.get("text", "")

        if not chat_id or not text:
            return {"ok": False, "error": "missing chat_id or text"}

        reply_to = message.get("reply_to_message", {})
        reply_to_text = reply_to.get("text", "") or reply_to.get("caption", "")
        reply_to_is_bot = reply_to.get("from", {}).get("is_bot")
        reply_to_message_id = reply_to.get("message_id")

        result = await run_in_threadpool(
            runtime.process,
            chat_id,
            text,
            reply_to=reply_to_text or None,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
        )
        return {
            "ok": True,
            "reply": result.reply,
            "notice": result.notice,
            "metrics": result.metrics,
        }

    @app.post("/plan/create", response_model=PlanResponse, dependencies=[Depends(_require_api_key)])
    def plan_create(req: PlanCreateRequest) -> PlanResponse:
        tasks = []
        for t in req.tasks:
            task_data = t.model_dump()
            task_data["chat_id"] = req.chat_id
            tasks.append(Task.model_validate(task_data))
        try:
            raw = command_handler.call(
                method="plan_create",
                name=req.name,
                description=req.description,
                chat_id=req.chat_id,
                tasks=tasks,
                catch=False,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _plan_to_response(raw)

    @app.post(
        "/plan/task/start", response_model=TaskResponse, dependencies=[Depends(_require_api_key)]
    )
    def plan_task_start(req: PlanTaskStartRequest) -> TaskResponse:
        try:
            raw = command_handler.call(
                method="plan_task_start",
                plan_id=req.plan_id,
                task_id=req.task_id,
                requires_chat_id=False,
                catch=False,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _task_to_response(raw)

    @app.post(
        "/plan/task/done", response_model=TaskResponse, dependencies=[Depends(_require_api_key)]
    )
    def plan_task_done(req: PlanTaskDoneRequest) -> TaskResponse:
        try:
            raw = command_handler.call(
                method="plan_task_done",
                plan_id=req.plan_id,
                task_id=req.task_id,
                result=req.result,
                log=req.log,
                requires_chat_id=False,
                catch=False,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _task_to_response(raw)

    @app.get("/plan/list", response_model=list[PlanResponse])
    def plan_list() -> list[PlanResponse]:
        try:
            raw = command_handler.call(
                method="plan_list",
                requires_chat_id=False,
                catch=False,
            )
        except AttributeError:
            raw = []
        plans = raw if isinstance(raw, list) else []
        return [_plan_to_response(p) for p in plans]

    @app.get("/plan/{plan_id}", response_model=PlanResponse)
    def plan_get(plan_id: str) -> PlanResponse:
        try:
            raw = command_handler.call(
                method="plan_get",
                plan_id=plan_id,
                requires_chat_id=False,
                catch=False,
            )
        except AttributeError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Plan retrieval not supported",
            )
        if raw is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plan {plan_id} not found",
            )
        return _plan_to_response(raw)

    @app.post(
        "/runtime/start",
        dependencies=[Depends(_require_api_key)],
    )
    def runtime_start() -> dict[str, bool]:
        command_handler.call(
            method="start",
            requires_chat_id=False,
            catch=False,
        )
        return {"ok": True}

    @app.post(
        "/runtime/stop",
        dependencies=[Depends(_require_api_key)],
    )
    def runtime_stop() -> dict[str, bool]:
        command_handler.call(
            method="shutdown",
            requires_chat_id=False,
            catch=False,
        )
        return {"ok": True}

    @app.get("/runtime/status", response_model=RuntimeStatusResponse)
    def runtime_status() -> RuntimeStatus:
        return command_handler.call(
            method="get_status",
            requires_chat_id=False,
            catch=False,
        )

    @app.get("/config")
    def config_get() -> dict[str, Any]:
        """Return the current live runtime configuration (excluding secrets)."""
        return command_handler.call(
            method="get_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.patch(
        "/config",
        dependencies=[Depends(_require_api_key)],
    )
    def config_update(patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial live update to Telegram and/or plugin configuration."""
        try:
            return command_handler.call(
                method="update_config",
                patch=patch,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    @app.get(
        "/task/config",
        response_model=TaskConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def task_config_get() -> TaskConfig:
        return command_handler.call(
            method="get_task_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/task/config",
        response_model=TaskConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def task_config_update(req: TaskConfig) -> TaskConfig:
        try:
            command_handler.call(
                method="update_task_config",
                task_config=req,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return command_handler.call(
            method="get_task_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.get(
        "/waker/config",
        response_model=WakerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def waker_config_get() -> WakerConfig:
        return command_handler.call(
            method="get_waker_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/waker/config",
        response_model=WakerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def waker_config_update(req: WakerConfig) -> WakerConfig:
        try:
            command_handler.call(
                method="update_waker_config",
                waker_config=req,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return command_handler.call(
            method="get_waker_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.get(
        "/timer/config",
        response_model=TimerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def timer_config_get() -> TimerConfig:
        return command_handler.call(
            method="get_timer_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/timer/config",
        response_model=TimerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def timer_config_update(req: TimerConfig) -> TimerConfig:
        try:
            command_handler.call(
                method="update_timer_config",
                timer_config=req,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return command_handler.call(
            method="get_timer_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.get(
        "/notifications/config",
        response_model=NotificationsConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def notifications_config_get() -> NotificationsConfig:
        return command_handler.call(
            method="get_notifications_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/notifications/config",
        response_model=NotificationsConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def notifications_config_update(req: NotificationsConfig) -> NotificationsConfig:
        try:
            command_handler.call(
                method="update_notifications_config",
                notifications_config=req,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return command_handler.call(
            method="get_notifications_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post("/timer", dependencies=[Depends(_require_api_key)])
    def timer_create(req: TimerRequest) -> dict[str, str]:
        event = WakeEvent(
            id="",
            chat_id=req.chat_id,
            reason=req.reason,
            priority=req.priority,
            scheduled_at=req.scheduled_at,
            payload=req.payload,
            silent=req.silent,
            created_at=time.time(),
            ready=True,
        )
        enqueued = runtime.wake_queue.enqueue(event)
        return {"event_id": enqueued.id}

    return app


class HttpTransport(Transport):
    """HTTP transport exposing the harness as a FastAPI application."""

    def __init__(self, config: Config, runtime: RuntimeAPI | None = None) -> None:
        self._config = config
        self._runtime = runtime
        self._app: FastAPI | None = None
        self._server: uvicorn.Server | None = None

    def start(self, runtime: RuntimeAPI | None = None) -> None:
        if runtime is not None:
            self._runtime = runtime
        self._app = create_app(self._config, self._runtime)
        uvicorn.run(
            self._app,
            host=self._config.harness.listen_host,
            port=self._config.harness.listen_port,
        )

    def stop(self) -> None:
        """HTTP transport currently has no running background server to stop."""
        return

    def send(self, message: OutboundMessage) -> None:
        """HTTP transport does not send outbound messages; it returns None."""
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "harness.yaml",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    transport = HttpTransport(config)
    transport.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
