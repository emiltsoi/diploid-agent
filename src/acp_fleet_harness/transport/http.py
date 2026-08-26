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
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import PlainTextResponse

from acp_fleet_harness.config import (
    Config,
    ConfigPersistenceError,
    NotificationsConfig,
    PluginConfig,
    TaskConfig,
    TimerConfig,
    WakerConfig,
)
from acp_fleet_harness.models import ChatResult, RuntimeStatus, WakeEvent
from acp_fleet_harness.plan.models import Plan, Task
from acp_fleet_harness.plugins import PluginManager
from acp_fleet_harness.runtime.agent_runtime import AgentRuntime
from acp_fleet_harness.transport.base import OutboundMessage, RuntimeAPI, Transport


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
    dispatch_id: str | None = None
    session_id: str | None = None
    session_number: int | None = None
    turn_number: int | None = None
    metrics: dict[str, Any] | None = None


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

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            runtime.start()
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(title="acp-fleet-harness", lifespan=lifespan)
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
        return runtime.health()

    @app.get("/prometheus")
    def prometheus() -> PlainTextResponse:
        return PlainTextResponse(runtime.get_prometheus_metrics())

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
        return _to_response(runtime.dispatch(req.chat_id, context=req.context))

    @app.post("/continue", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def continue_turn(req: ContinueRequest) -> ChatResponse:
        return _to_response(runtime.continue_turn(req.dispatch_id, req.result))

    @app.post("/wake", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def wake(req: WakeRequest) -> ChatResponse:
        result = runtime.wake(
            req.chat_id,
            event_id=req.event_id,
            reason=req.reason,
            silent=req.silent,
        )
        if result.reply == "Chat is busy; wake re-enqueued.":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.reply,
            )
        if result.reply == "Unknown or already completed wake event.":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=result.reply,
            )
        if req.event_id is not None:
            runtime.wake_queue.complete(req.event_id)
        return _to_response(result)

    @app.get("/models")
    def models() -> dict[str, list[str]]:
        return {"models": runtime.list_models()}

    @app.post(
        "/switch-model", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def switch_model(req: SwitchModelRequest) -> ChatResponse:
        result = runtime.switch_model(req.chat_id, req.model)
        return _to_response(result)

    @app.post(
        "/new/{chat_id}", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def new_session(chat_id: str) -> ChatResponse:
        return _to_response(runtime.new_session(chat_id))

    @app.get("/sessions/{chat_id}")
    def sessions(chat_id: str) -> dict[str, Any]:
        return runtime.list_sessions(chat_id)

    @app.post("/resume", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def resume(req: ResumeRequest) -> ChatResponse:
        return _to_response(runtime.resume_session(req.chat_id, req.session_number))

    @app.post("/branch", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def branch(req: BranchRequest) -> ChatResponse:
        return _to_response(runtime.branch_session(req.chat_id, req.session_number))

    @app.post("/stop", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def stop(req: StopRequest) -> ChatResponse:
        return _to_response(runtime.stop(req.chat_id))

    @app.post("/state", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def state_event(req: StateEventRequest) -> ChatResponse:
        return _to_response(
            runtime.plugin_event(req.chat_id, req.plugin, event=req.event, **(req.params or {}))
        )

    @app.get("/turn/{chat_id}")
    def turn_status(
        chat_id: str, wait: float = Query(0.0, ge=0, le=60, description="Long-poll wait in seconds")
    ) -> dict[str, Any]:
        """Return the partial state of an active turn for streaming clients."""
        return runtime.turn_status(chat_id, wait=wait)

    @app.get("/status/{chat_id}")
    def chat_status(chat_id: str) -> dict[str, object]:
        return runtime.status(chat_id)

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return runtime.get_metrics()

    @app.get("/metrics/{chat_id}")
    def chat_metrics(chat_id: str) -> dict[str, Any]:
        return runtime.get_metrics(chat_id)

    @app.get("/mcp/{chat_id}")
    def mcp_get(chat_id: str) -> ChatResponse:
        return ChatResponse(reply=runtime.mcp_list(chat_id))

    @app.post("/mcp", dependencies=[Depends(_require_api_key)])
    def mcp_post(req: McpCommandRequest) -> ChatResponse:
        if req.command == "list":
            reply = runtime.mcp_list(req.chat_id)
        elif req.command == "enable" and req.name:
            reply = runtime.mcp_enable(req.chat_id, req.name)
        elif req.command == "disable" and req.name:
            reply = runtime.mcp_disable(req.chat_id, req.name)
        else:
            reply = "Usage: command=list|enable|disable, name required for enable/disable"
        return ChatResponse(reply=reply)

    @app.get("/skill/{chat_id}")
    def skill_get(chat_id: str) -> ChatResponse:
        return ChatResponse(reply=runtime.skill_list(chat_id))

    @app.post("/skill", dependencies=[Depends(_require_api_key)])
    def skill_post(req: SkillCommandRequest) -> ChatResponse:
        if req.command == "list":
            reply = runtime.skill_list(req.chat_id)
        elif req.command == "enable" and req.name:
            reply = runtime.skill_enable(req.chat_id, req.name)
        elif req.command == "disable" and req.name:
            reply = runtime.skill_disable(req.chat_id, req.name)
        elif req.command == "create" and req.name and req.content:
            reply = runtime.skill_create(req.chat_id, req.name, req.content)
        else:
            reply = (
                "Usage: command=list|enable|disable|create, name and content required for create"
            )
        return ChatResponse(reply=reply)

    @app.get("/plugins/{chat_id}", response_model=PluginListResponse)
    def plugin_list(chat_id: str) -> PluginListResponse:
        return PluginListResponse(plugins=runtime.plugin_list(chat_id))

    @app.post(
        "/plugin/enable", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def plugin_enable(req: PluginEnableRequest) -> ChatResponse:
        return _to_response(runtime.plugin_set_enabled(req.chat_id, req.name, req.enabled))

    @app.post(
        "/plugin/reload", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def plugin_reload(req: PluginCommandRequest) -> ChatResponse:
        return _to_response(runtime.plugin_reload(req.chat_id, req.name))

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
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        try:
            return _to_response(runtime.plugin_add(config))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post(
        "/plugin/sandbox",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_sandbox(req: PluginSandboxRequest) -> ChatResponse:
        try:
            result = runtime.plugin_sandbox(req.module, req.plugin or {})
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return _to_response(ChatResult(reply=json.dumps(result, ensure_ascii=False)))

    @app.delete(
        "/plugins/{name}",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_remove(name: str) -> ChatResponse:
        try:
            return _to_response(runtime.plugin_remove(name))
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
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        if req.dry_run:
            try:
                PluginManager.validate_module(config.module)
                return _to_response(ChatResult(reply=f"Dry run OK for {name}"))
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Validate the module before applying the update.
        try:
            PluginManager.validate_module(config.module)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Merge via the existing update_plugins_config path, then reload.
        patch = {"plugins": [{"name": name, **req.plugin}]}
        runtime.update_config(patch)
        runtime.plugin_reload(name=name, chat_id="0")  # chat_id is ignored by reload for modules
        return _to_response(ChatResult(reply=f"Plugin {name} updated"))

    @app.post(
        "/plugins/{name}/toggle",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_toggle(name: str, req: PluginToggleRequest) -> ChatResponse:
        try:
            return _to_response(
                runtime.plugin_toggle(name=name, enabled=req.enabled, chat_id=req.chat_id)
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post(
        "/config/rollback",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def config_rollback(req: PluginRollbackRequest) -> ChatResponse:
        try:
            return _to_response(runtime.plugin_rollback(req.steps))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get(
        "/plugin-incidents",
        response_model=PluginIncidentListResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def plugin_incidents() -> PluginIncidentListResponse:
        return PluginIncidentListResponse(incidents=runtime.incidents())

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
        return {"chat_id": chat_id, "memory": runtime.memory(chat_id)}

    @app.post("/summarize/{chat_id}", dependencies=[Depends(_require_api_key)])
    def summarize(chat_id: str) -> ChatResponse:
        return _to_response(runtime.summarize(chat_id))

    @app.post("/recall", dependencies=[Depends(_require_api_key)])
    def recall(req: RecallRequest) -> ChatResponse:
        return _to_response(
            runtime.recall(req.chat_id, req.query, tags=req.tags, max_tokens=req.max_tokens)
        )

    @app.post("/retain", dependencies=[Depends(_require_api_key)])
    def retain(req: RetainRequest) -> ChatResponse:
        return _to_response(
            runtime.retain(req.chat_id, req.content, tags=req.tags, context=req.context)
        )

    @app.post("/promote", dependencies=[Depends(_require_api_key)])
    def promote(req: ChatRequest) -> ChatResponse:
        return _to_response(runtime.promote(req.chat_id, req.message))

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
            plan = runtime.plan_create(
                req.name,
                description=req.description,
                chat_id=req.chat_id,
                tasks=tasks,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _plan_to_response(plan)

    @app.post(
        "/plan/task/start", response_model=TaskResponse, dependencies=[Depends(_require_api_key)]
    )
    def plan_task_start(req: PlanTaskStartRequest) -> TaskResponse:
        try:
            task = runtime.plan_task_start(req.plan_id, req.task_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _task_to_response(task)

    @app.post(
        "/plan/task/done", response_model=TaskResponse, dependencies=[Depends(_require_api_key)]
    )
    def plan_task_done(req: PlanTaskDoneRequest) -> TaskResponse:
        try:
            task = runtime.plan_task_done(
                req.plan_id,
                req.task_id,
                result=req.result,
                log=req.log,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _task_to_response(task)

    @app.get("/plan/list", response_model=list[PlanResponse])
    def plan_list() -> list[PlanResponse]:
        if hasattr(runtime, "plan_list"):
            plans = runtime.plan_list()
        else:
            plans = []
        return [_plan_to_response(p) for p in plans]

    @app.get("/plan/{plan_id}", response_model=PlanResponse)
    def plan_get(plan_id: str) -> PlanResponse:
        if not hasattr(runtime, "plan_get"):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Plan retrieval not supported",
            )
        plan = runtime.plan_get(plan_id)
        if plan is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plan {plan_id} not found",
            )
        return _plan_to_response(plan)

    @app.post(
        "/runtime/start",
        dependencies=[Depends(_require_api_key)],
    )
    def runtime_start() -> dict[str, bool]:
        runtime.start()
        return {"ok": True}

    @app.post(
        "/runtime/stop",
        dependencies=[Depends(_require_api_key)],
    )
    def runtime_stop() -> dict[str, bool]:
        runtime.shutdown()
        return {"ok": True}

    @app.get("/runtime/status", response_model=RuntimeStatusResponse)
    def runtime_status() -> RuntimeStatus:
        return runtime.get_status()

    @app.get("/config")
    def config_get() -> dict[str, Any]:
        """Return the current live runtime configuration (excluding secrets)."""
        return runtime.get_config()

    @app.patch(
        "/config",
        dependencies=[Depends(_require_api_key)],
    )
    def config_update(patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial live update to Telegram and/or plugin configuration."""
        try:
            return runtime.update_config(patch)
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
        return runtime.get_task_config()

    @app.post(
        "/task/config",
        response_model=TaskConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def task_config_update(req: TaskConfig) -> TaskConfig:
        try:
            runtime.update_task_config(req)
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return runtime.get_task_config()

    @app.get(
        "/waker/config",
        response_model=WakerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def waker_config_get() -> WakerConfig:
        return runtime.get_waker_config()

    @app.post(
        "/waker/config",
        response_model=WakerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def waker_config_update(req: WakerConfig) -> WakerConfig:
        try:
            runtime.update_waker_config(req)
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return runtime.get_waker_config()

    @app.get(
        "/timer/config",
        response_model=TimerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def timer_config_get() -> TimerConfig:
        return runtime.get_timer_config()

    @app.post(
        "/timer/config",
        response_model=TimerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def timer_config_update(req: TimerConfig) -> TimerConfig:
        try:
            runtime.update_timer_config(req)
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return runtime.get_timer_config()

    @app.get(
        "/notifications/config",
        response_model=NotificationsConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def notifications_config_get() -> NotificationsConfig:
        return runtime.get_notifications_config()

    @app.post(
        "/notifications/config",
        response_model=NotificationsConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def notifications_config_update(req: NotificationsConfig) -> NotificationsConfig:
        try:
            runtime.update_notifications_config(req)
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return runtime.get_notifications_config()

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
