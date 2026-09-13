"""RuntimeActions: public runtime actions extracted from AgentRuntime."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from diploid_agent.models import ChatResult, WakeEvent
from diploid_agent.plan.models import Plan, Task, TaskStatus
from diploid_agent.plugins.contexts import PromoteContext, RetainContext
from diploid_agent.runtime.event_bus import Event
from diploid_agent.runtime.state import RuntimeState

if TYPE_CHECKING:
    import threading

    from diploid_agent.acp_client import AcpLifecycleLog
    from diploid_agent.config import Config
    from diploid_agent.engine import AgentEngine
    from diploid_agent.memory import MemoryManager
    from diploid_agent.plan.manager import PlanManager
    from diploid_agent.plugin_incidents import PluginIncidentStore
    from diploid_agent.plugins import PluginManager
    from diploid_agent.runtime.event_bus import EventBus
    from diploid_agent.runtime.metrics import RuntimeMetrics
    from diploid_agent.runtime.outbox import RuntimeOutbox
    from diploid_agent.runtime.prompts import RuntimePrompts
    from diploid_agent.runtime.restart import RuntimeRestart
    from diploid_agent.runtime.store import ChatSessionStore
    from diploid_agent.runtime.subagent import RuntimeSubagent
    from diploid_agent.runtime.wake_queue import WakeQueue
    from diploid_agent.task.engine import TaskEngine
    from diploid_agent.turn.controller import TurnController


logger = logging.getLogger(__name__)


def _actions_locked(method: Any) -> Any:
    """Run a RuntimeActions method under the runtime's RLock."""

    @functools.wraps(method)
    def wrapper(self: RuntimeActions, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class RuntimeActions:
    """Public, non-turn runtime actions backed by an AgentRuntime."""

    def __init__(
        self,
        *,
        state: RuntimeState,
        config: Config,
        lock: threading.RLock,
        chat_store: ChatSessionStore,
        lifecycle_log: AcpLifecycleLog,
        memory_manager: Callable[[str], MemoryManager],
        runtime_metrics: RuntimeMetrics,
        prompts: RuntimePrompts,
        outbox: RuntimeOutbox,
        plugins: PluginManager,
        subagent: RuntimeSubagent,
        restart: RuntimeRestart,
        incidents: PluginIncidentStore,
        wake_queue: WakeQueue,
        plan_manager: PlanManager,
        task_engine: TaskEngine,
        event_bus: EventBus,
        turn_controller: TurnController,
        instance_id: str,
        engine_fn: Callable[[], AgentEngine],
        call_unlocked_fn: Callable[..., Any],
        suppress_auto_continue_fn: Callable[..., None],
        acp_client_fn: Callable[[], Any],
    ) -> None:
        self._state = state
        self.config = config
        self._lock = lock
        self._chat_store = chat_store
        self._lifecycle_log = lifecycle_log
        self._memory_manager = memory_manager
        self._runtime_metrics = runtime_metrics
        self._prompts = prompts
        self._outbox = outbox
        self._plugins = plugins
        self._subagent = subagent
        self._restart = restart
        self._incidents = incidents
        self.wake_queue = wake_queue
        self.plan_manager = plan_manager
        self.task_engine = task_engine
        self.event_bus = event_bus
        self.turn_controller = turn_controller
        self.instance_id = instance_id
        self._engine_fn = engine_fn
        self._call_unlocked = call_unlocked_fn
        self._suppress_auto_continue = suppress_auto_continue_fn
        self._acp_client_fn = acp_client_fn

    @property
    def engine(self) -> Any:
        return self._engine_fn()

    @property
    def acp_client(self) -> Any:
        return self._acp_client_fn()

    def _continuity_status(self, chat_id: str, record: Any) -> dict[str, Any]:
        """Return ACP continuity status for the active session."""
        continuity: dict[str, Any] = {
            "resume_enabled": self.config.engine.acp_resume_enabled,
            "current_session_id": record.session_id,
            "state": "unknown",
            "state_reason": None,
            "last_restart_at": None,
            "last_restart_reason": None,
            "restart_count_in_window": 0,
            "resume_metrics": {},
        }
        if self._lifecycle_log is None:
            return continuity

        events = self._lifecycle_log.recent_events_for(
            chat_id,
            limit=500,
        )
        if not events:
            return continuity

        window = self.config.engine.acp_restart_backoff_window
        now = time.time()
        restart_events = [
            e
            for e in events
            if e.get("event") == "transport.restart"
            and now - datetime.fromisoformat(e["timestamp"]).timestamp() < window
        ]
        if restart_events:
            last = restart_events[-1]
            continuity["last_restart_at"] = last["timestamp"]
            continuity["last_restart_reason"] = last.get("reason")
            continuity["restart_count_in_window"] = len(restart_events)

        # Determine whether the current session was resumed, rebuilt, or is new.
        session_id = record.session_id
        for event in reversed(events):
            if event.get("session_id") != session_id:
                # A transport restart for this chat usually means the session was rebuilt.
                if event.get("event") == "transport.restart" and event.get("chat_id") == chat_id:
                    continuity["state"] = "rebuilt"
                    continuity["state_reason"] = event.get("reason")
                    break
                continue
            ev = event.get("event")
            if ev in ("rehydrate.resume.success", "session.resume.success", "session.load.success"):
                continuity["state"] = "resumed"
                continuity["state_reason"] = event.get("reason")
                break
            if ev == "rehydrate.new_session.success":
                continuity["state"] = "rebuilt"
                continuity["state_reason"] = event.get("reason")
                break
            if ev in ("session.new.success", "session.new"):
                continuity["state"] = "new"
                continuity["state_reason"] = event.get("reason")
                break

        # Resume telemetry for this chat: count successes/failures by method.
        resume_events = [
            e
            for e in events
            if e.get("event")
            in (
                "session.resume.success",
                "session.resume.failure",
                "session.load.success",
                "session.load.failure",
                "session.new.success",
                "session.new.failure",
                "rehydrate.resume.success",
                "rehydrate.resume.failure",
                "rehydrate.new_session.success",
            )
        ]
        durations: list[float] = []
        resume_counts: dict[str, int] = {}
        for event in resume_events:
            ev = event.get("event", "")
            result = "success" if ".success" in ev else "failure"
            method = ev.split(".")[-2] if "." in ev else "unknown"
            key = f"{method}_{result}"
            resume_counts[key] = resume_counts.get(key, 0) + 1
            detail = event.get("detail") or {}
            if isinstance(detail, dict) and "duration_ms" in detail:
                durations.append(float(detail["duration_ms"]))

        continuity["resume_metrics"] = {
            "counts": resume_counts,
            "total_events": len(resume_events),
            "average_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
        }

        # Last wake-relevant event for the chat.
        continuity["last_wake_event"] = self._lifecycle_log.last_wake_event_for(chat_id)

        return continuity

    def status(self, chat_id: str) -> dict[str, Any]:
        """Return the harness-recorded status for a chat."""
        with self._lock:
            record = self._chat_store._active_record(chat_id)
            if not record:
                return {"chat_id": chat_id, "active": False}

        memory_stats: dict[str, Any] = {}
        try:
            memory_stats = self._memory_manager(chat_id).stats()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load memory stats for %s: %s", chat_id, exc)

        context_usage = self._runtime_metrics._context_usage(record)
        background_tasks = self._subagent.subagent_status(chat_id)

        active_turn = self.turn_controller.turn_status(chat_id, wait=0.0)

        with self._lock:
            record = self._chat_store._active_record(chat_id)
            if not record:
                return {"chat_id": chat_id, "active": False}

            if context_usage and record.last_stop_reason:
                context_usage["last_turn"]["stop_reason"] = record.last_stop_reason

            return {
                "chat_id": chat_id,
                "active": True,
                "persona": record.persona,
                "model": record.model,
                "session_number": record.session_number,
                "session_id": record.session_id,
                "cwd": record.cwd,
                "turn_number": record.turn_number,
                "persona_memory_exceeded": record.persona_memory_exceeded,
                "chat_memory_exceeded": record.chat_memory_exceeded,
                "memory": memory_stats,
                "last_turn_metrics": record.last_turn_metrics,
                "cumulative_metrics": record.cumulative_metrics,
                "context_usage": context_usage,
                "enabled_mcp_servers": record.enabled_mcp_servers,
                "enabled_skills": record.enabled_skills,
                "disabled_skills": record.disabled_skills,
                "background_tasks": background_tasks,
                "active_turn": active_turn,
                "continuity": self._continuity_status(chat_id, record),
            }

    def list_sessions(self, chat_id: str) -> dict[str, Any]:
        """Return all non-pruned sessions for a chat, with the active one marked."""
        with self._lock:
            state = self._chat_store._chat_state(chat_id)
            active = self._chat_store._active_record(chat_id)
            sessions = []
            for record in sorted(state.sessions.values(), key=lambda r: r.session_number):
                sessions.append(
                    {
                        "number": record.session_number,
                        "label": record.label,
                        "model": record.model,
                        "turn_number": record.turn_number,
                        "updated_at": record.updated_at,
                        "parent": record.parent,
                        "is_active": active is not None
                        and record.session_number == active.session_number,
                    }
                )
            return {
                "chat_id": chat_id,
                "active": active.session_number if active else None,
                "sessions": sessions,
            }

    @_actions_locked
    def memory(self, chat_id: str) -> str:
        """Return the per-chat memory content."""
        return self._memory_manager(chat_id).memory_content()

    @_actions_locked
    def summarize(self, chat_id: str) -> ChatResult:
        """Trigger a manual summarization for a chat."""
        record = self._chat_store._active_record(chat_id)
        model = self._prompts._model(record)
        mgr = self._memory_manager(chat_id)
        self._call_unlocked(mgr._summarize, model)

        notice = None
        if record:
            notice = self._prompts._check_chat_memory_transition(chat_id, record)
            self._chat_store._append_record(record)

        return ChatResult(reply="Summarization complete.", notice=notice)

    @_actions_locked
    def recall(
        self,
        chat_id: str,
        query: str,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Recall relevant memories for a query."""
        max_tokens = max_tokens or self.config.harness.memory.hindsight.max_recall_tokens
        reply = self._memory_manager(chat_id).backend.recall(
            query,
            tags=tags,
            max_tokens=max_tokens,
        )
        return ChatResult(reply=reply or "No relevant memory found.")

    @_actions_locked
    def retain(
        self,
        chat_id: str,
        content: str,
        tags: list[str] | None = None,
        context: str | None = None,
    ) -> ChatResult:
        """Retain an observation in the chat's memory."""
        ctx = self._plugins.before_retain(
            chat_id,
            RetainContext(
                chat_id=chat_id,
                content=content,
                tags=tags or [],
                context=context,
            ),
        )
        self._memory_manager(chat_id).retain(ctx.content, tags=ctx.tags, context=ctx.context)
        self._plugins.after_retain(chat_id, ctx)
        return ChatResult(reply="Retained.")

    @_actions_locked
    def promote(self, chat_id: str, fact: str) -> ChatResult:
        """Promote a fact to the chat's curated memory pocket."""
        record = self._chat_store._active_record(chat_id)
        ctx = self._plugins.before_promote(
            chat_id,
            PromoteContext(chat_id=chat_id, fact=fact, record=record),
        )
        self._memory_manager(chat_id).promote(ctx.fact)

        record = self._chat_store._active_record(chat_id)
        if record:
            self._chat_store._append_record(record)
            self._plugins.after_promote(
                chat_id,
                PromoteContext(chat_id=chat_id, fact=ctx.fact, record=record),
            )

        return ChatResult(reply="Promoted.")

    def list_models(self) -> list[str]:
        """Return the list of models the ACP server accepts."""
        return self.engine.list_models()

    def record_mesh_message(
        self,
        chat_id: str,
        display_text: str,
        mesh_payload: dict[str, Any],
    ) -> ChatResult:
        """Persist a terminal mesh message (e.g. a DSN) without running a turn."""
        record = self._chat_store._active_record(chat_id)

        event = WakeEvent(
            id=f"mesh:{mesh_payload.get('message_id', 'unknown')}",
            chat_id=chat_id,
            reason="mesh_dsn",
            priority=1,
            scheduled_at=time.time(),
            payload={"user_message": display_text, "mesh_meta": mesh_payload},
            silent=True,
            ready=False,
        )
        self._plugins.on_waking(chat_id, record, time.time(), wake_event=event)

        self._memory_manager(chat_id).append_mesh_note(display_text)

        return ChatResult(reply="", notice=None)

    @_actions_locked
    def graceful_service_restart(
        self,
        chat_id: str,
        service: str | None = None,
        reason: str = "",
    ) -> ChatResult:
        """Public API for a graceful service restart (HTTP/Telegram/MCP)."""
        if service is None:
            service = f"{self.config.persona.name}.service"
        now = time.time()
        if now - self._state.last_service_restart_at < self._state.service_restart_cooldown_seconds:
            return ChatResult(
                reply=f"A restart for {service} is already scheduled.",
                notice="Please wait for it to complete.",
            )
        self._state.last_service_restart_at = now

        # Suppress auto-continue for this chat and cancel pending wakes.
        self._suppress_auto_continue(chat_id=chat_id, seconds=300)
        if self.wake_queue is not None:
            try:
                self.wake_queue.cancel(chat_id=chat_id, reason="auto_continue")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to cancel auto-continue wakes for %s: %s", chat_id, exc)

        if self._incidents is not None:
            try:
                self._incidents.record(
                    plugin="self_management",
                    phase="graceful_restart",
                    error=f"User requested restart of {service}: {reason}",
                    action="scheduled",
                    chat_id=chat_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to record restart incident for %s: %s", chat_id, exc)

        if not self._restart._schedule_draining_restart(service, chat_id=chat_id, reason=reason):
            self._state.last_service_restart_at = 0.0
            return ChatResult(
                reply=f"Could not restart {service}: no such systemd user unit.",
                notice="Check the service name and try again.",
            )

        return ChatResult(
            reply=(
                f"Restarting {service} after the current turn finishes "
                "(or in ≤2 min, whichever comes first). I'll be back in a moment."
            ),
            notice="The service drains in-flight work before restarting.",
        )

    @_actions_locked
    def switch_model(self, chat_id: str, model: str, *, in_place: bool = False) -> ChatResult:
        """Switch the model for a chat."""
        return self.turn_controller.switch_model(chat_id, model, in_place=in_place)

    def new_session(self, chat_id: str, model: str | None = None) -> ChatResult:
        """Start a fresh ACP session for a chat."""
        return self.turn_controller.new_session(chat_id, model=model)

    def resume_session(self, chat_id: str, session_number: int) -> ChatResult:
        """Resume a previous session for a chat."""
        return self.turn_controller.resume_session(chat_id, session_number)

    def branch_session(self, chat_id: str, session_number: int) -> ChatResult:
        """Branch a previous session for a chat."""
        return self.turn_controller.branch_session(chat_id, session_number)

    def plan_create(
        self,
        name: str,
        description: str = "",
        chat_id: str | None = None,
        tasks: list[Task] | None = None,
    ) -> Plan:
        """Create a new plan."""
        return self.plan_manager.create_plan(
            name, description=description, chat_id=chat_id, tasks=tasks or []
        )

    def plan_task_start(self, plan_id: str, task_id: str | None = None) -> Task:
        """Start a ready task in a plan."""
        return self.task_engine.start_task(plan_id, task_id)

    @_actions_locked
    def plan_task_done(
        self,
        plan_id: str,
        task_id: str,
        result: str = "",
        log: str = "",
    ) -> Task:
        """Manually mark a task as done and emit the completion event."""
        existing = self.plan_manager.get_task(plan_id, task_id)
        already_done = existing is not None and existing.status == TaskStatus.DONE
        task = self.plan_manager.complete_task(plan_id, task_id, result=result, log=log)
        if task is None:
            raise ValueError(f"Task {task_id} not found in plan {plan_id}")
        if not already_done:
            self.event_bus.post(
                Event(
                    type="task.completed",
                    payload={
                        "plan_id": plan_id,
                        "task_id": task_id,
                        "result": result,
                        "log": log,
                    },
                )
            )
        return task
