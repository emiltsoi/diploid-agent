"""A fake PluginRuntime for safely exercising plugins outside the live harness."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.config import Config, EngineConfig, HarnessConfig, PersonaConfig
from diploid_agent.engine.base import AgentEngine, TurnRequest, TurnResult
from diploid_agent.models import ChatResult, SessionRecord
from diploid_agent.plan.models import Plan, Task
from diploid_agent.runtime.wake_queue import WakeQueue


class FakeAgentEngine(AgentEngine):
    """AgentEngine that never leaves the sandbox."""

    def prompt(
        self,
        request: TurnRequest,
        *,
        session_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        return TurnResult(reply="ok", session_id=session_id or "sandbox")

    def cancel(self, session_id: str) -> None:
        return

    def list_models(self) -> list[str]:
        return ["swe-1-7"]

    def session_alive(self, session_id: str) -> bool:
        return True

    def resume_session(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        return session_id

    def close(self) -> None:
        return


class FakeWakeQueue:
    """In-memory wake queue that never writes to disk."""

    def __init__(self) -> None:
        self._events: dict[str, Any] = {}

    def due_count(self, now: float | None = None) -> int:
        return 0

    def pending_count(self) -> int:
        return 0

    def pending(
        self,
        chat_id: str | None = None,
        now: float | None = None,
    ) -> list[Any]:
        return []

    def enqueue(self, event: Any) -> Any:
        event.id = event.id or f"wake-sandbox-{id(event)}"
        self._events[event.id] = event
        return event

    def complete(self, event_id: str) -> Any | None:
        return self._events.pop(event_id, None)

    def ready(self, event_id: str, now: float | None = None) -> Any | None:
        return self._events.get(event_id)

    def pop_due(self, now: float | None = None, lease_seconds: float = 300.0) -> list[Any]:
        return []


class FakePluginRuntime:
    """Minimal, safe runtime surface for the plugin sandbox.

    Implements the same shape as diploid_agent.runtime.plugin_runtime.PluginRuntime
    so candidate plugins can call engine, wake_queue, plan_create, recall, etc. without
    touching the live harness.
    """

    def __init__(self, sessions_root: Path | None = None, chat_id: str = "sandbox") -> None:
        self._sessions_root = sessions_root or Path(tempfile.mkdtemp())
        self._chat_id = chat_id

    @property
    def config(self) -> Config:
        return Config(
            diploid=EngineConfig(),
            persona=PersonaConfig(name="sandbox", profile_root=Path("/tmp")),
            harness=HarnessConfig(
                sessions_root=self._sessions_root,
                session_store_path=self._sessions_root / "sessions.jsonl",
                session_prune_enabled=False,
            ),
        )

    @property
    def instance_id(self) -> str:
        return "harness-sandbox"

    @property
    def instance_started_at(self) -> float:
        return 0.0

    @property
    def sessions_root(self) -> Path:
        return self._sessions_root

    @property
    def engine(self) -> AgentEngine:
        return FakeAgentEngine()

    @property
    def wake_queue(self) -> WakeQueue:
        return FakeWakeQueue()  # type: ignore[return-value]

    def plan_create(
        self,
        name: str,
        description: str = "",
        chat_id: str | None = None,
        tasks: list[Task] | None = None,
    ) -> Plan:
        return Plan(
            name=name,
            description=description,
            chat_id=chat_id or self._chat_id,
            tasks=tasks or [],
        )

    def plan_task_start(self, plan_id: str, task_id: str | None = None) -> Task:
        return Task(name="sandbox", plan_id=plan_id, chat_id=self._chat_id)

    def call_engine_unlocked(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    def is_continuation_message(self, text: str) -> bool:
        return text.lower() in {"continue", "go on", "proceed", "resume"}

    def recall(
        self,
        chat_id: str,
        query: str,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        return ChatResult(reply="")

    def promote(self, chat_id: str, fact: str) -> ChatResult:
        return ChatResult(reply="promoted")

    def suppress_auto_continue(self, chat_id: str, seconds: float = 300.0) -> None:
        return None

    def is_auto_continue_suppressed(self, chat_id: str) -> bool:
        return False

    def subagent_start(
        self,
        chat_id: str,
        prompt: str,
        *,
        context: str | None = None,
        model: str | None = None,
        cwd: Path | None = None,
        acp_timeout: float | None = None,
    ) -> ChatResult:
        return ChatResult(reply="subagent started", dispatch_id="dispatch-test")

    def subagent_status(self, chat_id: str) -> dict[str, Any]:
        return {"chat_id": chat_id, "subagents": []}

    def _active_record(self, chat_id: str) -> SessionRecord | None:
        return None

    def _append_record(self, record: SessionRecord) -> None:
        return
