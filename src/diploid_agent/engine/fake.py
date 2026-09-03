"""Fake AgentEngine for tests and local development."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diploid_agent.config import EngineConfig
from diploid_agent.engine.base import AgentEngine, TurnRequest, TurnResult


@dataclass
class FakeAgentEngine(AgentEngine):
    """Deterministic, in-memory AgentEngine that records calls and returns
    pre-configured results.
    """

    config: EngineConfig | None = None
    api_key: str | None = None
    metrics: Any | None = None
    default_reply: str = "ok"
    default_session_id: str = "fake-session-1"
    models: list[str] = field(default_factory=lambda: ["swe-1-7"])
    alive: bool = True
    resume_fail: bool = False
    resumed: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    call_log: list[tuple[str, Any]] = field(default_factory=list)
    lifecycle_log: Any | None = None
    session_counter: int = field(default=0, init=False)

    def _next_session_id(self) -> str:
        self.session_counter += 1
        return f"{self.default_session_id}-{self.session_counter}"

    def prompt(
        self,
        request: TurnRequest,
        *,
        session_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        self.call_log.append(("prompt", request, session_id))
        if on_chunk:
            on_chunk(request.prompt[:20])
        reply = self.replies.pop(0) if self.replies else self.default_reply
        return TurnResult(
            reply=reply,
            session_id=session_id or self._next_session_id(),
            stop_reason=None,
        )

    def cancel(self, session_id: str) -> None:
        self.call_log.append(("cancel", session_id))

    def list_models(self) -> list[str]:
        self.call_log.append(("list_models", None))
        return list(self.models)

    def session_alive(self, session_id: str) -> bool:
        self.call_log.append(("session_alive", session_id))
        return self.alive

    def resume_session(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        self.call_log.append(("resume_session", session_id, cwd, model, mcp_servers))
        if self.resume_fail:
            raise RuntimeError(f"resume failed for {session_id}")
        self.resumed.append(session_id)
        return session_id

    def close(self) -> None:
        self.call_log.append(("close", None))

    def model_context_window(self, model: str) -> int | None:
        self.call_log.append(("model_context_window", model))
        if self.config is not None and self.config.context_window is not None:
            return self.config.context_window
        return None
