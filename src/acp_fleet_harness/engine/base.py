"""Base AgentEngine protocol for acp-fleet-harness."""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TurnRequest:
    """A request to run one prompt turn against an agent engine."""

    prompt: str
    cwd: Path
    model: str | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    soft_timeout: float | None = None


@dataclass
class TurnResult:
    """Result of one prompt turn. Mirrors AcpPromptResult."""

    reply: str
    session_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    cancelled: bool = False
    partial: bool = False
    timed_out: bool = False
    updates: list[dict[str, Any]] = field(default_factory=list)


class AgentEngine(abc.ABC):
    """Abstract protocol for an agent backend.

    The canonical implementation is `DevinAcpEngine`, which speaks ACP over
    stdio. Other implementations can drop in by satisfying this interface.
    """

    @abc.abstractmethod
    def prompt(
        self,
        request: TurnRequest,
        *,
        session_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        """Send a prompt and return the result.

        If `session_id` is None, the engine should create a new session.
        If `session_id` is provided, the engine should continue the session.
        """

    @abc.abstractmethod
    def cancel(self, session_id: str) -> None:
        """Cancel an in-flight turn."""

    @abc.abstractmethod
    def list_models(self) -> list[str]:
        """Return the list of models the engine supports."""

    @abc.abstractmethod
    def session_alive(self, session_id: str) -> bool:
        """Return True if the session can still be used."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the engine and release resources."""

    def restart(self) -> None:
        """Restart the underlying transport, if supported."""
        return

    def is_stale_session_error(self, exc: BaseException) -> bool:
        """Return True if the exception indicates a stale session."""
        return False

    def health(self) -> bool:
        """Return True if the engine is healthy enough to accept prompts."""
        return True
