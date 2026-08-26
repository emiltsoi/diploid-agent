"""Stable runtime surface exposed to state plugins."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from diploid_agent.config import Config
from diploid_agent.engine.base import AgentEngine
from diploid_agent.models import ChatResult
from diploid_agent.plan.models import Plan, Task
from diploid_agent.runtime.wake_queue import WakeQueue


@runtime_checkable
class PluginRuntime(Protocol):
    """The narrow runtime surface a state plugin may touch.

    This is intentionally smaller than ``AgentRuntime``. New plugin-facing
    helpers should be added here instead of letting plugins reach into
    ``AgentRuntime`` internals.
    """

    @property
    def config(self) -> Config: ...

    @property
    def instance_id(self) -> str: ...

    @property
    def instance_started_at(self) -> float: ...

    @property
    def sessions_root(self) -> Path: ...

    @property
    def engine(self) -> AgentEngine: ...

    @property
    def wake_queue(self) -> WakeQueue: ...

    def plan_create(
        self,
        name: str,
        description: str = "",
        chat_id: str | None = None,
        tasks: list[Task] | None = None,
    ) -> Plan: ...

    def plan_task_start(self, plan_id: str, task_id: str | None = None) -> Task: ...

    def call_engine_unlocked(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...

    def is_continuation_message(self, text: str) -> bool: ...

    def recall(
        self,
        chat_id: str,
        query: str,
        tags: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult: ...

    def promote(self, chat_id: str, fact: str) -> ChatResult: ...
