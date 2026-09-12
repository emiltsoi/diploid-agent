"""Shared base for the Runtime* collaborator classes.

Each runtime component is bound to the owning ``AgentRuntime`` and reaches
sibling components through it. ``RuntimeComponent`` holds the accessors that
were previously copied into every component. Single-use accessors stay on the
concrete classes; ``_store`` is deliberately excluded because
``ChatSessionStore`` uses that name for its own dict.
"""

from __future__ import annotations

from typing import Any


class RuntimeComponent:
    """Runtime accessors shared by the Runtime* components."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def config(self) -> Any:
        return self._runtime.config

    @property
    def _lock(self) -> Any:
        return self._runtime._lock

    @property
    def _plugins(self) -> Any:
        return self._runtime._plugins

    @property
    def _chat_store(self) -> Any:
        return self._runtime._chat_store

    @property
    def context_builder(self) -> Any:
        return self._runtime.context_builder

    @property
    def _mcp_skills(self) -> Any:
        return self._runtime._mcp_skills

    @property
    def engine(self) -> Any:
        return self._runtime.engine

    @property
    def _prompts(self) -> Any:
        return self._runtime._prompts

    @property
    def _runtime_metrics(self) -> Any:
        return self._runtime._runtime_metrics

    @property
    def _config_manager(self) -> Any:
        return self._runtime._config_manager

    @property
    def _runtime_plugins(self) -> Any:
        return self._runtime._runtime_plugins

    @property
    def _incidents(self) -> Any:
        return self._runtime._incidents

    @property
    def wake_queue(self) -> Any:
        return self._runtime.wake_queue

    @property
    def instance_started_at(self) -> Any:
        return self._runtime.instance_started_at

    @property
    def plan_manager(self) -> Any:
        return self._runtime.plan_manager

    @property
    def task_engine(self) -> Any:
        return self._runtime.task_engine

    @property
    def store_path(self) -> Any:
        return self._runtime.store_path

    @property
    def _active_record(self) -> Any:
        return self._runtime._active_record

    @property
    def notifier(self) -> Any:
        return self._runtime.notifier

    @property
    def skills(self) -> Any:
        return self._runtime.skills
