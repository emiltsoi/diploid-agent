"""Conversational harness: thin compatibility wrapper around AgentRuntime."""

from __future__ import annotations

from typing import Any

from diploid_agent.config import Config
from diploid_agent.engine import AgentEngine
from diploid_agent.runtime import AgentRuntime, TurnController


class ConversationHarness:
    """Backward-compatible wrapper that delegates to an AgentRuntime."""

    def __init__(self, config: Config) -> None:
        self.runtime = AgentRuntime(config)
        self.turn_controller: TurnController = self.runtime.turn_controller

    @property
    def client(self) -> AgentEngine:
        """Backward-compatible alias for the engine."""
        return self.runtime.client

    @client.setter
    def client(self, value: AgentEngine) -> None:
        self.runtime.client = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)
