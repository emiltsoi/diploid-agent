"""Engine factory."""

from __future__ import annotations

from typing import Any

from diploid_agent.config import EngineConfig
from diploid_agent.engine.acp import AcpEngine
from diploid_agent.engine.base import AgentEngine
from diploid_agent.engine.fake import FakeAgentEngine

ENGINES: dict[str, type[AgentEngine]] = {
    "diploid": AcpEngine,
    "generic": AcpEngine,
    "fake": FakeAgentEngine,
}


def build_engine(
    config: EngineConfig, *, api_key: str | None = None, metrics: Any | None = None
) -> AgentEngine:
    """Build an AgentEngine from the configured provider."""
    engine_cls = ENGINES.get(config.provider)
    if engine_cls is None:
        raise ValueError(f"unknown engine provider: {config.provider}")
    if engine_cls is FakeAgentEngine:
        return FakeAgentEngine()
    return engine_cls(config, api_key=api_key, metrics=metrics)
