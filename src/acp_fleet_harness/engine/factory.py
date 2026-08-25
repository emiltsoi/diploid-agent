"""Engine factory."""

from __future__ import annotations

from typing import Any

from acp_fleet_harness.config import EngineConfig
from acp_fleet_harness.engine.base import AgentEngine
from acp_fleet_harness.engine.devin_acp import AcpEngine
from acp_fleet_harness.engine.fake import FakeAgentEngine

ENGINES: dict[str, type[AgentEngine]] = {
    "devin": AcpEngine,
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
    return engine_cls(config, api_key=api_key, metrics=metrics)
