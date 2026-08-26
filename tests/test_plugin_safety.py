# tests/test_plugin_safety.py

from pathlib import Path

import pytest

from acp_fleet_harness.config import Config, DevinConfig, HarnessConfig, PersonaConfig, PluginConfig
from acp_fleet_harness.runtime.agent_runtime import AgentRuntime


def test_startup_disables_broken_plugin(tmp_path: Path) -> None:
    config = Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(name="test", profile_root=tmp_path / "persona"),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            session_prune_enabled=False,
            plugins=[
                PluginConfig(name="broken", enabled=True, module="definitely.not.a.module"),
            ],
        ),
    )
    runtime = AgentRuntime(config)
    cfg = next(p for p in runtime.config.harness.plugins if p.name == "broken")
    assert cfg.enabled is False
