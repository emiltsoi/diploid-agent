"""Tests for plugin scaffolding and creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from diploid_agent.config import Config, DiploidConfig, HarnessConfig, PersonaConfig, PluginConfig
from diploid_agent.runtime.agent_runtime import AgentRuntime


class FakeEngine:
    def prompt(self, *a, **k):
        from diploid_agent.engine import TurnResult

        return TurnResult(reply="ok", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def _make_config(tmp_path: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(name="test", profile_root=tmp_path / "persona"),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            session_prune_enabled=False,
            plugin_paths=[tmp_path / "plugins"],
            plugins=[
                PluginConfig(
                    name="continuity", enabled=True, module="diploid_plugins.continuity"
                )
            ],
        ),
    )


def test_plugin_create_scaffolds_and_sandboxes(tmp_path: Path):
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()

    result = runtime.plugin_create(name="hello_world", prompt_slot="persona_state")

    assert result.get("name") == "hello_world"
    assert result.get("module") == "hello_world"
    plugin_dir = tmp_path / "plugins" / "hello_world"
    assert plugin_dir.exists()
    assert (plugin_dir / "__init__.py").exists()
    text = (plugin_dir / "__init__.py").read_text()
    assert "class Plugin(StatePlugin):" in text


def test_plugin_create_rejects_existing(tmp_path: Path):
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()

    runtime.plugin_create(name="duplicate")
    with pytest.raises(ValueError, match="already exists"):
        runtime.plugin_create(name="duplicate")


def test_plugin_create_rejects_unsafe_name(tmp_path: Path):
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()

    with pytest.raises(ValueError, match="Unsafe"):
        runtime.plugin_create(name="../bad")
