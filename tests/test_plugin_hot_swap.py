"""Tests for plugin hot-swap operations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    MemoryConfig,
    PersonaConfig,
    PlanConfig,
    PluginConfig,
    Secrets,
    TimerConfig,
)
from diploid_agent.engine import TurnResult
from diploid_agent.runtime.agent_runtime import AgentRuntime


class FakeEngine:
    """Stand-in engine that avoids real Devin calls during tests."""

    def prompt(self, *a, **k):
        return TurnResult(reply="ok", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def _make_runtime(tmp_path: Path) -> AgentRuntime:
    profile_root = tmp_path / "persona"
    profile_root.mkdir(parents=True, exist_ok=True)
    config = Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(name="test", profile_root=profile_root),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            session_prune_enabled=False,
            plan=PlanConfig(root=tmp_path / "plans"),
            memory=MemoryConfig(backend="file"),
            timer=TimerConfig(enabled=False),
            plugins=[
                PluginConfig(
                    name="continuity",
                    enabled=True,
                    module="diploid_agent.plugins.continuity",
                ),
            ],
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )
    runtime = AgentRuntime(config)
    runtime.engine = FakeEngine()
    return runtime


def test_add_plugin(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    cfg = PluginConfig(
        name="self_state",
        enabled=True,
        module="diploid_agent.plugins.self_state",
        prompt_slot="persona_state",
        state_file="chat_self_state.md",
    )
    result = runtime.plugin_add(cfg)
    assert "added" in result.reply
    names = {p.name for p in runtime.config.harness.plugins}
    assert "self_state" in names


def test_remove_plugin(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    result = runtime.plugin_remove("continuity")
    assert "removed" in result.reply
    names = {p.name for p in runtime.config.harness.plugins}
    assert "continuity" not in names


def test_toggle_plugin(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    result = runtime.plugin_toggle("continuity", enabled=False)
    assert "disabled" in result.reply
    cfg = next(p for p in runtime.config.harness.plugins if p.name == "continuity")
    assert cfg.enabled is False


def test_rollback(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.plugin_add(
        PluginConfig(
            name="planner",
            enabled=True,
            module="diploid_agent.plugins.planner",
            prompt_slot="persona_state",
        )
    )
    runtime.plugin_add(
        PluginConfig(
            name="self_state",
            enabled=True,
            module="diploid_agent.plugins.self_state",
            prompt_slot="persona_state",
        )
    )
    runtime.plugin_rollback(1)
    names = {p.name for p in runtime.config.harness.plugins}
    assert "self_state" not in names
    assert "planner" in names
    runtime.plugin_rollback(1)
    names = {p.name for p in runtime.config.harness.plugins}
    assert "planner" not in names
    assert "continuity" in names


def test_persisted_to_runtime_overrides(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.plugin_add(
        PluginConfig(
            name="self_state",
            enabled=True,
            module="diploid_agent.plugins.self_state",
            prompt_slot="persona_state",
        )
    )
    overrides_path = runtime._runtime_overrides_path
    assert overrides_path.exists()
    data = yaml.safe_load(overrides_path.read_text())
    names = {p["name"] for p in data["plugins"]}
    assert "self_state" in names


def test_stop_all_calls_stop(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    fake = MagicMock()
    runtime._plugins._load_plugin = lambda _config, _chat_id: fake

    runtime._plugins._get_or_create("chat-1", PluginConfig(name="tracker", enabled=True))
    assert fake.start.called, "start() should be called when the instance is created"

    runtime._plugins.stop_all()
    assert fake.stop.called, "stop() should be called by stop_all()"


def test_reconfigure_stops_removed_plugin(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    fake = MagicMock()
    runtime._plugins._instances["chat-1"]["tracker"] = fake

    before_snapshots = len(runtime._plugins._config_history)
    runtime._plugins.reconfigure(
        [PluginConfig(name="continuity", enabled=True, module="diploid_agent.plugins.continuity")]
    )

    assert fake.stop.called, "reconfigure() should stop a plugin no longer in the new list"
    assert "chat-1" not in runtime._plugins._instances
    assert len(runtime._plugins._config_history) > before_snapshots


def test_add_plugin_rejects_empty_name_and_duplicates(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    with pytest.raises(ValueError, match="Plugin must have a name"):
        runtime.plugin_add(PluginConfig(name="", enabled=True))

    with pytest.raises(ValueError, match="already exists"):
        runtime.plugin_add(
            PluginConfig(
                name="continuity",
                enabled=True,
                module="diploid_agent.plugins.continuity",
            )
        )


def test_toggle_unknown_plugin_raises(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    with pytest.raises(ValueError, match="Unknown plugin"):
        runtime.plugin_toggle("unknown", enabled=False)


def test_rollback_bounds(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        runtime.plugin_rollback(0)
    with pytest.raises(ValueError, match="No earlier configuration"):
        runtime.plugin_rollback(100)


def test_persisted_plugins_loaded_on_new_runtime(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.plugin_add(
        PluginConfig(
            name="self_state",
            enabled=True,
            module="diploid_agent.plugins.self_state",
            prompt_slot="persona_state",
        )
    )

    fresh = _make_runtime(tmp_path)
    names = {p.name for p in fresh.config.harness.plugins}
    assert "self_state" in names
    assert "continuity" in names
