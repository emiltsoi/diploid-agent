"""Tests for plugin hot reload."""

from pathlib import Path

from diploid_agent.config import PluginConfig
from diploid_agent.plugins.manager import PluginManager


def test_reload_plugin_clears_instance() -> None:
    cfg = PluginConfig(name="json", enabled=True)
    manager = PluginManager(
        plugins=[cfg],
        sessions_root=Path("/tmp"),
        instance_id="test",
        instance_started_at=0.0,
    )
    plugins = manager._plugins_for("1")
    assert len(plugins) == 1
    assert manager._instances["1"]["json"] is not None

    result = manager.reload_plugin("1", "json")
    assert "reloaded" in result
    assert "json" not in manager._instances["1"]


def test_reload_unknown_plugin() -> None:
    manager = PluginManager(
        plugins=[],
        sessions_root=Path("/tmp"),
        instance_id="test",
        instance_started_at=0.0,
    )
    assert "Unknown plugin" in manager.reload_plugin("1", "missing")
