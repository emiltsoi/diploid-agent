"""Tests for safe plugin loading and FailedPlugin fallback."""

from pathlib import Path

from devin_fleet_harness.config import PluginConfig
from devin_fleet_harness.plugins import PluginManager
from devin_fleet_harness.plugins.broken import FailedPlugin


def _manager(tmp_path: Path, plugins: list[PluginConfig]) -> PluginManager:
    return PluginManager(
        plugins=plugins,
        sessions_root=tmp_path,
        instance_id="test-instance",
        instance_started_at=0.0,
        runtime=None,
    )


def test_missing_module_loads_failed_plugin(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        [
            PluginConfig(
                name="missing",
                module="devin_fleet_harness.plugins.this_module_does_not_exist",
                enabled=True,
            ),
        ],
    )
    plugin = manager._get_or_create("chat-1", manager._plugins[0])
    assert isinstance(plugin, FailedPlugin)


def test_unsafe_module_name_loads_failed_plugin(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        [
            PluginConfig(
                name="unsafe",
                module="os.path",
                enabled=True,
            ),
        ],
    )
    plugin = manager._get_or_create("chat-1", manager._plugins[0])
    assert isinstance(plugin, FailedPlugin)


def test_built_in_module_loads_failed_plugin(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        [
            PluginConfig(
                name="builtin",
                module="sys",
                enabled=True,
            ),
        ],
    )
    plugin = manager._get_or_create("chat-1", manager._plugins[0])
    assert isinstance(plugin, FailedPlugin)


def test_plugin_without_plugin_class_loads_failed_plugin(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        [
            PluginConfig(
                name="no_plugin_class",
                module="devin_fleet_harness.config",
                enabled=True,
            ),
        ],
    )
    plugin = manager._get_or_create("chat-1", manager._plugins[0])
    assert isinstance(plugin, FailedPlugin)
