"""Tests for plugin hot reload."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diploid_agent.config import PluginConfig
from diploid_agent.plugins.manager import PluginManager


def _manager(*plugins: PluginConfig) -> PluginManager:
    return PluginManager(
        plugins=list(plugins),
        sessions_root=Path("/tmp"),
        instance_id="test",
        instance_started_at=0.0,
    )


def test_reload_plugin_clears_instance() -> None:
    cfg = PluginConfig(name="json", enabled=True)
    manager = _manager(cfg)
    plugins = manager._plugins_for("1")
    assert len(plugins) == 1
    assert manager._instances["1"]["json"] is not None

    result = manager.reload_plugin("1", "json")
    assert "reloaded" in result
    assert "json" not in manager._instances["1"]


def test_reload_unknown_plugin() -> None:
    manager = _manager()
    assert "Unknown plugin" in manager.reload_plugin("1", "missing")


_IMPL_TEMPLATE = """\
class Impl:
    MARKER = "{marker}"

    def __init__(self, config, chat_id, sessions_root, runtime=None):
        self.config = config
        self.chat_id = chat_id
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True
"""


def _write_pkg(root: Path, name: str, marker: str) -> None:
    pkg = root / name
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text(
        f"from {name}.impl import Impl\n\nPlugin = Impl\n", encoding="utf-8"
    )
    (pkg / "impl.py").write_text(_IMPL_TEMPLATE.format(marker=marker), encoding="utf-8")
    # Drop stale bytecode: same-tick mtimes can otherwise serve an old .pyc.
    shutil.rmtree(pkg / "__pycache__", ignore_errors=True)
    importlib.invalidate_caches()


def _drop_pkg_from_sys_modules(name: str) -> None:
    for mod_name in [n for n in sys.modules if n == name or n.startswith(name + ".")]:
        sys.modules.pop(mod_name, None)


def test_reload_picks_up_submodule_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reloading a package must rebind code living in a submodule, not just __init__."""
    name = "hotswap_pkg_sub"
    _write_pkg(tmp_path, name, "version-one")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        manager = _manager(PluginConfig(name="p", enabled=True, module=name))
        old = manager._plugins_for("c1")[0]
        assert old.MARKER == "version-one"

        _write_pkg(tmp_path, name, "version-two-longer")
        assert "reloaded" in manager.reload_plugin("c1", "p")
        assert old.stopped, "reload should stop the old instance"

        new = manager._plugins_for("c1")[0]
        assert new.MARKER == "version-two-longer"
        assert type(new) is not type(old)
    finally:
        _drop_pkg_from_sys_modules(name)


def test_reload_failure_keeps_running_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken edit must not kill the currently running plugin."""
    name = "hotswap_pkg_fail"
    _write_pkg(tmp_path, name, "version-one")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        manager = _manager(PluginConfig(name="p", enabled=True, module=name))
        old = manager._plugins_for("c1")[0]

        (tmp_path / name / "impl.py").write_text("def broken(:\n", encoding="utf-8")
        shutil.rmtree(tmp_path / name / "__pycache__", ignore_errors=True)
        importlib.invalidate_caches()

        with pytest.raises(SyntaxError):
            manager.reload_plugin("c1", "p")

        assert manager._instances["c1"]["p"] is old
        assert not old.stopped
        assert manager._plugins_for("c1")[0] is old
    finally:
        _drop_pkg_from_sys_modules(name)


def test_reload_clears_instances_for_all_chats() -> None:
    """A module reload changes code globally; every chat's instance must recycle."""
    manager = _manager(PluginConfig(name="p", enabled=True, module=None))
    fakes: dict[str, MagicMock] = {}

    def factory(config: PluginConfig, chat_id: str, *args: object) -> MagicMock:
        fake = MagicMock()
        fakes[chat_id] = fake
        return fake

    manager._load_plugin = factory  # type: ignore[method-assign]
    manager._plugins_for("c1")
    manager._plugins_for("c2")
    assert set(fakes) == {"c1", "c2"}

    assert "reloaded" in manager.reload_plugin("c1", "p")
    assert "p" not in manager._instances.get("c1", {})
    assert "p" not in manager._instances.get("c2", {})
    for fake in fakes.values():
        fake.stop.assert_called_once()
