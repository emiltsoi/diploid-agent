"""State plugin manager: load, dispatch, and lifecycle."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from diploid_agent.config import PluginConfig
from diploid_agent.dispatch import DispatchStore
from diploid_agent.plugins.base import StatePlugin
from diploid_agent.plugins.broken import FailedPlugin
from diploid_agent.plugins.hooks import PluginHooks
from diploid_agent.runtime.plugin_runtime import PluginRuntime

logger = logging.getLogger(__name__)


class PluginManager:
    """Owns all enabled per-chat state plugins."""

    def __init__(
        self,
        plugins: list[PluginConfig],
        sessions_root: Path,
        instance_id: str,
        instance_started_at: float,
        dispatch_store: DispatchStore | None = None,
        runtime: PluginRuntime | None = None,
        incident_store: Any | None = None,
    ) -> None:
        self._sessions_root = sessions_root
        self._instance_id = instance_id
        self._instance_started_at = instance_started_at
        self._dispatch_store = dispatch_store
        self._runtime: PluginRuntime | None = runtime
        self._incident_store = incident_store
        self._instances: dict[str, dict[str, StatePlugin]] = defaultdict(dict)
        self._config_history: list[list[PluginConfig]] = []
        # Bumped whenever a chat's effective enabled-plugin set can change
        # (reconfigure, rollback, disable, per-chat override); PluginHooks
        # caches resolved config lists against it.
        self._generation = 0
        self._hooks = PluginHooks(self)
        self.reconfigure(plugins)

    def _record_incident(
        self,
        plugin: str,
        phase: str,
        error: str,
        action: str = "",
        chat_id: str = "",
    ) -> None:
        if self._incident_store is not None:
            self._incident_store.record(
                plugin=plugin,
                chat_id=chat_id or "",
                phase=phase,
                error=error,
                action=action,
            )

    def reconfigure(self, plugins: list[PluginConfig]) -> None:
        """Replace the active plugin list, stop all cached instances, and snapshot.

        Existing on-disk state is preserved; plugins are lazily reloaded on the
        next turn that needs them.  Every cached instance is stop()ed first —
        including still-enabled plugins, which are recycled so they pick up the
        new config — otherwise reconfigure would orphan live instances.
        """
        new_plugins = [p for p in plugins if p.name]

        # Stop every cached instance; retained plugins get a fresh instance on
        # next use, removed/disabled ones are gone for good.
        for chat_id, cache in list(self._instances.items()):
            for name in list(cache.keys()):
                plugin = cache.pop(name, None)
                if not isinstance(plugin, FailedPlugin):
                    try:
                        plugin.stop()
                    except Exception:
                        logger.exception("stop() failed for plugin %s", name)
                        self._record_incident(
                            plugin=name,
                            phase="lifecycle",
                            error=traceback.format_exc(),
                            action="failed_plugin",
                            chat_id=chat_id,
                        )

        self._plugins = sorted(new_plugins, key=lambda p: p.prompt_order)
        self._instances.clear()
        self._generation += 1
        self._snapshot_config(self._plugins)

    def _snapshot_config(self, plugins: list[PluginConfig]) -> None:
        """Store a deep copy of the current plugin list."""
        self._config_history.append(
            [PluginConfig(**p.model_dump(exclude_none=False)) for p in plugins]
        )
        if len(self._config_history) > 10:
            self._config_history.pop(0)

    def add_plugin(self, config: PluginConfig) -> str:
        """Append a new plugin to the live config."""
        if not config.name:
            raise ValueError("Plugin must have a name")
        if any(p.name == config.name for p in self._plugins):
            raise ValueError(f"Plugin {config.name} already exists")
        self._plugins.append(config)
        self._plugins.sort(key=lambda p: p.prompt_order)
        self._snapshot_config(self._plugins)
        return f"Plugin {config.name} added"

    def remove_plugin(self, name: str) -> str:
        """Remove a plugin from the live config and stop all its instances."""
        cfg = next((p for p in self._plugins if p.name == name), None)
        if cfg is None:
            raise ValueError(f"Unknown plugin: {name}")
        self._plugins = [p for p in self._plugins if p.name != name]
        for chat_id, cache in list(self._instances.items()):
            plugin = cache.pop(name, None)
            if plugin is not None and not isinstance(plugin, FailedPlugin):
                try:
                    plugin.stop()
                except Exception:
                    logger.exception("stop() failed for plugin %s", name)
                    self._record_incident(
                        plugin=name,
                        phase="lifecycle",
                        error=traceback.format_exc(),
                        action="failed_plugin",
                        chat_id=chat_id,
                    )
        self._snapshot_config(self._plugins)
        return f"Plugin {name} removed"

    def toggle_plugin(self, name: str, enabled: bool) -> str:
        """Toggle a plugin on or off globally."""
        cfg = next((p for p in self._plugins if p.name == name), None)
        if cfg is None:
            raise ValueError(f"Unknown plugin: {name}")
        cfg.enabled = enabled
        if not enabled:
            for chat_id, cache in list(self._instances.items()):
                plugin = cache.pop(name, None)
                if plugin is not None and not isinstance(plugin, FailedPlugin):
                    try:
                        plugin.stop()
                    except Exception:
                        logger.exception("stop() failed for plugin %s", name)
                        self._record_incident(
                            plugin=name,
                            phase="lifecycle",
                            error=traceback.format_exc(),
                            action="failed_plugin",
                            chat_id=chat_id,
                        )
        self._snapshot_config(self._plugins)
        return f"Plugin {name} {'enabled' if enabled else 'disabled'}"

    def rollback(self, steps: int = 1) -> str:
        """Restore the plugin list to an earlier snapshot.

        Only the global plugin list is restored — per-chat
        ``plugin_overrides`` on SessionRecord are not part of the
        snapshots and stay untouched; overrides that still mask the
        restored state are named in the returned message.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if len(self._config_history) < steps + 1:
            raise ValueError("No earlier configuration to roll back to")
        # Stop instances for any plugin that will disappear or be disabled.
        previous_names = {p.name for p in self._plugins if p.enabled}
        target = self._config_history[-(steps + 1)]
        target_names = {p.name for p in target if p.enabled}
        for name in previous_names - target_names:
            for chat_id, cache in list(self._instances.items()):
                plugin = cache.pop(name, None)
                if plugin is not None and not isinstance(plugin, FailedPlugin):
                    try:
                        plugin.stop()
                    except Exception:
                        logger.exception("stop() failed during rollback for plugin %s", name)
                        self._record_incident(
                            plugin=name,
                            phase="rollback",
                            error=traceback.format_exc(),
                            action="failed_plugin",
                            chat_id=chat_id,
                        )
        self._plugins = [PluginConfig(**p.model_dump()) for p in target]
        self._generation += 1
        # Replace the history tail with the restored config so the next snapshot is clean.
        self._config_history = self._config_history[: -(steps + 1)]
        self._snapshot_config(self._plugins)
        masked = self._masking_overrides()
        suffix = f"; per-chat overrides still active: {masked}" if masked else ""
        return f"Rolled back {steps} plugin configuration(s){suffix}"

    def _masking_overrides(self) -> str:
        """Per-chat overrides whose value differs from the restored global state."""
        if self._runtime is None:
            return ""
        global_enabled = {p.name: p.enabled for p in self._plugins}
        notes: list[str] = []
        for chat_id in sorted(self._instances):
            record = self._runtime.active_record(chat_id)
            if record is None or not record.plugin_overrides:
                continue
            masked = sorted(
                name
                for name, enabled in record.plugin_overrides.items()
                if name in global_enabled and enabled != global_enabled[name]
            )
            if masked:
                notes.append(f"{chat_id} ({', '.join(masked)})")
        return ", ".join(notes)

    def stop_all(self) -> None:
        """Stop every plugin instance and release all caches."""
        for chat_id, cache in list(self._instances.items()):
            for name, plugin in list(cache.items()):
                if not isinstance(plugin, FailedPlugin):
                    try:
                        plugin.stop()
                    except Exception:
                        logger.exception("stop() failed for plugin %s", name)
                        self._record_incident(
                            plugin=name,
                            phase="lifecycle",
                            error=traceback.format_exc(),
                            action="failed_plugin",
                            chat_id=chat_id,
                        )
        self._instances.clear()

    def _get_or_create(self, chat_id: str, config: PluginConfig) -> StatePlugin:
        cache = self._instances[chat_id]
        if config.name not in cache:
            try:
                plugin = self._load_plugin(config, chat_id)
            except Exception:
                logger.exception("Failed to load plugin %s", config.name)
                self._record_incident(
                    plugin=config.name,
                    phase="lifecycle",
                    error=traceback.format_exc(),
                    action="failed_plugin",
                    chat_id=chat_id,
                )
                plugin = FailedPlugin(
                    config,
                    chat_id,
                    self._sessions_root,
                    runtime=self._runtime,
                    error=traceback.format_exc(),
                )
            cache[config.name] = plugin
            if not isinstance(plugin, FailedPlugin):
                try:
                    plugin.start()
                except Exception:
                    logger.exception("start() failed for plugin %s", config.name)
                    self._record_incident(
                        plugin=config.name,
                        phase="lifecycle",
                        error=traceback.format_exc(),
                        action="failed_plugin",
                        chat_id=chat_id,
                    )
                    cache[config.name] = FailedPlugin(
                        config,
                        chat_id,
                        self._sessions_root,
                        runtime=self._runtime,
                        error=traceback.format_exc(),
                    )
        return cache[config.name]

    _MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

    @staticmethod
    def validate_module(module: str | None) -> None:
        """Validate a plugin module name without instantiating it.

        Raises ValueError for unsafe names and ImportError if the module cannot
        be loaded or does not expose a ``Plugin`` class.
        """
        if not module:
            return
        if not PluginManager._MODULE_NAME_RE.match(module) or ".." in module:
            raise ValueError(f"Invalid or unsafe plugin module name: {module}")
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None or spec.origin in ("built-in", "frozen"):
            raise ImportError(f"Plugin module {module} cannot be loaded or is not a file")
        mod = importlib.import_module(module)
        if not hasattr(mod, "Plugin"):
            raise ImportError(f"Plugin module {module} must expose a 'Plugin' class")

    def validate_all(self) -> list[str]:
        """Return names of plugins whose module cannot be loaded."""
        failed: list[str] = []
        for cfg in self._plugins:
            if cfg.module:
                try:
                    self.validate_module(cfg.module)
                except Exception:
                    logger.exception("Validation failed for plugin %s", cfg.name)
                    self._record_incident(
                        plugin=cfg.name,
                        phase="startup",
                        error=traceback.format_exc(),
                        action="validation_failed",
                    )
                    failed.append(cfg.name)
        return failed

    def disable_plugins(self, names: set[str]) -> None:
        """Disable the named plugins and snapshot the new config."""
        for cfg in self._plugins:
            if cfg.name in names:
                cfg.enabled = False
        self._generation += 1
        self._snapshot_config(self._plugins)

    def plugin_health(self, chat_id: str) -> list[dict[str, Any]]:
        results = []
        for cfg in self._plugins:
            if not cfg.enabled:
                continue
            plugin = self._instances.get(chat_id, {}).get(cfg.name)
            if plugin is None:
                results.append(
                    {"name": cfg.name, "healthy": True, "details": {"status": "not_started"}}
                )
                continue
            if isinstance(plugin, FailedPlugin):
                results.append({"name": cfg.name, "healthy": False, "error": plugin.error})
            else:
                try:
                    h = plugin.health()
                except Exception:
                    logger.exception("health() failed for plugin %s", cfg.name)
                    self._record_incident(
                        plugin=cfg.name,
                        phase="health",
                        error=traceback.format_exc(),
                        action="failed_plugin",
                        chat_id=chat_id,
                    )
                    h = {"error": traceback.format_exc()}
                results.append(
                    {
                        "name": cfg.name,
                        "healthy": h is None or h.get("healthy", True),
                        "details": h,
                    }
                )
        return results

    def validate_contract(self, module: str) -> list[str]:
        """Return contract violations for a plugin module without instantiating it."""
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            return [f"Failed to import {module}: {exc}"]
        if not hasattr(mod, "Plugin"):
            return [f"Module {module} must expose a 'Plugin' class"]
        return []

    def _load_plugin(self, config: PluginConfig, chat_id: str) -> StatePlugin:
        if config.module:
            self.validate_module(config.module)
            module = importlib.import_module(config.module)
            return module.Plugin(config, chat_id, self._sessions_root, self._runtime)

        from diploid_agent.plugins.json_state import JsonStatePlugin

        return JsonStatePlugin(config, chat_id, self._sessions_root, self._runtime)

    def _is_enabled_for(self, chat_id: str, config: PluginConfig) -> bool:
        record = self._runtime.active_record(chat_id) if self._runtime else None
        if record and record.plugin_overrides and config.name in record.plugin_overrides:
            return record.plugin_overrides[config.name]
        return config.enabled

    def _plugins_for(self, chat_id: str) -> list[StatePlugin]:
        return self._hooks._plugins_for(chat_id)

    def set_plugin_enabled(self, chat_id: str, name: str, enabled: bool) -> str:
        record = self._runtime.active_record(chat_id) if self._runtime else None
        if record is None:
            return f"No active session for {chat_id}"
        if record.plugin_overrides is None:
            record.plugin_overrides = {}
        record.plugin_overrides[name] = enabled
        self._runtime.append_record(record)
        self._generation += 1
        instance = self._instances[chat_id].pop(name, None)
        if instance is not None and not isinstance(instance, FailedPlugin):
            try:
                instance.stop()
            except Exception:
                logger.exception("stop() failed for plugin %s", name)
                self._record_incident(
                    plugin=name,
                    phase="lifecycle",
                    error=traceback.format_exc(),
                    action="failed_plugin",
                    chat_id=chat_id,
                )
        return f"Plugin {name} {'enabled' if enabled else 'disabled'}"

    def list_plugin_status(self, chat_id: str) -> list[dict[str, Any]]:
        result = []
        for cfg in self._plugins:
            enabled = self._is_enabled_for(chat_id, cfg)
            instance = self._instances.get(chat_id, {}).get(cfg.name)
            failed = isinstance(instance, FailedPlugin)
            result.append(
                {
                    "name": cfg.name,
                    "enabled": enabled,
                    "module": cfg.module,
                    "prompt_slot": cfg.prompt_slot,
                    "state_file": cfg.state_file,
                    "failed": failed,
                }
            )
        return result

    def reload_plugin(self, chat_id: str, name: str) -> str:
        """Hot-swap a plugin: reload its module tree, then recycle instances.

        ``chat_id`` is accepted for call-site compatibility; a module reload
        changes the code for every chat, so instances are recycled for all
        chats. The module tree is reloaded *before* any instance is dropped,
        so a broken edit raises here and leaves the running instances
        untouched.
        """
        cfg = next((p for p in self._plugins if p.name == name), None)
        if cfg is None:
            return f"Unknown plugin: {name}"
        if cfg.module:
            self._deep_reload(cfg.module)
        for cid, cache in list(self._instances.items()):
            plugin = cache.pop(name, None)
            if plugin is not None and not isinstance(plugin, FailedPlugin):
                try:
                    plugin.stop()
                except Exception:
                    logger.exception("stop() failed for plugin %s during reload", name)
                    self._record_incident(
                        plugin=name,
                        phase="lifecycle",
                        error=traceback.format_exc(),
                        action="failed_plugin",
                        chat_id=cid,
                    )
        return f"Plugin {name} reloaded"

    @staticmethod
    def _deep_reload(module_name: str) -> None:
        """Reload ``module_name`` and its already-imported submodules, deepest first.

        Plain ``importlib.reload`` on a package only re-executes ``__init__.py``;
        a ``from .impl import Plugin`` there still binds the stale submodule
        cached in ``sys.modules``. Reloading the subtree first makes the
        package rebind the fresh code. Submodules that were never imported are
        skipped and will import fresh on first use.
        """
        module = importlib.import_module(module_name)
        prefix = module_name + "."
        submodules = sorted(
            (
                (mod_name, mod)
                for mod_name, mod in list(sys.modules.items())
                if mod_name.startswith(prefix)
                and mod is not None
                and getattr(mod, "__file__", None) is not None
            ),
            key=lambda item: item[0].count("."),
            reverse=True,
        )
        for _, mod in submodules:
            importlib.reload(mod)
        importlib.reload(module)

    def load_errors(self, chat_id: str) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = []
        for cfg in self._plugins:
            if cfg.name in self._instances.get(chat_id, {}):
                plugin = self._instances[chat_id][cfg.name]
                if isinstance(plugin, FailedPlugin) and plugin.error:
                    errors.append({"name": cfg.name, "error": plugin.error})
        return errors

    # ---------------------------------------------------------------- hook dispatch
    #
    # Hook fan-out and consult dispatch live in ``PluginHooks``
    # (plugins/hooks.py). Every name in ``_HOOK_DELEGATES`` forwards to the
    # same-named ``PluginHooks`` method via ``__getattr__``, so delegates
    # expose the real signatures and adding a hook is a table entry.

    _HOOK_DELEGATES: frozenset[str] = frozenset(
        {
            "on_waking",
            "fill_prompt_slots",
            "mcp_server_configs",
            "default_skill_names",
            "default_mcp_names",
            "durable_files",
            "event",
            "memory_items",
            "on_turn_end",
            "on_sleeping",
            "on_shutdown",
            "before_turn",
            "before_format_user_message",
            "before_build_prompt",
            "after_prompt_built",
            "before_engine_call",
            "after_engine_call",
            "before_record_turn",
            "after_turn",
            "on_turn_error",
            "on_partial",
            "on_dispatch",
            "on_event",
            "on_idle",
            "before_session_archive",
            "before_session_clear",
            "before_session_start",
            "after_session_active",
            "before_dispatch",
            "after_dispatch",
            "before_dispatch_continue",
            "after_dispatch_continue",
            "on_chat_memory_transition",
            "on_persona_memory_transition",
            "after_first_prompt_built",
            "before_skill_enabled",
            "after_skill_enabled",
            "before_skill_disabled",
            "after_skill_disabled",
            "before_mcp_enabled",
            "after_mcp_enabled",
            "before_mcp_disabled",
            "after_mcp_disabled",
            "before_retain",
            "after_retain",
            "before_promote",
            "after_promote",
        }
    )

    def __getattr__(self, name: str) -> Any:
        if name in self._HOOK_DELEGATES:
            return getattr(self._hooks, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
