"""State plugin manager: load, dispatch, and lifecycle."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.config import McpServerConfig, PluginConfig
from diploid_agent.dispatch import Dispatch, DispatchStore
from diploid_agent.memory import MemoryItem
from diploid_agent.models import ChatResult, PartialTurn, SessionRecord, WakeEvent
from diploid_agent.plugins.base import StatePlugin, TurnInfo
from diploid_agent.plugins.broken import FailedPlugin
from diploid_agent.plugins.contexts import (
    DispatchCompleteContext,
    DispatchContinueContext,
    DispatchCreateContext,
    EngineCallContext,
    EngineResultContext,
    IdleContext,
    McpCommandContext,
    MemoryTransitionContext,
    PromoteContext,
    PromptBuildContext,
    PromptContext,
    RecordTurnContext,
    RetainContext,
    SessionActiveContext,
    SessionArchiveContext,
    SessionClearContext,
    SessionStartContext,
    ShutdownContext,
    SkillCommandContext,
    TurnErrorContext,
    TurnStartContext,
    UserMessageContext,
)
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
        self._snapshot_config(self._plugins)

    def plugin_health(self, chat_id: str) -> list[dict[str, Any]]:
        results = []
        for cfg in self._plugins:
            if not cfg.enabled:
                continue
            plugin = self._get_or_create(chat_id, cfg)
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
    # (plugins/hooks.py); the methods below are thin delegates kept for
    # call-site compatibility.

    def on_waking(
        self,
        chat_id: str,
        record: SessionRecord | None,
        now: float,
        *,
        wake_event: WakeEvent | None = None,
        other_instance_running: bool = False,
        rehydration_reason: str | None = None,
    ) -> None:
        self._hooks.on_waking(
            chat_id,
            record,
            now,
            wake_event=wake_event,
            other_instance_running=other_instance_running,
            rehydration_reason=rehydration_reason,
        )

    def fill_prompt_slots(
        self,
        chat_id: str,
        slots: dict[str, list[str]],
        is_first: bool,
        rehydrated: bool = False,
        last_blocks: dict[tuple[str, str], str | None] | None = None,
        last_prompt_time: float | None = None,
        force_slots: set[str] | None = None,
        compact: bool = False,
    ) -> dict[str, list[str]]:
        return self._hooks.fill_prompt_slots(
            chat_id,
            slots,
            is_first,
            rehydrated=rehydrated,
            last_blocks=last_blocks,
            last_prompt_time=last_prompt_time,
            force_slots=force_slots,
            compact=compact,
        )

    def mcp_server_configs(self) -> list[McpServerConfig]:
        return self._hooks.mcp_server_configs()

    def default_skill_names(self) -> list[str]:
        return self._hooks.default_skill_names()

    def default_mcp_names(self) -> list[str]:
        return self._hooks.default_mcp_names()

    def durable_files(self) -> list[str]:
        return self._hooks.durable_files()

    def event(
        self,
        chat_id: str,
        plugin_name: str,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> str:
        return self._hooks.event(chat_id, plugin_name, event=event, raw_args=raw_args, **params)

    def memory_items(self, chat_id: str, since: float) -> list[MemoryItem]:
        return self._hooks.memory_items(chat_id, since)

    def on_turn_end(self, chat_id: str, turn: TurnInfo) -> None:
        self._hooks.on_turn_end(chat_id, turn)

    def on_sleeping(self, chat_id: str, record: SessionRecord | None, reason: str) -> None:
        self._hooks.on_sleeping(chat_id, record, reason)

    def on_shutdown(self, chat_id: str, context: ShutdownContext) -> None:
        self._hooks.on_shutdown(chat_id, context)

    # ---------------------------------------------------------------- turn hooks

    def before_turn(
        self,
        chat_id: str,
        context: TurnStartContext,
    ) -> TurnStartContext | ChatResult | None:
        return self._hooks.before_turn(chat_id, context)

    def before_format_user_message(
        self,
        chat_id: str,
        context: UserMessageContext,
        formatter: Callable[[UserMessageContext], str],
    ) -> UserMessageContext:
        return self._hooks.before_format_user_message(chat_id, context, formatter)

    def before_build_prompt(
        self,
        chat_id: str,
        context: PromptBuildContext,
    ) -> PromptBuildContext:
        return self._hooks.before_build_prompt(chat_id, context)

    def after_prompt_built(
        self,
        chat_id: str,
        context: PromptContext,
    ) -> PromptContext:
        return self._hooks.after_prompt_built(chat_id, context)

    def before_engine_call(
        self,
        chat_id: str,
        context: EngineCallContext,
    ) -> EngineCallContext | ChatResult | None:
        return self._hooks.before_engine_call(chat_id, context)

    def after_engine_call(
        self,
        chat_id: str,
        context: EngineResultContext,
    ) -> EngineResultContext:
        return self._hooks.after_engine_call(chat_id, context)

    def before_record_turn(
        self,
        chat_id: str,
        context: RecordTurnContext,
    ) -> RecordTurnContext:
        return self._hooks.before_record_turn(chat_id, context)

    def after_turn(self, chat_id: str, turn: TurnInfo) -> None:
        self._hooks.after_turn(chat_id, turn)

    def on_turn_error(self, chat_id: str, context: TurnErrorContext) -> None:
        self._hooks.on_turn_error(chat_id, context)

    # ---------------------------------------------------------------- partial / dispatch / event / idle

    def on_partial(self, chat_id: str, partial: PartialTurn) -> None:
        self._hooks.on_partial(chat_id, partial)

    def on_dispatch(self, chat_id: str, dispatch: Dispatch) -> None:
        self._hooks.on_dispatch(chat_id, dispatch)

    def on_event(
        self,
        chat_id: str,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        self._hooks.on_event(chat_id, event, payload)

    def on_idle(self, chat_id: str, context: IdleContext) -> None:
        self._hooks.on_idle(chat_id, context)

    # ---------------------------------------------------------------- session hooks

    def before_session_archive(
        self,
        chat_id: str,
        context: SessionArchiveContext,
    ) -> SessionArchiveContext:
        return self._hooks.before_session_archive(chat_id, context)

    def before_session_clear(
        self,
        chat_id: str,
        context: SessionClearContext,
    ) -> SessionClearContext:
        return self._hooks.before_session_clear(chat_id, context)

    def before_session_start(
        self,
        chat_id: str,
        context: SessionStartContext,
    ) -> SessionStartContext | ChatResult | None:
        return self._hooks.before_session_start(chat_id, context)

    def after_session_active(
        self,
        chat_id: str,
        context: SessionActiveContext,
    ) -> SessionActiveContext:
        return self._hooks.after_session_active(chat_id, context)

    # ---------------------------------------------------------------- dispatch hooks

    def before_dispatch(
        self,
        chat_id: str,
        context: DispatchCreateContext,
    ) -> DispatchCreateContext:
        return self._hooks.before_dispatch(chat_id, context)

    def after_dispatch(
        self,
        chat_id: str,
        context: DispatchCreateContext,
    ) -> None:
        self._hooks.after_dispatch(chat_id, context)

    def before_dispatch_continue(
        self,
        chat_id: str,
        context: DispatchContinueContext,
    ) -> DispatchContinueContext:
        return self._hooks.before_dispatch_continue(chat_id, context)

    def after_dispatch_continue(
        self,
        chat_id: str,
        context: DispatchCompleteContext,
    ) -> None:
        self._hooks.after_dispatch_continue(chat_id, context)

    # ---------------------------------------------------------------- memory hooks

    def on_chat_memory_transition(
        self,
        chat_id: str,
        context: MemoryTransitionContext,
    ) -> MemoryTransitionContext:
        return self._hooks.on_chat_memory_transition(chat_id, context)

    def on_persona_memory_transition(
        self,
        chat_id: str,
        context: MemoryTransitionContext,
    ) -> MemoryTransitionContext:
        return self._hooks.on_persona_memory_transition(chat_id, context)

    # ---------------------------------------------------------------- wake / first prompt

    def after_first_prompt_built(self, chat_id: str, context: PromptContext) -> PromptContext:
        return self._hooks.after_first_prompt_built(chat_id, context)

    # ---------------------------------------------------------------- skill / mcp command hooks

    def before_skill_enabled(
        self, chat_id: str, context: SkillCommandContext
    ) -> SkillCommandContext:
        return self._hooks.before_skill_enabled(chat_id, context)

    def after_skill_enabled(self, chat_id: str, context: SkillCommandContext) -> None:
        self._hooks.after_skill_enabled(chat_id, context)

    def before_skill_disabled(
        self, chat_id: str, context: SkillCommandContext
    ) -> SkillCommandContext:
        return self._hooks.before_skill_disabled(chat_id, context)

    def after_skill_disabled(self, chat_id: str, context: SkillCommandContext) -> None:
        self._hooks.after_skill_disabled(chat_id, context)

    def before_mcp_enabled(self, chat_id: str, context: McpCommandContext) -> McpCommandContext:
        return self._hooks.before_mcp_enabled(chat_id, context)

    def after_mcp_enabled(self, chat_id: str, context: McpCommandContext) -> None:
        self._hooks.after_mcp_enabled(chat_id, context)

    def before_mcp_disabled(self, chat_id: str, context: McpCommandContext) -> McpCommandContext:
        return self._hooks.before_mcp_disabled(chat_id, context)

    def after_mcp_disabled(self, chat_id: str, context: McpCommandContext) -> None:
        self._hooks.after_mcp_disabled(chat_id, context)

    # ---------------------------------------------------------------- retain / promote hooks

    def before_retain(self, chat_id: str, context: RetainContext) -> RetainContext:
        return self._hooks.before_retain(chat_id, context)

    def after_retain(self, chat_id: str, context: RetainContext) -> None:
        self._hooks.after_retain(chat_id, context)

    def before_promote(self, chat_id: str, context: PromoteContext) -> PromoteContext:
        return self._hooks.before_promote(chat_id, context)

    def after_promote(self, chat_id: str, context: PromoteContext) -> None:
        self._hooks.after_promote(chat_id, context)
