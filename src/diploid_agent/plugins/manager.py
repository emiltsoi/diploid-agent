"""State plugin manager: load, dispatch, and lifecycle."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import re
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from diploid_agent.config import McpServerConfig, PluginConfig
from diploid_agent.dispatch import Dispatch, DispatchStatus, DispatchStore
from diploid_agent.memory import MemoryItem
from diploid_agent.models import ChatResult, PartialTurn, SessionRecord, WakeEvent
from diploid_agent.plugins.base import SleepContext, StatePlugin, TurnInfo, WakeContext
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
        """Replace the active plugin list, stop removed plugins, and snapshot.

        Existing on-disk state is preserved; plugins are lazily reloaded on the
        next turn that needs them.
        """
        new_plugins = [p for p in plugins if p.name]
        new_enabled = {p.name for p in new_plugins if p.enabled}

        # Stop instances for plugins that are removed or globally disabled.
        for chat_id, cache in list(self._instances.items()):
            for name in list(cache.keys()):
                if name not in new_enabled:
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
        """Restore the plugin list to an earlier snapshot."""
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
        return f"Rolled back {steps} plugin configuration(s)"

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
        record = self._runtime._active_record(chat_id) if self._runtime else None
        if record and record.plugin_overrides and config.name in record.plugin_overrides:
            return record.plugin_overrides[config.name]
        return config.enabled

    def _plugins_for(self, chat_id: str) -> list[StatePlugin]:
        enabled = [p for p in self._plugins if self._is_enabled_for(chat_id, p)]
        return [self._get_or_create(chat_id, cfg) for cfg in enabled]

    def set_plugin_enabled(self, chat_id: str, name: str, enabled: bool) -> str:
        record = self._runtime._active_record(chat_id) if self._runtime else None
        if record is None:
            return f"No active session for {chat_id}"
        if record.plugin_overrides is None:
            record.plugin_overrides = {}
        record.plugin_overrides[name] = enabled
        self._runtime._append_record(record)
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

    def _pending_dispatches(self, chat_id: str) -> list[dict[str, Any]]:
        if self._dispatch_store is None:
            return []
        return [
            d.to_dict() for d in self._dispatch_store.list_by_chat(chat_id, DispatchStatus.PENDING)
        ]

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
        previous_turn_at = record.updated_at if record is not None else None
        context = WakeContext(
            chat_id=chat_id,
            record=record,
            now=now,
            instance_id=self._instance_id,
            instance_started_at=self._instance_started_at,
            previous_turn_at=previous_turn_at,
            pending_dispatches=self._pending_dispatches(chat_id),
            wake_event=wake_event,
            other_instance_running=other_instance_running,
            rehydration_reason=rehydration_reason,
        )
        self._notify_hook(chat_id, "on_waking", context)

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
        force_slots = force_slots or set()
        for plugin in self._plugins_for(chat_id):
            slot = plugin.prompt_slot
            force = slot in force_slots
            if plugin.first_prompt_only and not is_first and not rehydrated and not force:
                continue

            key = (plugin.name, slot)

            if not is_first and not force and last_blocks is not None:
                changed = plugin.prompt_block_changed(last_prompt_time)
                if changed is not None and not changed:
                    continue

            max_chars = plugin.max_prompt_chars
            if compact and max_chars != 0:
                max_chars = (
                    min(max_chars, self._runtime.config.harness.compact_plugin_max_chars)
                    if self._runtime is not None
                    else min(max_chars, 200)
                )

            block = self._plugin_prompt_block(plugin, max_chars, compact=compact)
            if block is None:
                continue

            if not is_first and last_blocks is not None:
                if not force and block == last_blocks.get(key):
                    continue
                last_blocks[key] = block
            elif is_first and last_blocks is not None:
                last_blocks[key] = block

            if block:
                slots.setdefault(slot, []).append(block)
        return slots

    def _plugin_accepts_compact(self, plugin: StatePlugin) -> bool:
        """Return True if the plugin's prompt_block accepts a `compact` keyword."""
        try:
            sig = inspect.signature(plugin.prompt_block)
        except (ValueError, TypeError):
            return False
        for name, param in sig.parameters.items():
            if name == "compact":
                return True
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    def _plugin_prompt_block(
        self, plugin: StatePlugin, max_chars: int | None, compact: bool
    ) -> str | None:
        """Call a plugin's prompt_block, passing `compact` only if it supports it."""
        try:
            if compact and self._plugin_accepts_compact(plugin):
                block = plugin.prompt_block(max_chars, compact=True)
            else:
                block = plugin.prompt_block(max_chars)
        except Exception:
            logger.exception("prompt_block failed for plugin %s", plugin.name)
            return None
        return block

    def mcp_server_configs(self) -> list[McpServerConfig]:
        servers = []
        for cfg in self._plugins:
            if cfg.enabled and cfg.mcp_server and not cfg.mcp_server.disabled:
                servers.append(cfg.mcp_server)
        return servers

    def default_skill_names(self) -> list[str]:
        return [p.skill for p in self._plugins if p.enabled and p.skill]

    def default_mcp_names(self) -> list[str]:
        return [s.name for s in self.mcp_server_configs()]

    def durable_files(self) -> list[str]:
        files = []
        for cfg in self._plugins:
            if cfg.state_file:
                files.append(cfg.state_file)
        return files

    def event(
        self,
        chat_id: str,
        plugin_name: str,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> str:
        for cfg in self._plugins:
            if cfg.name == plugin_name:
                plugin = self._get_or_create(chat_id, cfg)
                result = plugin.event(event=event, raw_args=raw_args, **params)
                self.on_event(
                    chat_id,
                    event or "unknown",
                    {"plugin": plugin_name, **params},
                )
                return result
        raise KeyError(f"Unknown plugin: {plugin_name}")

    def memory_items(self, chat_id: str, since: float) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for plugin in self._plugins_for(chat_id):
            try:
                items.extend(plugin.memory_items(since))
            except Exception:
                logger.exception("memory_items failed for plugin %s", plugin.name)
        return items

    def on_turn_end(self, chat_id: str, turn: TurnInfo) -> None:
        self._notify_hook(chat_id, "on_turn_end", turn)

    def on_sleeping(self, chat_id: str, record: SessionRecord | None, reason: str) -> None:
        context = SleepContext(
            chat_id=chat_id,
            record=record,
            reason=reason,
            now=time.time(),
            instance_id=self._instance_id,
        )
        self._notify_hook(chat_id, "on_sleeping", context)

    def on_shutdown(self, chat_id: str, context: ShutdownContext) -> None:
        self._notify_hook(chat_id, "on_shutdown", context)
        self._notify_hook(
            chat_id,
            "on_sleeping",
            SleepContext(
                chat_id=chat_id,
                record=context.record,
                reason="shutdown",
                now=context.now,
                instance_id=context.instance_id,
            ),
        )

    # ---------------------------------------------------------------- hook dispatcher

    T = TypeVar("T")

    def _notify_hook(self, chat_id: str, hook_name: str, *args: Any) -> None:
        """Void fan-out: call ``hook_name`` on every plugin; log and continue on error."""
        for plugin in self._plugins_for(chat_id):
            method = getattr(plugin, hook_name, None)
            if method is None:
                continue
            try:
                method(*args)
            except Exception:
                logger.exception("%s failed for plugin %s", hook_name, plugin.name)

    def _apply_hook(
        self,
        chat_id: str,
        hook_name: str,
        context: T,
        can_short_circuit: bool = False,
        prepare: Callable[[T], None] | None = None,
    ) -> T | ChatResult | None:
        """Run a hook across all plugins for this chat.

        Returns the final context, or ChatResult if a gate hook short-circuits.
        ``prepare`` runs on the current context before each plugin call.
        """
        for plugin in self._plugins_for(chat_id):
            method = getattr(plugin, hook_name, None)
            if method is None:
                continue
            if prepare is not None:
                prepare(context)
            try:
                result = method(context)
            except Exception:
                logger.exception("%s failed for plugin %s", hook_name, plugin.name)
                continue
            if result is None:
                continue
            if isinstance(result, ChatResult):
                if not can_short_circuit:
                    logger.error(
                        "Plugin %s returned ChatResult from non-gate hook %s; ignoring",
                        plugin.name,
                        hook_name,
                    )
                    continue
                return result
            context = result
        return context

    def before_turn(
        self,
        chat_id: str,
        context: TurnStartContext,
    ) -> TurnStartContext | ChatResult | None:
        return self._apply_hook(chat_id, "before_turn", context, can_short_circuit=True)

    def before_format_user_message(
        self,
        chat_id: str,
        context: UserMessageContext,
        formatter: Callable[[UserMessageContext], str],
    ) -> UserMessageContext:
        """Consult hook: plugins can modify the raw/formatted user message."""

        def _ensure_formatted(ctx: UserMessageContext) -> None:
            if ctx.formatted_message is None:
                ctx.formatted_message = formatter(ctx)

        result = self._apply_hook(
            chat_id,
            "before_format_user_message",
            context,
            prepare=_ensure_formatted,
        )
        if result is None or isinstance(result, ChatResult):
            result = context
        _ensure_formatted(result)
        return result

    def before_build_prompt(
        self,
        chat_id: str,
        context: PromptBuildContext,
    ) -> PromptBuildContext:
        return self._apply_consult_hook(chat_id, "before_build_prompt", context)

    def after_prompt_built(
        self,
        chat_id: str,
        context: PromptContext,
    ) -> PromptContext:
        return self._apply_consult_hook(chat_id, "after_prompt_built", context)

    def before_engine_call(
        self,
        chat_id: str,
        context: EngineCallContext,
    ) -> EngineCallContext | ChatResult | None:
        return self._apply_hook(chat_id, "before_engine_call", context, can_short_circuit=True)

    def after_engine_call(
        self,
        chat_id: str,
        context: EngineResultContext,
    ) -> EngineResultContext:
        return self._apply_consult_hook(chat_id, "after_engine_call", context)

    def before_record_turn(
        self,
        chat_id: str,
        context: RecordTurnContext,
    ) -> RecordTurnContext:
        return self._apply_consult_hook(chat_id, "before_record_turn", context)

    def after_turn(self, chat_id: str, turn: TurnInfo) -> None:
        self._notify_hook(chat_id, "after_turn", turn)

    def on_turn_error(self, chat_id: str, context: TurnErrorContext) -> None:
        self._notify_hook(chat_id, "on_turn_error", context)

    # ---------------------------------------------------------------- partial / dispatch / event / idle

    def on_partial(self, chat_id: str, partial: PartialTurn) -> None:
        self._notify_hook(chat_id, "on_partial", partial)

    def on_dispatch(self, chat_id: str, dispatch: Dispatch) -> None:
        self._notify_hook(chat_id, "on_dispatch", chat_id, dispatch)

    def on_event(
        self,
        chat_id: str,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        self._notify_hook(chat_id, "on_event", event, payload)

    def on_idle(self, chat_id: str, context: IdleContext) -> None:
        self._notify_hook(chat_id, "on_idle", context)

    # ---------------------------------------------------------------- session hooks

    def before_session_archive(
        self,
        chat_id: str,
        context: SessionArchiveContext,
    ) -> SessionArchiveContext:
        return self._apply_consult_hook(chat_id, "before_session_archive", context)

    def before_session_clear(
        self,
        chat_id: str,
        context: SessionClearContext,
    ) -> SessionClearContext:
        return self._apply_consult_hook(chat_id, "before_session_clear", context)

    def before_session_start(
        self,
        chat_id: str,
        context: SessionStartContext,
    ) -> SessionStartContext | ChatResult | None:
        return self._apply_hook(chat_id, "before_session_start", context, can_short_circuit=True)

    def after_session_active(
        self,
        chat_id: str,
        context: SessionActiveContext,
    ) -> SessionActiveContext:
        return self._apply_consult_hook(chat_id, "after_session_active", context)

    # ---------------------------------------------------------------- dispatch hooks

    def before_dispatch(
        self,
        chat_id: str,
        context: DispatchCreateContext,
    ) -> DispatchCreateContext:
        return self._apply_consult_hook(chat_id, "before_dispatch", context)

    def after_dispatch(
        self,
        chat_id: str,
        context: DispatchCreateContext,
    ) -> None:
        self._notify_hook(chat_id, "after_dispatch", context)

    def before_dispatch_continue(
        self,
        chat_id: str,
        context: DispatchContinueContext,
    ) -> DispatchContinueContext:
        return self._apply_consult_hook(chat_id, "before_dispatch_continue", context)

    def after_dispatch_continue(
        self,
        chat_id: str,
        context: DispatchCompleteContext,
    ) -> None:
        self._notify_hook(chat_id, "after_dispatch_continue", context)

    # ---------------------------------------------------------------- memory hooks

    def on_chat_memory_transition(
        self,
        chat_id: str,
        context: MemoryTransitionContext,
    ) -> MemoryTransitionContext:
        return self._apply_consult_hook(chat_id, "on_chat_memory_transition", context)

    def on_persona_memory_transition(
        self,
        chat_id: str,
        context: MemoryTransitionContext,
    ) -> MemoryTransitionContext:
        return self._apply_consult_hook(chat_id, "on_persona_memory_transition", context)

    # ---------------------------------------------------------------- wake / first prompt

    def after_first_prompt_built(self, chat_id: str, context: PromptContext) -> PromptContext:
        return self._apply_consult_hook(chat_id, "after_first_prompt_built", context)

    # ---------------------------------------------------------------- skill / mcp command hooks

    def _apply_consult_hook(
        self,
        chat_id: str,
        hook_name: str,
        context: T,
    ) -> T:
        """Consult hook: plugins may mutate/replace context; ChatResult is ignored."""
        result = self._apply_hook(chat_id, hook_name, context, can_short_circuit=False)
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    def before_skill_enabled(
        self, chat_id: str, context: SkillCommandContext
    ) -> SkillCommandContext:
        return self._apply_consult_hook(chat_id, "before_skill_enabled", context)

    def after_skill_enabled(self, chat_id: str, context: SkillCommandContext) -> None:
        self._notify_hook(chat_id, "after_skill_enabled", context)

    def before_skill_disabled(
        self, chat_id: str, context: SkillCommandContext
    ) -> SkillCommandContext:
        return self._apply_consult_hook(chat_id, "before_skill_disabled", context)

    def after_skill_disabled(self, chat_id: str, context: SkillCommandContext) -> None:
        self._notify_hook(chat_id, "after_skill_disabled", context)

    def before_mcp_enabled(self, chat_id: str, context: McpCommandContext) -> McpCommandContext:
        return self._apply_consult_hook(chat_id, "before_mcp_enabled", context)

    def after_mcp_enabled(self, chat_id: str, context: McpCommandContext) -> None:
        self._notify_hook(chat_id, "after_mcp_enabled", context)

    def before_mcp_disabled(self, chat_id: str, context: McpCommandContext) -> McpCommandContext:
        return self._apply_consult_hook(chat_id, "before_mcp_disabled", context)

    def after_mcp_disabled(self, chat_id: str, context: McpCommandContext) -> None:
        self._notify_hook(chat_id, "after_mcp_disabled", context)

    # ---------------------------------------------------------------- retain / promote hooks

    def before_retain(self, chat_id: str, context: RetainContext) -> RetainContext:
        return self._apply_consult_hook(chat_id, "before_retain", context)

    def after_retain(self, chat_id: str, context: RetainContext) -> None:
        self._notify_hook(chat_id, "after_retain", context)

    def before_promote(self, chat_id: str, context: PromoteContext) -> PromoteContext:
        return self._apply_consult_hook(chat_id, "before_promote", context)

    def after_promote(self, chat_id: str, context: PromoteContext) -> None:
        self._notify_hook(chat_id, "after_promote", context)
