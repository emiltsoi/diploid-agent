"""State plugin manager: load, dispatch, and lifecycle."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from acp_fleet_harness.config import McpServerConfig, PluginConfig
from acp_fleet_harness.dispatch import Dispatch, DispatchStatus, DispatchStore
from acp_fleet_harness.memory import MemoryItem
from acp_fleet_harness.models import ChatResult, PartialTurn, SessionRecord, WakeEvent
from acp_fleet_harness.plugins.base import SleepContext, StatePlugin, TurnInfo, WakeContext
from acp_fleet_harness.plugins.broken import FailedPlugin
from acp_fleet_harness.plugins.contexts import (
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
from acp_fleet_harness.runtime.plugin_runtime import PluginRuntime

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
    ) -> None:
        self._sessions_root = sessions_root
        self._instance_id = instance_id
        self._instance_started_at = instance_started_at
        self._dispatch_store = dispatch_store
        self._runtime: PluginRuntime | None = runtime
        self._instances: dict[str, dict[str, StatePlugin]] = defaultdict(dict)
        self._config_history: list[list[PluginConfig]] = []
        self.reconfigure(plugins)

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
                        except BaseException:
                            logger.exception("stop() failed for plugin %s", name)

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
                except BaseException:
                    logger.exception("stop() failed for plugin %s", name)
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
                    except BaseException:
                        logger.exception("stop() failed for plugin %s", name)
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
                    except BaseException:
                        logger.exception("stop() failed during rollback for plugin %s", name)
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
                    except BaseException:
                        logger.exception("stop() failed for plugin %s", name)
        self._instances.clear()

    def _get_or_create(self, chat_id: str, config: PluginConfig) -> StatePlugin:
        cache = self._instances[chat_id]
        if config.name not in cache:
            try:
                plugin = self._load_plugin(config, chat_id)
            except BaseException:
                logger.exception("Failed to load plugin %s", config.name)
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
                except BaseException:
                    logger.exception("start() failed for plugin %s", config.name)
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
                    failed.append(cfg.name)
        return failed

    def disable_plugins(self, names: set[str]) -> None:
        """Disable the named plugins and snapshot the new config."""
        for cfg in self._plugins:
            if cfg.name in names:
                cfg.enabled = False
        self._snapshot_config(self._plugins)

    def _load_plugin(self, config: PluginConfig, chat_id: str) -> StatePlugin:
        if config.module:
            self.validate_module(config.module)
            module = importlib.import_module(config.module)
            return module.Plugin(config, chat_id, self._sessions_root, self._runtime)

        from acp_fleet_harness.plugins.json_state import JsonStatePlugin

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
            except BaseException:
                logger.exception("stop() failed for plugin %s", name)
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
        cfg = next((p for p in self._plugins if p.name == name), None)
        if cfg is None:
            return f"Unknown plugin: {name}"
        self._instances[chat_id].pop(name, None)
        if cfg.module:
            mod = importlib.import_module(cfg.module)
            importlib.reload(mod)
        return f"Plugin {name} reloaded"

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
        )
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_waking(context)
            except Exception:
                logger.exception("on_waking failed for plugin %s", plugin.name)

    def fill_prompt_slots(
        self,
        chat_id: str,
        slots: dict[str, list[str]],
        is_first: bool,
    ) -> dict[str, list[str]]:
        for plugin in self._plugins_for(chat_id):
            if plugin.first_prompt_only and not is_first:
                continue
            try:
                block = plugin.prompt_block(plugin.max_prompt_chars)
            except Exception:
                logger.exception("prompt_block failed for plugin %s", plugin.name)
                continue
            if block:
                slot = plugin.prompt_slot
                slots.setdefault(slot, []).append(block)
        return slots

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
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_turn_end(turn)
            except Exception:
                logger.exception("on_turn_end failed for plugin %s", plugin.name)

    def on_sleeping(self, chat_id: str, record: SessionRecord | None, reason: str) -> None:
        context = SleepContext(
            chat_id=chat_id,
            record=record,
            reason=reason,
            now=time.time(),
            instance_id=self._instance_id,
        )
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_sleeping(context)
            except Exception:
                logger.exception("on_sleeping failed for plugin %s", plugin.name)

    def on_shutdown(self, chat_id: str, context: ShutdownContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_shutdown(context)
            except Exception:
                logger.exception("on_shutdown failed for plugin %s", plugin.name)
            try:
                plugin.on_sleeping(
                    SleepContext(
                        chat_id=chat_id,
                        record=context.record,
                        reason="shutdown",
                        now=context.now,
                        instance_id=context.instance_id,
                    )
                )
            except Exception:
                logger.exception("on_sleeping failed for plugin %s", plugin.name)

    # ---------------------------------------------------------------- hook dispatcher

    T = TypeVar("T")

    def _apply_hook(
        self,
        chat_id: str,
        hook_name: str,
        context: T,
        can_short_circuit: bool = False,
    ) -> T | ChatResult | None:
        """Run a hook across all plugins for this chat.

        Returns the final context, or ChatResult if a gate hook short-circuits.
        """
        for plugin in self._plugins_for(chat_id):
            method = getattr(plugin, hook_name, None)
            if method is None:
                continue
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
        for plugin in self._plugins_for(chat_id):
            if context.formatted_message is None:
                context.formatted_message = formatter(context)
            method = getattr(plugin, "before_format_user_message", None)
            if method is None:
                continue
            try:
                result = method(context)
            except Exception:
                logger.exception("before_format_user_message failed for plugin %s", plugin.name)
                continue
            if result is None:
                continue
            if isinstance(result, ChatResult):
                logger.error(
                    "Plugin %s returned ChatResult from non-gate hook before_format_user_message; ignoring",
                    plugin.name,
                )
                continue
            context = result
        if context.formatted_message is None:
            context.formatted_message = formatter(context)
        return context

    def before_build_prompt(
        self,
        chat_id: str,
        context: PromptBuildContext,
    ) -> PromptBuildContext:
        result = self._apply_hook(chat_id, "before_build_prompt", context, can_short_circuit=False)
        if result is None:
            return context
        if isinstance(result, ChatResult):
            return context
        return result

    def after_prompt_built(
        self,
        chat_id: str,
        context: PromptContext,
    ) -> PromptContext:
        result = self._apply_hook(chat_id, "after_prompt_built", context, can_short_circuit=False)
        if result is None:
            return context
        if isinstance(result, ChatResult):
            return context
        return result

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
        result = self._apply_hook(chat_id, "after_engine_call", context, can_short_circuit=False)
        if result is None:
            return context
        if isinstance(result, ChatResult):
            return context
        return result

    def before_record_turn(
        self,
        chat_id: str,
        context: RecordTurnContext,
    ) -> RecordTurnContext:
        result = self._apply_hook(chat_id, "before_record_turn", context, can_short_circuit=False)
        if result is None:
            return context
        if isinstance(result, ChatResult):
            return context
        return result

    def after_turn(self, chat_id: str, turn: TurnInfo) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_turn(turn)
            except Exception:
                logger.exception("after_turn failed for plugin %s", plugin.name)

    def on_turn_error(self, chat_id: str, context: TurnErrorContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_turn_error(context)
            except Exception:
                logger.exception("on_turn_error failed for plugin %s", plugin.name)

    # ---------------------------------------------------------------- partial / dispatch / event / idle

    def on_partial(self, chat_id: str, partial: PartialTurn) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_partial(partial)
            except Exception:
                logger.exception("on_partial failed for plugin %s", plugin.name)

    def on_dispatch(self, chat_id: str, dispatch: Dispatch) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_dispatch(chat_id, dispatch)
            except Exception:
                logger.exception("on_dispatch failed for plugin %s", plugin.name)

    def on_event(
        self,
        chat_id: str,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_event(event, payload)
            except Exception:
                logger.exception("on_event failed for plugin %s", plugin.name)

    def on_idle(self, chat_id: str, context: IdleContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.on_idle(context)
            except Exception:
                logger.exception("on_idle failed for plugin %s", plugin.name)

    # ---------------------------------------------------------------- session hooks

    def before_session_archive(
        self,
        chat_id: str,
        context: SessionArchiveContext,
    ) -> SessionArchiveContext:
        result = self._apply_hook(
            chat_id, "before_session_archive", context, can_short_circuit=False
        )
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    def before_session_clear(
        self,
        chat_id: str,
        context: SessionClearContext,
    ) -> SessionClearContext:
        result = self._apply_hook(chat_id, "before_session_clear", context, can_short_circuit=False)
        if result is None or isinstance(result, ChatResult):
            return context
        return result

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
        result = self._apply_hook(chat_id, "after_session_active", context, can_short_circuit=False)
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    # ---------------------------------------------------------------- dispatch hooks

    def before_dispatch(
        self,
        chat_id: str,
        context: DispatchCreateContext,
    ) -> DispatchCreateContext:
        result = self._apply_hook(chat_id, "before_dispatch", context, can_short_circuit=False)
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    def after_dispatch(
        self,
        chat_id: str,
        context: DispatchCreateContext,
    ) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_dispatch(context)
            except Exception:
                logger.exception("after_dispatch failed for plugin %s", plugin.name)

    def before_dispatch_continue(
        self,
        chat_id: str,
        context: DispatchContinueContext,
    ) -> DispatchContinueContext:
        result = self._apply_hook(
            chat_id, "before_dispatch_continue", context, can_short_circuit=False
        )
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    def after_dispatch_continue(
        self,
        chat_id: str,
        context: DispatchCompleteContext,
    ) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_dispatch_continue(context)
            except Exception:
                logger.exception("after_dispatch_continue failed for plugin %s", plugin.name)

    # ---------------------------------------------------------------- memory hooks

    def on_chat_memory_transition(
        self,
        chat_id: str,
        context: MemoryTransitionContext,
    ) -> MemoryTransitionContext:
        result = self._apply_hook(
            chat_id,
            "on_chat_memory_transition",
            context,
            can_short_circuit=False,
        )
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    def on_persona_memory_transition(
        self,
        chat_id: str,
        context: MemoryTransitionContext,
    ) -> MemoryTransitionContext:
        result = self._apply_hook(
            chat_id,
            "on_persona_memory_transition",
            context,
            can_short_circuit=False,
        )
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    # ---------------------------------------------------------------- wake / first prompt

    def after_first_prompt_built(self, chat_id: str, context: PromptContext) -> PromptContext:
        result = self._apply_hook(
            chat_id, "after_first_prompt_built", context, can_short_circuit=False
        )
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    # ---------------------------------------------------------------- skill / mcp command hooks

    def _apply_command_hook(
        self,
        chat_id: str,
        hook_name: str,
        context: T,
    ) -> T:
        result = self._apply_hook(chat_id, hook_name, context, can_short_circuit=False)
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    def before_skill_enabled(
        self, chat_id: str, context: SkillCommandContext
    ) -> SkillCommandContext:
        return self._apply_command_hook(chat_id, "before_skill_enabled", context)

    def after_skill_enabled(self, chat_id: str, context: SkillCommandContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_skill_enabled(context)
            except Exception:
                logger.exception("after_skill_enabled failed for plugin %s", plugin.name)

    def before_skill_disabled(
        self, chat_id: str, context: SkillCommandContext
    ) -> SkillCommandContext:
        return self._apply_command_hook(chat_id, "before_skill_disabled", context)

    def after_skill_disabled(self, chat_id: str, context: SkillCommandContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_skill_disabled(context)
            except Exception:
                logger.exception("after_skill_disabled failed for plugin %s", plugin.name)

    def before_mcp_enabled(self, chat_id: str, context: McpCommandContext) -> McpCommandContext:
        return self._apply_command_hook(chat_id, "before_mcp_enabled", context)

    def after_mcp_enabled(self, chat_id: str, context: McpCommandContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_mcp_enabled(context)
            except Exception:
                logger.exception("after_mcp_enabled failed for plugin %s", plugin.name)

    def before_mcp_disabled(self, chat_id: str, context: McpCommandContext) -> McpCommandContext:
        return self._apply_command_hook(chat_id, "before_mcp_disabled", context)

    def after_mcp_disabled(self, chat_id: str, context: McpCommandContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_mcp_disabled(context)
            except Exception:
                logger.exception("after_mcp_disabled failed for plugin %s", plugin.name)

    # ---------------------------------------------------------------- retain / promote hooks

    def before_retain(self, chat_id: str, context: RetainContext) -> RetainContext:
        result = self._apply_hook(chat_id, "before_retain", context, can_short_circuit=False)
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    def after_retain(self, chat_id: str, context: RetainContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_retain(context)
            except Exception:
                logger.exception("after_retain failed for plugin %s", plugin.name)

    def before_promote(self, chat_id: str, context: PromoteContext) -> PromoteContext:
        result = self._apply_hook(chat_id, "before_promote", context, can_short_circuit=False)
        if result is None or isinstance(result, ChatResult):
            return context
        return result

    def after_promote(self, chat_id: str, context: PromoteContext) -> None:
        for plugin in self._plugins_for(chat_id):
            try:
                plugin.after_promote(context)
            except Exception:
                logger.exception("after_promote failed for plugin %s", plugin.name)
