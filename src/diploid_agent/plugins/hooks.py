"""Fan-out and consult dispatch over the plugin registry."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from diploid_agent.dispatch import Dispatch, DispatchStatus
from diploid_agent.memory import MemoryItem
from diploid_agent.models import ChatResult, PartialTurn, SessionRecord, WakeEvent
from diploid_agent.plugins.base import SleepContext, StatePlugin, TurnInfo, WakeContext
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

if TYPE_CHECKING:
    from diploid_agent.config import McpServerConfig
    from diploid_agent.plugins.manager import PluginManager

logger = logging.getLogger(__name__)


class PluginHooks:
    """Fan-out and consult dispatch over the plugin registry."""

    def __init__(self, manager: PluginManager) -> None:
        self._manager = manager

    def _plugins_for(self, chat_id: str) -> list[StatePlugin]:
        enabled = [p for p in self._manager._plugins if self._manager._is_enabled_for(chat_id, p)]
        return [self._manager._get_or_create(chat_id, cfg) for cfg in enabled]

    def _pending_dispatches(self, chat_id: str) -> list[dict[str, Any]]:
        if self._manager._dispatch_store is None:
            return []
        return [
            d.to_dict()
            for d in self._manager._dispatch_store.list_by_chat(chat_id, DispatchStatus.PENDING)
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
            instance_id=self._manager._instance_id,
            instance_started_at=self._manager._instance_started_at,
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
        for plugin in self._manager._plugins_for(chat_id):
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
                    min(
                        max_chars,
                        self._manager._runtime.config.harness.compact_plugin_max_chars,
                    )
                    if self._manager._runtime is not None
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
        for cfg in self._manager._plugins:
            if cfg.enabled and cfg.mcp_server and not cfg.mcp_server.disabled:
                servers.append(cfg.mcp_server)
        return servers

    def default_skill_names(self) -> list[str]:
        return [p.skill for p in self._manager._plugins if p.enabled and p.skill]

    def default_mcp_names(self) -> list[str]:
        return [s.name for s in self.mcp_server_configs()]

    def durable_files(self) -> list[str]:
        files = []
        for cfg in self._manager._plugins:
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
        for cfg in self._manager._plugins:
            if cfg.name == plugin_name:
                plugin = self._manager._get_or_create(chat_id, cfg)
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
        for plugin in self._manager._plugins_for(chat_id):
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
            instance_id=self._manager._instance_id,
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
        for plugin in self._manager._plugins_for(chat_id):
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
        for plugin in self._manager._plugins_for(chat_id):
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
