"""Base class and context objects for state plugins."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from devin_fleet_harness.config import McpServerConfig, PluginConfig
from devin_fleet_harness.dispatch import Dispatch
from devin_fleet_harness.memory import MemoryItem
from devin_fleet_harness.models import ChatResult, PartialTurn, SessionRecord, WakeEvent
from devin_fleet_harness.plugins.contexts import (
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
from devin_fleet_harness.runtime.plugin_runtime import PluginRuntime


@dataclass
class TurnInfo:
    """Summary of a completed turn for plugin lifecycle hooks."""

    chat_id: str
    session_id: str
    session_number: int
    turn_number: int
    updated_at: float
    last_stop_reason: str | None
    user_message: str
    reply: str
    notice: str | None = None
    partial_notice: str | None = None


@dataclass
class WakeContext:
    """Context passed to plugins when the harness is building a first prompt."""

    chat_id: str
    record: SessionRecord | None
    now: float
    instance_id: str
    instance_started_at: float
    previous_turn_at: float | None
    pending_dispatches: list[dict[str, Any]]
    wake_event: WakeEvent | None = None
    other_instance_running: bool = False


@dataclass
class SleepContext:
    """Context passed to plugins when the harness is shutting down a turn."""

    chat_id: str
    record: SessionRecord | None
    reason: str
    now: float
    instance_id: str


class StatePlugin(abc.ABC):
    """A per-chat state plugin that can contribute prompt blocks, MCP tools,
    skills, memory items, and lifecycle hooks.
    """

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Any,
        runtime: PluginRuntime | None = None,
    ) -> None:
        self.config = config
        self.chat_id = chat_id
        self.sessions_root = sessions_root
        self._runtime: PluginRuntime | None = runtime

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def state_file(self) -> str | None:
        return self.config.state_file

    @property
    def skill_name(self) -> str | None:
        return self.config.skill

    @property
    def prompt_slot(self) -> str:
        return self.config.prompt_slot

    @property
    def first_prompt_only(self) -> bool:
        return self.config.first_prompt_only

    @property
    def prompt_order(self) -> int:
        return self.config.prompt_order

    @property
    def max_prompt_chars(self) -> int:
        return self.config.max_prompt_chars

    def state_path(self) -> Any:
        """Return the chat-specific state file path, or None if no state_file."""
        if not self.state_file:
            return None
        chat_dir = self.sessions_root / self.chat_id.replace("/", "_")
        return chat_dir / self.state_file

    def mcp_server(self) -> McpServerConfig | None:
        return self.config.mcp_server

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        """Return a prompt block for the current chat, or None to skip."""
        return None

    def event(
        self,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> str:
        """Handle a user- or MCP-triggered event.

        HTTP callers pass typed `params` as kwargs. Telegram callers pass
        `raw_args` as the unparsed remainder of the command. The plugin parses
        whichever is present.
        """
        return "This plugin does not support events."

    def memory_items(self, since: float) -> list[MemoryItem]:
        """Return extra memory items to record for this turn."""
        return []

    # ---------------------------------------------------------------- turn hooks

    def before_turn(
        self,
        context: TurnStartContext,
    ) -> TurnStartContext | ChatResult | None:
        """Gate hook: called at the start of a turn."""
        return None

    def before_format_user_message(
        self,
        context: UserMessageContext,
    ) -> UserMessageContext | None:
        """Consult hook: called before the user message is formatted."""
        return None

    def before_build_prompt(
        self,
        context: PromptBuildContext,
    ) -> PromptBuildContext | None:
        """Consult hook: called before a prompt is built."""
        return None

    def after_prompt_built(
        self,
        context: PromptContext,
    ) -> PromptContext | None:
        """Consult hook: called after a prompt is built."""
        return None

    def before_engine_call(
        self,
        context: EngineCallContext,
    ) -> EngineCallContext | ChatResult | None:
        """Gate hook: called immediately before the engine is called."""
        return None

    def after_engine_call(
        self,
        context: EngineResultContext,
    ) -> EngineResultContext | None:
        """Consult hook: called after the engine returns a result."""
        return None

    def before_record_turn(
        self,
        context: RecordTurnContext,
    ) -> RecordTurnContext | None:
        """Consult hook: called before the turn is recorded."""
        return None

    def after_turn(self, turn: TurnInfo) -> None:
        """Notify hook: called after a turn is recorded and appended."""

    def on_turn_error(self, context: TurnErrorContext) -> None:
        """Notify hook: called when a turn raises an unhandled exception."""

    # ---------------------------------------------------------------- legacy hooks

    def on_waking(self, context: WakeContext) -> None:
        """Called before a first prompt is built."""

    def on_turn_end(self, turn: TurnInfo) -> None:
        """Called after a turn is recorded."""

    def on_sleeping(self, context: SleepContext) -> None:
        """Called when the harness is shutting down."""

    # ---------------------------------------------------------------- session hooks

    def before_session_archive(
        self, context: SessionArchiveContext
    ) -> SessionArchiveContext | None:
        """Consult hook: called before the active session is archived."""
        return None

    def before_session_clear(self, context: SessionClearContext) -> SessionClearContext | None:
        """Consult hook: called before the active session directory is cleared."""
        return None

    def before_session_start(
        self,
        context: SessionStartContext,
    ) -> SessionStartContext | ChatResult | None:
        """Gate hook: called before a new active session is started."""
        return None

    def after_session_active(self, context: SessionActiveContext) -> SessionActiveContext | None:
        """Consult/notify hook: called after a new active record is stored."""
        return None

    # ---------------------------------------------------------------- dispatch hooks

    def before_dispatch(
        self,
        context: DispatchCreateContext,
    ) -> DispatchCreateContext | None:
        """Consult hook: called when a dispatch is being created."""
        return None

    def after_dispatch(self, context: DispatchCreateContext) -> None:
        """Notify hook: called after a dispatch is registered."""

    def before_dispatch_continue(
        self,
        context: DispatchContinueContext,
    ) -> DispatchContinueContext | None:
        """Consult hook: called before a dispatch result is continued."""
        return None

    def after_dispatch_continue(self, context: DispatchCompleteContext) -> None:
        """Notify hook: called after a dispatch continuation finishes."""

    # ---------------------------------------------------------------- memory hooks

    def on_chat_memory_transition(
        self,
        context: MemoryTransitionContext,
    ) -> MemoryTransitionContext | None:
        """Consult hook: called when the chat memory budget transitions."""
        return None

    def on_persona_memory_transition(
        self,
        context: MemoryTransitionContext,
    ) -> MemoryTransitionContext | None:
        """Consult hook: called when the persona memory budget transitions."""
        return None

    # ---------------------------------------------------------------- wake / first prompt

    def after_first_prompt_built(self, context: PromptContext) -> PromptContext | None:
        """Consult hook: called after the first prompt of a session is built."""
        return None

    # ---------------------------------------------------------------- shutdown

    def on_shutdown(self, context: ShutdownContext) -> None:
        """Notify hook: called when the harness is shutting down."""

    # ---------------------------------------------------------------- partial / dispatch / event / idle

    def on_partial(self, partial: PartialTurn) -> None:
        """Called whenever partial output arrives during a streaming turn."""

    def on_dispatch(self, chat_id: str, dispatch: Dispatch) -> None:
        """Called after a dispatch is registered."""

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        """Called when a generic state event is fired."""

    def on_idle(self, context: IdleContext) -> None:
        """Called when the harness detects an idle period for a chat."""

    # ---------------------------------------------------------------- skill / mcp command hooks

    def before_skill_enabled(
        self,
        context: SkillCommandContext,
    ) -> SkillCommandContext | None:
        """Consult hook: called before a skill is enabled for this chat."""
        return None

    def after_skill_enabled(self, context: SkillCommandContext) -> None:
        """Notify hook: called after a skill is enabled for this chat."""

    def before_skill_disabled(
        self,
        context: SkillCommandContext,
    ) -> SkillCommandContext | None:
        """Consult hook: called before a skill is disabled for this chat."""
        return None

    def after_skill_disabled(self, context: SkillCommandContext) -> None:
        """Notify hook: called after a skill is disabled for this chat."""

    def before_mcp_enabled(
        self,
        context: McpCommandContext,
    ) -> McpCommandContext | None:
        """Consult hook: called before an MCP server is enabled for this chat."""
        return None

    def after_mcp_enabled(self, context: McpCommandContext) -> None:
        """Notify hook: called after an MCP server is enabled for this chat."""

    def before_mcp_disabled(
        self,
        context: McpCommandContext,
    ) -> McpCommandContext | None:
        """Consult hook: called before an MCP server is disabled for this chat."""
        return None

    def after_mcp_disabled(self, context: McpCommandContext) -> None:
        """Notify hook: called after an MCP server is disabled for this chat."""

    # ---------------------------------------------------------------- retain / promote hooks

    def before_retain(self, context: RetainContext) -> RetainContext | None:
        """Consult hook: called before an observation is retained."""
        return None

    def after_retain(self, context: RetainContext) -> None:
        """Notify hook: called after an observation is retained."""

    def before_promote(self, context: PromoteContext) -> PromoteContext | None:
        """Consult hook: called before a fact is promoted to persona memory."""
        return None

    def after_promote(self, context: PromoteContext) -> None:
        """Notify hook: called after a fact is promoted to persona memory."""
