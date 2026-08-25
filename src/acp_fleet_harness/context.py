"""Prompt context assembly for the conversational harness.

This module extracts the prompt-building logic from `ConversationHarness` into
a dedicated `ContextBuilder`.  It keeps the same ordering, formatting, and
memory-flag rules as the original harness methods.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from acp_fleet_harness.config import Config
from acp_fleet_harness.dispatch import Dispatch
from acp_fleet_harness.memory import MemoryManager, RecallResult
from acp_fleet_harness.models import SessionRecord, WakeEvent
from acp_fleet_harness.persona_composer import (
    PersonaPrompt,
    _trim_to_section,
    compose_persona,
    identity_anchor,
)
from acp_fleet_harness.plugins import PluginManager
from acp_fleet_harness.plugins.contexts import (
    PromptBuildContext,
    PromptContext,
    UserMessageContext,
)
from acp_fleet_harness.skills import SkillManager

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Assemble first-turn and follow-up prompts from persona, memory, and slots."""

    def __init__(
        self,
        config: Config,
        plugin_manager: PluginManager,
        memory_factory: Callable[[str], MemoryManager],
        skill_manager: SkillManager | None = None,
        active_skill_names: Callable[[str], set[str]] | None = None,
    ) -> None:
        self.config = config
        self.plugin_manager = plugin_manager
        self.memory_factory = memory_factory
        self.skill_manager = skill_manager
        self.active_skill_names = active_skill_names
        # Shared per-chat metrics store.  The harness sets this to its own dict.
        self.metrics: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------- helpers

    def generate_label(self, chat_id: str, user_message: str) -> str:
        """Auto-generate a short label from the first user message."""
        short = user_message.strip().replace("\n", " ")
        if len(short) > 40:
            short = short[:37] + "..."
        return short

    def metrics_context_for_prompt(self, chat_id: str, compact: bool = False) -> str | None:
        """Return a metrics notice for injection into the LLM prompt."""
        if not self.config.harness.metrics.expose_in_prompt:
            return None
        per_chat = self.metrics.get(chat_id)
        if not per_chat or not per_chat.get("cumulative"):
            return None
        cum = per_chat["cumulative"]

        if compact:
            return (
                f"[Cumulative usage: {cum['turns']} turns, "
                f"{cum['total_tokens']} tokens, "
                f"{cum['latency_seconds']:.2f}s latency]"
            )

        return (
            f"## Cumulative usage\n\n"
            f"This conversation has used {cum['turns']} turn(s), "
            f"{cum['total_tokens']} total token(s) "
            f"({cum['input_tokens']} input / {cum['output_tokens']} output), "
            f"and {cum['latency_seconds']:.2f} second(s) of model latency. "
            f"Keep the context budget in mind."
        )

    def _skill_context(self, chat_id: str, skill_names: set[str] | None = None) -> str | None:
        """Build a compact skill index for the prompt.

        Full skill content is no longer injected here.  Active skills are
        copied into the chat workspace by ``SkillManager.sync_to_chat`` so
        ``devin acp`` discovers and loads them at session start.
        """
        if self.skill_manager is None:
            return None

        if skill_names is None and self.active_skill_names is not None:
            skill_names = self.active_skill_names(chat_id)

        return self.skill_manager.skill_index_text(chat_id, active=skill_names or set())

    def build_system_notice(
        self,
        persona: PersonaPrompt,
        recall: RecallResult,
        chat_status: dict[str, Any],
    ) -> str | None:
        """Build a notice when memory is truncated for a first turn prompt."""
        lines: list[str] = []

        if persona.memory_truncated:
            lines.append(
                f"Persona memory ({persona.memory_path}) was trimmed to "
                f"{persona.loaded} of {persona.total} characters "
                f"(limit: {persona.limit})."
            )

        if recall.truncated:
            # The prompt's combined chat-memory block (recall + short-term
            # transcript) was trimmed to fit. Report recall's own numbers; do
            # not mix in the on-disk file size from chat_status and do not
            # falsely claim the MEMORY.md file itself was trimmed, because the
            # prompt block may be dominated by the short-term transcript.
            lines.append(
                "Chat memory (short-term transcript + recalled content) "
                f"was trimmed to {recall.loaded} of {recall.total} characters "
                f"(limit: {recall.limit})."
            )
        elif chat_status.get("exceeded"):
            # The on-disk memory file is over the cap, even if the recall query
            # did not match anything for this particular prompt.
            total = chat_status.get("total", 0)
            limit = chat_status.get("limit", 0)
            loaded = min(total, limit)
            lines.append(
                f"Chat memory ({chat_status.get('path')}) was trimmed to "
                f"{loaded} of {total} characters "
                f"(limit: {limit})."
            )

        if not lines:
            return None

        return (
            "The following memory files or context are larger than the context budget "
            "and were partially loaded. Older content is still saved on disk but is not "
            "shown in this prompt.\n\n" + "\n".join(f"- {line}" for line in lines) + "\n\n"
            "You may use your file tools to read the full files and, if you choose, "
            "rewrite them. Preserve user-promoted facts unless the user explicitly "
            "says otherwise. If you edit a memory file, tell the user what changed."
        )

    def trim_reply_quote_to(self, quote: str, limit: int) -> str:
        """Trim a reply-to quote to a given budget, with a truncation marker."""
        if not quote or limit <= 0:
            return ""
        if len(quote) <= limit:
            return quote
        trimmed = _trim_to_section(quote, limit - 30)
        return f"{trimmed}\n\n[... {len(quote) - len(trimmed)} characters truncated ...]"

    def trim_reply_quote(self, quote: str) -> str:
        """Trim a reply-to quote to the configured budget, with a truncation marker."""
        limit = self.config.harness.memory.max_reply_quote_chars
        return self.trim_reply_quote_to(quote, limit)

    def _telegram_message_registry_path(self, chat_id: str) -> Path:
        safe = chat_id.replace("/", "_")
        return (
            Path(self.config.harness.sessions_root).expanduser() / safe / "telegram_messages.jsonl"
        )

    def _load_telegram_message_registry(self, chat_id: str) -> dict[int, dict[str, Any]]:
        path = self._telegram_message_registry_path(chat_id)
        if not path.exists():
            return {}
        entries: dict[int, dict[str, Any]] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            message_id = entry.get("message_id")
            if message_id is not None:
                entries[message_id] = entry
        return entries

    def format_user_message(
        self,
        user_message: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        chat_id: str | None = None,
    ) -> str:
        """Wrap the user message with a labeled reply-to reference if present.

        When `chat_id` is provided, the `before_format_user_message` hook is
        invoked and plugins can modify the raw or formatted message.
        """
        if chat_id is None:
            return self._format_message_impl(
                user_message,
                reply_to=reply_to,
                reply_to_is_bot=reply_to_is_bot,
                reply_to_message_id=reply_to_message_id,
                chat_id=chat_id,
            )

        context = UserMessageContext(
            chat_id=chat_id,
            raw_message=user_message,
            formatted_message=None,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
        )

        def _formatter(ctx: UserMessageContext) -> str:
            return self._format_message_impl(
                ctx.raw_message,
                reply_to=ctx.reply_to,
                reply_to_is_bot=ctx.reply_to_is_bot,
                reply_to_message_id=ctx.reply_to_message_id,
                chat_id=ctx.chat_id,
            )

        context = self.plugin_manager.before_format_user_message(chat_id, context, _formatter)
        return context.formatted_message or context.raw_message

    def _format_message_impl(
        self,
        user_message: str,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        chat_id: str | None = None,
    ) -> str:
        """Apply reply-to quoting to the raw user message."""
        if not reply_to and not reply_to_message_id:
            return user_message

        quote = ""
        label = ""

        if reply_to_message_id and chat_id:
            registry = self._load_telegram_message_registry(chat_id)
            entry = registry.get(reply_to_message_id)
            if entry:
                preview = entry.get("preview", "")
                original_length = entry.get("original_length", len(preview))
                session_number = entry.get("session_number")
                turn_number = entry.get("turn_number")
                label = "[In reply to the assistant's earlier message"
                if session_number is not None and turn_number is not None:
                    label += f" (session {session_number}, turn {turn_number})"
                label += ":]"
                if preview:
                    quote = preview
                    if original_length > len(preview):
                        quote += (
                            f"\n\n[... {original_length - len(preview)} characters truncated ...]"
                        )

        if not quote and reply_to:
            if reply_to_is_bot is True:
                limit = self.config.harness.memory.max_bot_reply_quote_chars
                label = "[In reply to the assistant's earlier message:]"
            elif reply_to_is_bot is False:
                limit = self.config.harness.memory.max_reply_quote_chars
                label = "[In reply to your earlier message:]"
            else:
                limit = self.config.harness.memory.max_reply_quote_chars
                label = "[In reply to an earlier message:]"
            quote = self.trim_reply_quote_to(reply_to.strip(), limit)

        if not quote and not reply_to_message_id:
            return user_message

        if quote:
            return f"{label}\n{quote}\n\n[Your new message:]\n{user_message}"
        return f"{label}\n\n[Your new message:]\n{user_message}"

    def is_continuation_message(self, user_message: str) -> bool:
        """Return True if the user message is a continuation trigger."""
        normalized = re.sub(r"[^\w\s]", "", user_message).strip().lower()
        if not normalized:
            return False
        return normalized in {t.strip().lower() for t in self.config.engine.continuation_triggers}

    def continuation_anchor(self, record: SessionRecord | None, user_message: str) -> str | None:
        """Return a prompt anchor when resuming an interrupted turn."""
        if record is None or record.last_stop_reason is None:
            return None
        if not self.is_continuation_message(user_message):
            return None
        if record.last_stop_reason == "timeout":
            return (
                "The previous assistant turn was interrupted by the hard time limit "
                'and did not produce a final response. The user has sent "Continue". '
                "Resume the task from the conversation context above. If the context "
                "is insufficient, ask the user for the missing piece."
            )
        if record.last_stop_reason == "cancelled":
            return (
                "The previous assistant turn was interrupted (cancelled or soft timeout). "
                'The user has sent "Continue". Pick up the task from the partial '
                "result above and continue where you left off."
            )
        if record.last_stop_reason == "stopped":
            return (
                "The previous assistant turn was stopped by the user. "
                'The user has sent "Continue". If they want you to resume, '
                "pick up the task from the partial result above and continue where you left off."
            )
        return None

    def build_dispatch_continuation(self, dispatch: Dispatch) -> str:
        """Build a continuation anchor for a completed background dispatch."""
        parts = [
            "## System notice",
            "",
            "A background task has completed.",
        ]
        if dispatch.context:
            parts.append(f"Context: {dispatch.context}")
        parts.extend(
            [
                "",
                "Result:",
                dispatch.result or "(no result)",
                "",
                "Please continue and present the result to the user.",
            ]
        )
        return "\n".join(parts)

    # ---------------------------------------------------------------- prompt builders

    def build_first(
        self,
        chat_id: str,
        user_message: str,
        record: SessionRecord | None,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        continuation_anchor: str | None = None,
        skill_names: set[str] | None = None,
        mcp_names: list[str] | None = None,
        wake_event: WakeEvent | None = None,
        other_instance_running: bool = False,
    ) -> PromptContext:
        """Build a first-turn prompt and any memory truncation notice.

        Stale-session rehydration (and other session-boundary turns) call this
        to rebuild the full prompt from persona + memory + the optional
        continuation anchor. This is not an incremental re-injection; it is a
        known cost of rehydration.
        """
        self.plugin_manager.on_waking(
            chat_id,
            record,
            time.time(),
            wake_event=wake_event,
            other_instance_running=other_instance_running,
        )

        build_ctx = PromptBuildContext(
            chat_id=chat_id,
            record=record,
            model=model or self.config.engine.model,
            is_first=True,
            continuation_anchor=continuation_anchor,
        )
        build_ctx = self.plugin_manager.before_build_prompt(chat_id, build_ctx)
        effective_model = build_ctx.model or self.config.engine.model

        formatted = self.format_user_message(
            user_message,
            reply_to,
            reply_to_is_bot,
            reply_to_message_id,
            chat_id,
        )
        persona = compose_persona(self.config.persona)
        mgr = self.memory_factory(chat_id)
        recall = mgr.recall_context(formatted, model=effective_model)
        chat_status = mgr.chat_memory_status()
        pm = mgr.persona_memory(self.config.harness.memory.max_persona_memory_chars)
        persona.memory_text = pm["text"]
        persona.memory_truncated = pm["truncated"]
        persona.memory_path = pm["path"]
        persona.limit = pm["limit"]
        persona.loaded = pm["loaded"]
        persona.total = pm["total"]
        notice = self.build_system_notice(persona, recall, chat_status)

        slots: dict[str, list[str]] = {
            "identity": [persona.text],
            "self_narrative": [],
            "system_notice": [],
            "memory": [],
            "recall": [],
            "persistent_memory": [],
            "wake": [],
            "working_memory": [],
            "persona_state": [],
            "metrics": [],
            "skills": [],
            "continuation": [],
            "user": [formatted],
        }

        if notice:
            slots["system_notice"].append(f"## System notice\n\n{notice}")
        if persona.memory_text:
            slots["memory"].append(
                f"## Current memory ({persona.memory_path.name})\n\n{persona.memory_text}"
            )
        if recall.text:
            slots["recall"].append("## Chat memory\n\n" + recall.text)

        self.plugin_manager.fill_prompt_slots(chat_id, slots, is_first=True)

        metrics_context = self.metrics_context_for_prompt(chat_id)
        if metrics_context:
            slots["metrics"].append(metrics_context)

        if build_ctx.continuation_anchor:
            slots["continuation"].append(build_ctx.continuation_anchor)

        skill_context = self._skill_context(chat_id, skill_names)
        if skill_context:
            slots["skills"].append(skill_context)

        parts: list[str] = []
        for slot in [
            "identity",
            "self_narrative",
            "system_notice",
            "memory",
            "recall",
            "persistent_memory",
            "wake",
            "working_memory",
            "persona_state",
            "metrics",
            "skills",
            "continuation",
            "user",
        ]:
            parts.extend(slots.get(slot, []))

        flags = {
            "persona_memory_exceeded": persona.memory_truncated,
            # The record's chat_memory_exceeded flag should track only the
            # on-disk chat memory file size, not the prompt's recall/context
            # truncation. Prompt truncation is reported in the system notice
            # and is not a persistent file-exceeded state.
            "chat_memory_exceeded": chat_status.get("exceeded", False),
        }

        prompt = "\n\n".join(parts)
        pctx = PromptContext(prompt, notice, flags, slots, model=effective_model)
        pctx = self.plugin_manager.after_prompt_built(chat_id, pctx)
        return self.plugin_manager.after_first_prompt_built(chat_id, pctx)

    def build_follow_up(
        self,
        chat_id: str,
        user_message: str,
        record: SessionRecord | None,
        *,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        continuation_anchor: str | None = None,
        skill_names: set[str] | None = None,
    ) -> PromptContext:
        """Build a follow-up prompt for an existing session."""
        build_ctx = PromptBuildContext(
            chat_id=chat_id,
            record=record,
            model=record.model if record else None,
            is_first=False,
            continuation_anchor=continuation_anchor,
        )
        build_ctx = self.plugin_manager.before_build_prompt(chat_id, build_ctx)

        anchor = identity_anchor(self.config.persona)
        formatted = self.format_user_message(
            user_message,
            reply_to,
            reply_to_is_bot,
            reply_to_message_id,
            chat_id,
        )

        slots: dict[str, list[str]] = {
            "anchor": [anchor],
            "self_narrative": [],
            "working_memory": [],
            "persistent_memory": [],
            "persona_state": [],
            "continuation": [],
            "skills": [],
            "user": [formatted],
        }

        self.plugin_manager.fill_prompt_slots(chat_id, slots, is_first=False)

        if build_ctx.continuation_anchor:
            slots["continuation"].append(build_ctx.continuation_anchor)

        skill_context = self._skill_context(chat_id, skill_names)
        if skill_context:
            slots["skills"].append(skill_context)

        parts: list[str] = []
        for slot in [
            "anchor",
            "self_narrative",
            "working_memory",
            "persistent_memory",
            "persona_state",
            "continuation",
            "skills",
            "user",
        ]:
            parts.extend(slots.get(slot, []))

        prompt = "\n\n".join(parts)
        metrics_context = self.metrics_context_for_prompt(chat_id, compact=True)
        if metrics_context:
            prompt += f"\n\n{metrics_context}"
        pctx = PromptContext(prompt, None, {}, slots, model=build_ctx.model)
        return self.plugin_manager.after_prompt_built(chat_id, pctx)
