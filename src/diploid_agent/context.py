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

from diploid_agent.config import Config
from diploid_agent.dispatch import Dispatch, DispatchStatus
from diploid_agent.memory import MemoryManager, RecallResult
from diploid_agent.models import SessionRecord, WakeEvent
from diploid_agent.persona_composer import (
    PersonaPrompt,
    _trim_to_section,
    compose_persona,
    identity_anchor,
)
from diploid_agent.plugins import PluginManager
from diploid_agent.plugins.contexts import (
    PromptBuildContext,
    PromptContext,
    UserMessageContext,
)
from diploid_agent.skills import SkillManager

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
        # Per-chat cache of the last injected plugin blocks and file mtimes. This
        # lets follow-ups skip unchanged plugin slots, persona memory, and chat
        # memory without re-reading them every turn.
        self._last_blocks: dict[str, dict[tuple[str, str], str | None]] = {}
        self._last_file_mtimes: dict[str, dict[str, float]] = {}
        self._last_prompt_time: dict[str, float] = {}

    # ---------------------------------------------------------------- helpers

    def _reset_cache(self, chat_id: str) -> None:
        """Reset the follow-up change-detection cache for a chat."""
        self._last_blocks[chat_id] = {}
        self._last_file_mtimes[chat_id] = {}
        self._last_prompt_time[chat_id] = 0.0

    def _file_changed(self, chat_id: str, path: Path | None) -> bool:
        """Return True if `path` has been modified since the last prompt."""
        if path is None or not path.exists():
            return False
        last = self._last_file_mtimes.get(chat_id, {}).get(str(path))
        return last is None or path.stat().st_mtime > last

    def _record_file(self, chat_id: str, path: Path | None) -> None:
        """Record the current mtime of `path` for future change checks."""
        if path is None or not path.exists():
            return
        if chat_id not in self._last_file_mtimes:
            self._last_file_mtimes[chat_id] = {}
        self._last_file_mtimes[chat_id][str(path)] = path.stat().st_mtime

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

    @staticmethod
    def _human_duration(seconds: float) -> str:
        """Return a compact, human-readable duration."""
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes}m {secs}s"

    def _subagent_result_path(self, chat_id: str, dispatch_id: str) -> Path:
        """Return the absolute path where a subagent full result should live."""
        safe = chat_id.replace("/", "_")
        return (
            Path(self.config.harness.sessions_root).expanduser()
            / safe
            / "subagent-results"
            / f"subagent-{dispatch_id}.md"
        )

    def _dispatch_status_name(self, dispatch: Dispatch) -> str:
        """Derive a display status from the dispatch and its stop reason."""
        if dispatch.status in (DispatchStatus.TIMEOUT, DispatchStatus.CANCELLED):
            return dispatch.status.value
        if dispatch.stop_reason in ("timeout", "cancelled", "failed"):
            return dispatch.stop_reason
        if dispatch.status == DispatchStatus.PENDING:
            if dispatch.result or dispatch.finished_at is not None:
                return "completed"
            return "running"
        return dispatch.status.value

    def build_dispatch_continuation(self, dispatch: Dispatch) -> str:
        """Build a clean, structured continuation anchor for a background dispatch."""
        status = self._dispatch_status_name(dispatch)
        start = dispatch.started_at or 0.0
        end = dispatch.finished_at or time.time()
        duration = self._human_duration(max(0.0, end - start))
        summary = dispatch.summary or "(no summary)"
        result_path = dispatch.full_result_path or str(
            self._subagent_result_path(dispatch.chat_id or "unknown", dispatch.id)
        )

        lines: list[str] = [
            "## Subagent result",
            "",
            f"- **status:** {status}",
            f"- **duration:** {duration}",
            f"- **summary:** {summary}",
            f"- **full_result_path:** {result_path}",
        ]
        if dispatch.context:
            lines.append(f"- **context:** {dispatch.context}")

        if status in ("timeout", "cancelled"):
            reason = "it ran out of time" if status == "timeout" else "it was cancelled"
            lines.extend(
                [
                    "",
                    f"The subagent stopped because {reason}. The summary below is partial.",
                ]
            )

        lines.extend(["", "Please continue and present the result to the user."])
        return "\n".join(lines)

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
        rehydrated: bool = False,
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

        # A new ACP session starts here; reset the follow-up change cache.
        self._reset_cache(chat_id)

        build_ctx = PromptBuildContext(
            chat_id=chat_id,
            record=record,
            model=model or self.config.engine.model,
            is_first=True,
            continuation_anchor=continuation_anchor,
            rehydrated=rehydrated,
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
            "chat_memory": [],
            "persistent_memory": [],
            "wake": [],
            "working_memory": [],
            "body": [],
            "self_state": [],
            "mesh": [],
            "metrics": [],
            "skills": [],
            "continuation": [],
            "user": [formatted],
        }

        rehydration_notice = ""
        if build_ctx.rehydrated:
            rehydration_notice = (
                "This ACP session was rehydrated. Full persona memory and "
                "long-term chat memory have been re-injected into the prompt."
            )

        if notice or rehydration_notice:
            parts = [p for p in [rehydration_notice, notice] if p]
            slots["system_notice"].append("## System notice\n\n" + "\n\n".join(parts))
        if persona.memory_text:
            slots["memory"].append(
                f"## Current memory ({persona.memory_path.name})\n\n{persona.memory_text}"
            )
        if recall.text:
            slots["recall"].append("## Chat memory\n\n" + recall.text)

        chat_mem = self.memory_factory(chat_id).chat_memory_block()
        if chat_mem:
            slots["chat_memory"].append("## Chat memory (on disk)\n\n" + chat_mem)

        # Record mtimes for the files we just loaded so follow-ups can tell if
        # persona or chat memory has changed.
        self._record_file(chat_id, persona.memory_path)
        self._record_file(chat_id, mgr.chat_memory_path)

        self.plugin_manager.fill_prompt_slots(
            chat_id,
            slots,
            is_first=True,
            last_blocks=self._last_blocks[chat_id],
        )

        metrics_context = self.metrics_context_for_prompt(chat_id)
        if metrics_context:
            slots["metrics"].append(metrics_context)

        if build_ctx.continuation_anchor:
            slots["continuation"].append(build_ctx.continuation_anchor)

        skill_context = self._skill_context(chat_id, skill_names)
        if skill_context:
            slots["skills"].append(skill_context)

        # Remember when this prompt was built so plugins can use mtime-based
        # change detection on the next follow-up.
        self._last_prompt_time[chat_id] = time.time()

        parts: list[str] = []
        # Slots are rendered in this order.  The former `persona_state` slot
        # is split into three dedicated slots: body (sensation), self_state
        # (private mood/resume note), and mesh (external protocol).
        for slot in [
            "identity",
            "self_narrative",
            "system_notice",
            "memory",
            "recall",
            "chat_memory",
            "persistent_memory",
            "wake",
            "working_memory",
            "body",
            "self_state",
            "mesh",
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
        rehydrated: bool = False,
    ) -> PromptContext:
        """Build a follow-up prompt with a short identity anchor and changed state.

        The ACP session already holds the full persona and prior conversation, so
        follow-ups only need a linked identity anchor and the user message. Long-
        term recall, persona memory, chat memory, and plugin blocks are only
        re-injected when they have changed since the last prompt.
        """
        build_ctx = PromptBuildContext(
            chat_id=chat_id,
            record=record,
            model=record.model if record else None,
            is_first=False,
            continuation_anchor=continuation_anchor,
            rehydrated=rehydrated,
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
        anchor = identity_anchor(self.config.persona)
        mgr = self.memory_factory(chat_id)
        if self.config.harness.memory.recall_on_follow_up:
            recall = mgr.recall_context(formatted, model=effective_model)
        else:
            recall = RecallResult(
                text="",
                truncated=False,
                memory_path=None,
                limit=0,
                loaded=0,
                total=0,
            )
        chat_status = mgr.chat_memory_status()

        # Only load persona memory if the file has changed since the last prompt.
        persona_memory_path = mgr.persona_memory_path
        if self._file_changed(chat_id, persona_memory_path):
            pm = mgr.persona_memory(self.config.harness.memory.max_persona_memory_chars)
            self._record_file(chat_id, persona_memory_path)
        else:
            pm = {
                "text": "",
                "truncated": False,
                "path": persona_memory_path,
                "limit": 0,
                "loaded": 0,
                "total": 0,
            }

        persona = PersonaPrompt(
            text=anchor,
            memory_text=pm["text"],
            memory_truncated=pm["truncated"],
            memory_path=pm["path"],
            limit=pm["limit"],
            loaded=pm["loaded"],
            total=pm["total"],
        )
        notice = self.build_system_notice(persona, recall, chat_status)

        # Ensure the change-detection cache exists. A resumed ACP session may
        # start with an empty cache, in which case we re-inject changed blocks
        # once and then cache them.
        if chat_id not in self._last_blocks:
            self._reset_cache(chat_id)

        slots: dict[str, list[str]] = {
            "identity": [persona.text],
            "self_narrative": [],
            "system_notice": [],
            "memory": [],
            "recall": [],
            "chat_memory": [],
            "persistent_memory": [],
            "wake": [],
            "working_memory": [],
            "body": [],
            "self_state": [],
            "mesh": [],
            "metrics": [],
            "skills": [],
            "continuation": [],
            "user": [formatted],
        }

        rehydration_notice = ""
        if build_ctx.rehydrated:
            rehydration_notice = (
                "This ACP session was rehydrated. Full persona memory and "
                "long-term chat memory have been re-injected into the prompt."
            )

        if notice or rehydration_notice:
            parts = [p for p in [rehydration_notice, notice] if p]
            slots["system_notice"].append("## System notice\n\n" + "\n\n".join(parts))
        if persona.memory_text:
            slots["memory"].append(
                f"## Current memory ({persona.memory_path.name})\n\n{persona.memory_text}"
            )
        if recall.text:
            slots["recall"].append("## Chat memory\n\n" + recall.text)

        # Only load the on-disk chat memory if it has changed.
        chat_mem = None
        chat_memory_path = mgr.chat_memory_path
        if self._file_changed(chat_id, chat_memory_path):
            chat_mem = mgr.chat_memory_block()
            self._record_file(chat_id, chat_memory_path)
        if chat_mem:
            slots["chat_memory"].append("## Chat memory (on disk)\n\n" + chat_mem)

        self.plugin_manager.fill_prompt_slots(
            chat_id,
            slots,
            is_first=False,
            rehydrated=build_ctx.rehydrated,
            last_blocks=self._last_blocks[chat_id],
            last_prompt_time=self._last_prompt_time.get(chat_id),
        )

        self._last_prompt_time[chat_id] = time.time()

        if build_ctx.continuation_anchor:
            slots["continuation"].append(build_ctx.continuation_anchor)

        skill_context = self._skill_context(chat_id, skill_names)
        if skill_context:
            slots["skills"].append(skill_context)

        metrics_context = self.metrics_context_for_prompt(chat_id, compact=True)
        if metrics_context:
            slots["metrics"].append(metrics_context)

        parts: list[str] = []
        # Slots are rendered in this order.  The former `persona_state` slot
        # is split into three dedicated slots: body (sensation), self_state
        # (private mood/resume note), and mesh (external protocol).
        for slot in [
            "identity",
            "self_narrative",
            "system_notice",
            "memory",
            "recall",
            "chat_memory",
            "persistent_memory",
            "wake",
            "working_memory",
            "body",
            "self_state",
            "mesh",
            "metrics",
            "skills",
            "continuation",
            "user",
        ]:
            parts.extend(slots.get(slot, []))

        flags = {
            "persona_memory_exceeded": persona.memory_truncated,
            "chat_memory_exceeded": chat_status.get("exceeded", False),
        }

        prompt = "\n\n".join(parts)
        pctx = PromptContext(prompt, notice, flags, slots, model=effective_model)
        return self.plugin_manager.after_prompt_built(chat_id, pctx)
