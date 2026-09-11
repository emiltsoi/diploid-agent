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
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from diploid_agent.acp_client.lifecycle import AcpLifecycleLog
from diploid_agent.config import Config
from diploid_agent.dispatch import Dispatch, DispatchStatus
from diploid_agent.memory import MemoryManager, RecallResult
from diploid_agent.models import PartialTurn, SessionRecord, WakeEvent
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
    RehydrationReason,
    UserMessageContext,
)
from diploid_agent.skills import SkillManager

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Assemble first-turn and follow-up prompts from persona, memory, and slots."""

    # Hand-maintained characters-per-token table for known models.  If the model
    # is not listed, the conservative 4:1 fallback is used.
    _CHARS_PER_TOKEN: ClassVar[dict[str, float]] = {
        "swe-1-7": 3.5,
        "claude-sonnet-4-20250514": 4.0,
        "claude-sonnet-4": 4.0,
        "claude-opus-4": 4.0,
        "gpt-4o": 4.0,
        "gpt-4o-mini": 4.0,
    }

    # Plugin slots that are cheap and identity-defining; they are forced into
    # the prompt whenever the ACP session is under context pressure.
    # Promoted memory is included because it is the user-curated "me" pocket
    # that must survive compact/fresh mode.
    SOUL_SLOTS: frozenset[str] = frozenset(
        {
            "self_narrative",
            "self_state",
            "body",
            "authorship",
            "wake",
            "bridge",
            "mesh",
            "promoted",
        }
    )

    # Slots that must always be rendered, even when an allowlist/denylist is active.
    PROMPT_REQUIRED_SLOTS: frozenset[str] = frozenset(
        {
            "identity",
            "system_notice",
            "user",
            "continuation",
        }
    )

    # Slots that must survive the tiered wake-budget trimmer.
    PROTECTED_SLOTS: frozenset[str] = PROMPT_REQUIRED_SLOTS | SOUL_SLOTS

    # Rehydration reasons that create a fresh ACP child and should use the
    # compact prompt layout to avoid dumping the full conversation context.
    COMPACT_REASONS: frozenset[RehydrationReason] = frozenset(
        {
            RehydrationReason.FRESH,
            RehydrationReason.RESTART,
            RehydrationReason.TIMEOUT,
            RehydrationReason.TRANSPORT_ERROR,
        }
    )

    # Order used to render prompt slots into the final string.
    PROMPT_SLOT_ORDER: ClassVar[list[str]] = [
        "identity",
        "self_narrative",
        "system_notice",
        "memory",
        "promoted",
        "recall",
        "chat_memory",
        "persistent_memory",
        "bridge",
        "checkpoint",
        "wake",
        "working_memory",
        "body",
        "self_state",
        "authorship",
        "mesh",
        "metrics",
        "skills",
        "continuation",
        "user",
    ]

    # Trim steps for first-turn compact prompts that exceed wake_context_token_budget.
    # Each step clears the listed slots and re-renders. Earlier steps drop lower-value
    # context before higher-value context.
    WAKE_TRIM_STEPS: ClassVar[list[frozenset[str]]] = [
        frozenset({"metrics"}),
        frozenset({"memory", "chat_memory", "persistent_memory", "mesh"}),
        frozenset(
            {
                "promoted",
                "recall",
                "self_narrative",
                "working_memory",
                "body",
                "self_state",
                "authorship",
                "bridge",
                "checkpoint",
            }
        ),
    ]

    def __init__(
        self,
        config: Config,
        plugin_manager: PluginManager,
        memory_factory: Callable[[str], MemoryManager],
        skill_manager: SkillManager | None = None,
        active_skill_names: Callable[[str], set[str]] | None = None,
        context_window_fn: Callable[[str], int | None] | None = None,
        lifecycle_log: AcpLifecycleLog | None = None,
    ) -> None:
        self.config = config
        self.plugin_manager = plugin_manager
        self.memory_factory = memory_factory
        self.skill_manager = skill_manager
        self.active_skill_names = active_skill_names
        self.context_window_fn = context_window_fn
        self.lifecycle_log = lifecycle_log
        # Shared per-chat metrics store.  The harness sets this to its own dict.
        self.metrics: dict[str, dict[str, Any]] = {}
        # Per-chat cache of the last injected plugin blocks and file mtimes. This
        # lets follow-ups skip unchanged plugin slots, persona memory, and chat
        # memory without re-reading them every turn.
        self._last_blocks: dict[str, dict[tuple[str, str], str | None]] = {}
        self._last_file_mtimes: dict[str, dict[str, float]] = {}
        self._last_prompt_time: dict[str, float] = {}
        # Last turn number at which we injected a full soul, used for the turn
        # budget fallback when the model's context window is unknown.
        self._last_full_soul_turn: dict[str, int] = {}

    # ---------------------------------------------------------------- helpers

    def _reset_cache(self, chat_id: str) -> None:
        """Reset the follow-up change-detection cache for a chat."""
        self._last_blocks[chat_id] = {}
        self._last_file_mtimes[chat_id] = {}
        self._last_prompt_time[chat_id] = 0.0
        self._last_full_soul_turn[chat_id] = 0

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

    def _chars_per_token(self, model: str | None, record: SessionRecord | None = None) -> float:
        """Return a character-to-token ratio for `model`.

        Prefer live calibration from the last turn's prompt length and token
        count, then fall back to the hand-maintained table, then 4:1.
        """
        if not model:
            return 4.0

        if self.config.harness.proactive_calibration_enabled and record is not None:
            min_chars = self.config.harness.proactive_calibration_min_prompt_chars
            # The first turn of a session is the cleanest sample: the prompt we
            # sent dominates input_tokens before session history accumulates.
            # Later turns are only trusted when the ratio still lands in range —
            # on established sessions `input_tokens` includes the accumulated
            # session history, so prompt_chars / input_tokens collapses toward
            # zero and the next-prompt estimate explodes.
            for metrics in (record.first_turn_metrics, record.last_turn_metrics):
                if not metrics:
                    continue
                prompt_chars = metrics.get("prompt_chars") or 0
                input_tokens = metrics.get("input_tokens") or 0
                if prompt_chars < min_chars or input_tokens <= 0:
                    continue
                ratio = prompt_chars / input_tokens
                if 1.0 <= ratio <= 10.0:
                    return ratio

        model_lower = model.lower()
        for name, ratio in self._CHARS_PER_TOKEN.items():
            if name in model_lower:
                return ratio
        return 4.0

    @staticmethod
    def _format_silent_duration(seconds: float) -> str:
        """Return a short, human-readable sleep/duration string."""
        seconds = max(seconds, 0)
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            minutes, secs = divmod(int(seconds), 60)
            return f"{minutes}m {secs}s" if secs else f"{minutes}m"
        hours, rem = divmod(int(seconds), 3600)
        minutes, secs = divmod(rem, 60)
        if hours == 1:
            return f"1h {minutes}m" if minutes else "1h"
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"

    @staticmethod
    def _strip_recall_heading(text: str) -> str:
        """Remove the backend's default recall prefix so headings stay clean."""
        prefix = "Memory from previous turns:\n\n"
        if text.startswith(prefix):
            return text[len(prefix) :]
        return text

    def _wake_narrative(
        self,
        chat_id: str,
        event: dict[str, Any] | None,
        record: SessionRecord | None = None,
    ) -> str:
        """Render a continuity note from a lifecycle event and the prior record."""
        if event is None:
            return ""
        ev = event.get("event", "")
        reason = event.get("reason") or ""
        session_id = event.get("session_id")
        notes: list[str] = []

        if ev == "transport.restart" and reason == "mcp_change":
            notes.append("I restarted a moment ago so a new tool could load.")
        elif ev == "rehydrate.transport_restart_failure" or ("restart" in ev and "failure" in ev):
            notes.append("I had trouble restarting the ACP transport and rebuilt from files.")
        elif "restart" in ev:
            notes.append("I restarted a moment ago; the thread is intact.")
        elif ev in (
            "session.resume.success",
            "session.load.success",
            "rehydrate.resume.success",
        ):
            notes.append("I resumed the previous session; the thread continues.")
        elif ev == "rehydrate.session_alive.success":
            notes.append("The previous session was still alive; I picked up where we left off.")
        elif ev == "rehydrate.timeout" or reason == "timeout":
            notes.append("I am waking up after a hard timeout.")
        elif ev == "rehydrate.start":
            notes.append("I am waking up and rehydrating my state.")
        elif ev in ("session.new", "session.new.success", "rehydrate.new_session.success"):
            notes.append("I woke in a fresh session; earlier memory is loaded.")
        else:
            return ""

        # Add how long we were silent, using the prior record's last update.
        ts = event.get("timestamp")
        if ts and record is not None and record.updated_at:
            try:
                wake_ts = datetime.fromisoformat(ts).timestamp()
                silent = wake_ts - record.updated_at
                if silent > 1:
                    notes.append(f"I was silent for {self._format_silent_duration(silent)}.")
            except (ValueError, OSError, TypeError):
                pass

        # Mention the stop reason from the prior record if it adds useful colour.
        stop = record.last_stop_reason if record is not None else None
        if stop and stop not in ("completed", "new_session"):
            if stop == "timeout" and any("hard timeout" in note for note in notes):
                pass
            elif not any(stop in note for note in notes):
                notes.append(f"The previous turn stopped with reason: {stop}.")

        if session_id:
            notes.append(f"Session: {session_id}.")

        return " ".join(notes)

    def _last_wake_event(self, chat_id: str) -> dict[str, Any] | None:
        """Return the last wake-relevant lifecycle event for this chat."""
        if self.lifecycle_log is None:
            return None
        return self.lifecycle_log.last_wake_event_for(chat_id)

    def _compact_wake_state_line(self, record: SessionRecord | None) -> str:
        """Return a one-line wake state for compact prompts."""
        parts: list[str] = []
        if record is not None:
            stop = record.last_stop_reason or "completed"
            session = record.session_number or 0
            turn = record.turn_number or 0
            parts.append(f"Last turn: session {session}, turn {turn}, {stop}.")
        return " ".join(parts)

    def _format_system_notice(self, parts: list[str], compact: bool) -> str | None:
        """Render a system notice, one-line in compact mode or a section otherwise."""
        if not parts:
            return None
        if compact:
            return " ".join(parts)
        return "## System notice\n\n" + "\n\n".join(parts)

    def _context_window_for(self, model: str | None) -> int | None:
        """Resolve the context window for a model, if known."""
        if self.context_window_fn is not None and model:
            return self.context_window_fn(model)
        return self.config.engine.context_window

    def _context_pressure(self, record: SessionRecord | None) -> dict[str, Any]:
        """Return context-window pressure metrics for the current record.

        Uses the cumulative token count as the primary pressure signal and the
        last turn's input tokens as the secondary signal.  When the context
        window size is unknown, both percentages are zero.
        """
        context_window = self._context_window_for(record.model if record else None)
        cumulative = record.cumulative_metrics if record else {}
        last_turn = record.last_turn_metrics if record else {}

        result: dict[str, Any] = {
            "context_window": context_window,
            "cumulative_ratio": 0.0,
            "input_ratio": 0.0,
        }
        if not context_window:
            return result

        total = cumulative.get("total_tokens", 0) or 0
        input_tokens = last_turn.get("input_tokens", 0) or 0
        result["cumulative_ratio"] = round(total / context_window, 4)
        result["input_ratio"] = round(input_tokens / context_window, 4)
        return result

    def _estimate_next_prompt_tokens(
        self,
        chat_id: str,
        record: SessionRecord,
        formatted_message: str,
    ) -> dict[str, int]:
        """Estimate token footprint of the next prompt for proactive sizing.

        Uses the last turn's actual token usage as the primary signal and a
        hand-maintained characters-per-token table for the model.  Returns a
        dict with the components so callers can log them.
        """
        last_turn = record.last_turn_metrics or {}
        last_input = last_turn.get("input_tokens", 0) or 0
        last_output = last_turn.get("output_tokens", 0) or 0
        last_total = last_input + last_output

        chars_per_token = self._chars_per_token(record.model, record)

        memory_cfg = self.config.harness.memory
        short_term_estimate = int((memory_cfg.max_short_term_chars or 0) / chars_per_token)

        # Cheap fresh-soul budget: identity anchor + cheap soul slots.
        anchor_len = len(identity_anchor(self.config.persona))
        cheap_soul_estimate = (
            int(anchor_len / chars_per_token) + self.config.harness.proactive_soul_token_budget
        )

        user_estimate = int(len(formatted_message) / chars_per_token)

        buffer_factor = self.config.harness.proactive_input_buffer_factor
        buffered_turn = int(last_total * buffer_factor)

        return {
            "chars_per_token": chars_per_token,
            "last_total": last_total,
            "buffered_turn": buffered_turn,
            "soul": cheap_soul_estimate,
            "user": user_estimate,
            "short_term": short_term_estimate,
            "total": buffered_turn + cheap_soul_estimate + user_estimate + short_term_estimate,
        }

    def _wants_memory_recall(self, message: str) -> bool:
        """Return True if a fresh-mode message is asking about remembered facts."""
        lower = (message or "").lower()
        return any(
            trigger.lower() in lower for trigger in self.config.harness.memory.fresh_recall_triggers
        )

    def _load_recall_and_short_term(
        self,
        mgr: MemoryManager,
        formatted: str,
        effective_model: str,
        is_compact: bool,
        soul_mode: str,
    ) -> tuple[RecallResult, str]:
        """Load recall and short-term context for first or follow-up prompts.

        The four modes are:

        - explicit memory query in compact/fresh mode: capped full recall;
        - fresh compact mode: tiny auto recall plus a short-term tail;
        - full soul or ``recall_on_follow_up``: full long-term recall;
        - everything else: no recall.
        """
        is_fresh_memory_query = is_compact and self._wants_memory_recall(formatted)
        if is_fresh_memory_query:
            return (
                mgr.recall_context(
                    formatted,
                    model=effective_model,
                    max_chars=self.config.harness.memory.fresh_recall_max_chars,
                    max_tokens=self.config.harness.memory.fresh_recall_max_results * 150,
                ),
                "",
            )
        if soul_mode == "fresh":
            auto_recall = mgr.recall_context(
                formatted or "current threads and open state",
                model=effective_model,
                max_chars=self.config.harness.memory.fresh_auto_recall_max_chars,
                max_tokens=self.config.harness.memory.fresh_auto_recall_max_results * 150,
                include_short_term=False,
            )
            recall = RecallResult(
                text=self._strip_recall_heading(auto_recall.text),
                truncated=auto_recall.truncated,
                memory_path=auto_recall.memory_path,
                limit=auto_recall.limit,
                loaded=len(auto_recall.text),
                total=auto_recall.total,
            )
            short_term = mgr.compaction_context(model=effective_model)
            if short_term and is_compact:
                short_term = _trim_to_section(
                    short_term,
                    self.config.harness.memory.new_session_tail_max_chars,
                )
            return recall, short_term
        if soul_mode == "full" or self.config.harness.memory.recall_on_follow_up:
            return mgr.recall_context(formatted, model=effective_model), ""
        return (
            RecallResult(
                text="",
                truncated=False,
                memory_path=None,
                limit=0,
                loaded=0,
                total=0,
            ),
            "",
        )

    def _soul_mode(
        self,
        chat_id: str,
        record: SessionRecord | None,
        rehydrated: bool,
        formatted_message: str = "",
    ) -> tuple[str, bool]:
        """Decide whether to inject a small soul, full soul, or nothing new.

        Returns a tuple of (soul_mode, force_new_session) where soul_mode is
        one of "normal", "small", "full", or "fresh".  force_new_session is
        True when the context window is so full that we should start a fresh
        ACP subprocess.
        """
        if record is None:
            return "normal", False
        if rehydrated:
            return "full", False

        pressure = self._context_pressure(record)
        context_window = pressure["context_window"]
        input_ratio = pressure["input_ratio"]

        thresholds = self.config.harness
        estimated_ratio = 0.0

        # Proactive sizing: estimate the next prompt and trigger a compact fresh
        # session before the prompt overflows the ACP child context window.
        if context_window:
            estimate = self._estimate_next_prompt_tokens(chat_id, record, formatted_message)
            estimated_ratio = estimate["total"] / context_window
            logger.debug(
                "Proactive prompt estimate for %s: %s (ratio %.3f)",
                chat_id,
                estimate,
                estimated_ratio,
            )
            if estimated_ratio > thresholds.proactive_new_session_threshold:
                return "fresh", True

        # Context pressure must reflect the *current* session's occupancy.
        # `input_ratio` (last turn's input tokens / window) is that signal: the
        # ACP child reports the full prompt it consumed, including accumulated
        # history.  `cumulative_ratio` is lifetime chat usage and never resets,
        # so it must not drive fresh-session decisions.
        if input_ratio > thresholds.reinject_soul_full_threshold:
            return "fresh", True

        last_full = self._last_full_soul_turn.get(chat_id, 0)
        turn_number = record.turn_number or 0
        turns_since = turn_number - last_full

        if context_window and (
            estimated_ratio > thresholds.reinject_soul_threshold
            or input_ratio > thresholds.reinject_soul_input_threshold
        ):
            return "small", False

        if not context_window and turns_since > thresholds.reinject_soul_turns:
            return "small", False

        return "normal", False

    @staticmethod
    def _rehydration_notice(reason: RehydrationReason) -> str:
        """Return the system-notice text to explain why a session was re-created."""
        notices = {
            RehydrationReason.NONE: "",
            RehydrationReason.RESUMED: ("Resumed ACP session. The conversation history is intact."),
            RehydrationReason.STALE: (
                "This ACP session was rehydrated. Full persona memory and "
                "long-term chat memory have been re-injected into the prompt."
            ),
            RehydrationReason.TIMEOUT: (
                "The previous turn stopped due to a hard timeout. "
                "A fresh ACP session is being used."
            ),
            RehydrationReason.TRANSPORT_ERROR: (
                "The ACP transport was restarted due to an error. "
                "A fresh ACP session is being used."
            ),
            RehydrationReason.RESTART: (
                "The ACP transport was restarted. A fresh ACP session is being used."
            ),
            RehydrationReason.FRESH: (
                "Fresh ACP session for context pressure. "
                "Persona memory is compacted and long-term recall is skipped."
            ),
        }
        return notices[reason]

    def interrupted_turn_anchor(
        self,
        partial: PartialTurn | None,
        reason: RehydrationReason | None = None,
    ) -> str | None:
        """Return a compact anchor describing a turn that was interrupted mid-stream.

        The new ACP session can use this to pick up where the previous session left
        off without re-doing work that was already in progress.
        """
        if partial is None:
            return None

        current_intent = (partial.current_intent or "").strip()
        if not current_intent and partial.user_message:
            current_intent = (partial.user_message or "").strip().splitlines()[0][:120]

        if (
            not current_intent
            and not partial.last_side_effect
            and not partial.message_text
            and not partial.thought_text
            and not partial.side_effects
        ):
            return None

        header = "The current assistant turn was interrupted"
        if reason is not None and reason != RehydrationReason.NONE:
            header += f" ({reason.value})"
        header += "."

        parts: list[str] = [header]
        if current_intent:
            parts.append(f"Current intent: {current_intent}")

        last_side_effect = (partial.last_side_effect or "").strip()
        if last_side_effect:
            age = ""
            if partial.last_side_effect_at:
                age = self._format_silent_duration(time.time() - partial.last_side_effect_at)
            if age:
                parts.append(f"Last activity: {last_side_effect} ({age} ago)")
            else:
                parts.append(f"Last activity: {last_side_effect}")

        if partial.side_effects:
            lines: list[str] = []
            for eff in partial.side_effects[-8:]:
                title = str(eff.get("title") or "tool")
                status = str(eff.get("status") or "running")
                at = eff.get("at") or 0.0
                age = ""
                if at:
                    age = self._format_silent_duration(time.time() - at)
                lines.append(f"- {title} ({status})" + (f" ({age} ago)" if age else ""))
            parts.append("Tool trace before interruption:\n" + "\n".join(lines))

        message_text = (partial.message_text or "").strip()
        if message_text:
            cap = self.config.harness.interrupted_turn_message_cap
            trimmed = _trim_to_section(message_text, cap)
            if len(trimmed) < len(message_text):
                trimmed += "\n\n[... truncated ...]"
            parts.append(f"Partial reply produced so far:\n\n{trimmed}")

        thought_text = (partial.thought_text or "").strip()
        if thought_text:
            cap = self.config.harness.interrupted_turn_thought_cap
            trimmed = _trim_to_section(thought_text, cap)
            if len(trimmed) < len(thought_text):
                trimmed += "\n\n[... truncated ...]"
            parts.append(f"Partial thought so far:\n\n{trimmed}")

        return "\n\n".join(parts)

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

    def _context_budget_line(self, record: SessionRecord | None, soul_mode: str) -> str:
        """Return a one-line context budget/pressure indicator.

        Reports the *current session's* occupancy (last turn's input tokens),
        not lifetime cumulative usage — the latter only grows and is not a
        measure of how full the context window is.
        """
        pressure = self._context_pressure(record)
        context_window = pressure.get("context_window") or 0
        if not context_window:
            return f"[Context mode: {soul_mode}; window unknown]"
        last_turn = record.last_turn_metrics if record else {}
        used = int((last_turn or {}).get("input_tokens", 0) or 0)
        ratio = pressure.get("input_ratio", 0.0)
        return (
            f"[Context budget: {used}/{context_window} tokens "
            f"({ratio * 100:.1f}%); mode: {soul_mode}]"
        )

    @staticmethod
    def _render_slots(slots: dict[str, list[str]]) -> str:
        """Join slot contents into the final prompt in the canonical order."""
        parts: list[str] = []
        for slot in ContextBuilder.PROMPT_SLOT_ORDER:
            parts.extend(slots.get(slot, []))
        return "\n\n".join(parts)

    def _apply_prompt_blocks(
        self,
        slots: dict[str, list[str]],
        compact: bool,
        force_slots: set[str] | None = None,
    ) -> dict[str, list[str]]:
        """Apply per-persona allowlist/denylist/caps to prompt slots."""
        pb = self.config.harness.prompt_blocks
        if not pb.allow and not pb.deny and not pb.caps:
            return slots

        force = force_slots or set()
        allowed: set[str] = set(pb.allow) if pb.allow else set(slots.keys())
        denied = set(pb.deny) - self.PROMPT_REQUIRED_SLOTS
        allowed |= force | self.PROMPT_REQUIRED_SLOTS
        allowed -= denied

        for slot in list(slots.keys()):
            if slot not in allowed:
                slots.pop(slot, None)
                continue
            cap = pb.caps.get(slot)
            if cap is not None and slots[slot]:
                joined = "\n\n".join(slots[slot])
                if len(joined) > cap:
                    trimmed = _trim_to_section(joined, cap)
                    if trimmed:
                        slots[slot] = [trimmed]
                    else:
                        slots.pop(slot, None)
        return slots

    def _estimate_prompt_tokens(self, prompt: str, record: SessionRecord | None) -> int:
        """Return a rough token estimate for a prompt string."""
        return int(len(prompt) / self._chars_per_token(record.model if record else None, record))

    def _chat_memory_summary(self, chat_status: dict[str, Any]) -> str | None:
        """Return a one-line summary of the on-disk chat memory file."""
        total = chat_status.get("total", 0) or 0
        if not total:
            return None
        path = chat_status.get("path")
        name = path.name if path else "chat memory"
        return (
            f"{total} bytes in {name}. Say `/memory` or a memory phrase to recall the full history."
        )

    def _combined_chat_memory_block(
        self,
        recall: RecallResult,
        short_term: str,
        chat_mem: str | None,
    ) -> tuple[str | None, str | None]:
        """Build a merged `## Chat memory` block split across two prompt slots.

        The first non-empty part carries the top-level `## Chat memory` header.
        The continuation uses sub-headings (`### Recalled`, `### Recent turns`,
        `### On disk`) so trim steps can drop the on-disk slice earlier without
        losing recalled/recent content.
        """
        recall_parts: list[str] = []
        if recall.text:
            recall_parts.append(f"### Recalled\n\n{recall.text}")
        if short_term:
            recall_parts.append(f"### Recent turns\n\n{short_term}")

        chat_mem_part: str | None = None
        if chat_mem:
            chat_mem_part = f"### On disk\n\n{chat_mem}"

        if not recall_parts and not chat_mem_part:
            return None, None

        if not recall_parts:
            # On-disk only; it must carry the top-level header.
            return None, f"## Chat memory\n\n{chat_mem_part}"

        return "## Chat memory\n\n" + "\n\n".join(recall_parts), chat_mem_part

    def _wake_context_budget_line(
        self,
        record: SessionRecord | None,
        soul_mode: str,
        estimated_tokens: int,
    ) -> str | None:
        """Return a one-line wake-context budget/pressure indicator, or None if disabled."""
        budget = self.config.harness.wake_context_token_budget
        if not budget:
            return None
        ratio = min(estimated_tokens / budget, 9.99) if budget else 0.0
        return (
            f"[Wake context budget: {estimated_tokens}/{budget} tokens "
            f"({ratio * 100:.1f}%); mode: {soul_mode}]"
        )

    def _trim_slots_to_budget(
        self,
        slots: dict[str, list[str]],
        record: SessionRecord | None,
        budget: int,
    ) -> tuple[dict[str, list[str]], str]:
        """Drop lower-value slots until the prompt fits the wake budget.

        Returns the (possibly mutated) slots and the re-rendered prompt.
        """
        prompt = self._render_slots(slots)
        if self._estimate_prompt_tokens(prompt, record) <= budget:
            return slots, prompt

        for step in self.WAKE_TRIM_STEPS:
            for slot in step:
                if slot in self.PROTECTED_SLOTS:
                    continue
                if slot in slots:
                    slots[slot] = []
            prompt = self._render_slots(slots)
            if self._estimate_prompt_tokens(prompt, record) <= budget:
                return slots, prompt

        return slots, prompt

    def _skill_context(
        self,
        chat_id: str,
        skill_names: set[str] | None = None,
        compact: bool = False,
        message: str | None = None,
    ) -> str | None:
        """Build a skill index for the prompt.

        In compact/fresh mode only the active/relevant skills are shown as a
        tag list. If a user message matches a skill's triggers, the full skill
        content is also injected so the model can follow it without relying on
        an external skill loader.
        """
        if self.skill_manager is None:
            return None

        if skill_names is None and self.active_skill_names is not None:
            skill_names = self.active_skill_names(chat_id)

        active = skill_names or set()
        index = self.skill_manager.skill_index_text(
            chat_id,
            active=active,
            compact=compact,
            relevant_only=compact,
            message=message,
        )

        if not (message and index and active):
            return index

        matched = self.skill_manager.match_skills(message, chat_id, enabled=active)
        if not matched:
            return index

        content = self.skill_manager.active_skills_text(matched, chat_id)
        if not content:
            return index

        return f"{index}\n\n{content}"

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
            chat_path = chat_status.get("path")
            if chat_path:
                archive = Path(str(chat_path)).with_name(
                    f"{Path(str(chat_path)).stem}_archive{Path(str(chat_path)).suffix}"
                )
                if archive.exists():
                    lines.append(
                        f"Older chat memory sections were moved to {archive.name}; "
                        "read that file for the archived history."
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

    def _assemble_prompt(
        self,
        chat_id: str,
        record: SessionRecord | None,
        effective_model: str,
        formatted: str,
        persona: PersonaPrompt,
        recall: RecallResult,
        short_term: str,
        chat_status: dict[str, Any],
        promoted: dict[str, Any],
        chat_mem: str | None,
        build_ctx: PromptBuildContext,
        system_parts: list[str],
        skill_names: set[str] | None,
        force_slots: set[str] | None,
        is_compact: bool,
        soul_mode: str,
        is_first: bool,
        record_persona: bool = True,
        record_chat: bool = True,
        chat_memory_path: Path | None = None,
        last_prompt_time: float | None = None,
        force_new_session: bool = False,
        notice: str = "",
        metrics_compact: bool = False,
        wake_budget: int | None = None,
    ) -> PromptContext:
        """Build a ``PromptContext`` from the loaded prompt components.

        This is the shared tail of ``build_first`` and ``build_follow_up``:
        it assembles the slot dictionary, applies context-pressure formatting,
        records file mtimes, runs plugin slot fills, and renders the prompt.
        """
        if chat_id not in self._last_blocks:
            self._reset_cache(chat_id)

        slots: dict[str, list[str]] = {
            "identity": [persona.text],
            "self_narrative": [],
            "system_notice": [],
            "memory": [],
            "promoted": [],
            "recall": [],
            "chat_memory": [],
            "persistent_memory": [],
            "wake": [],
            "working_memory": [],
            "body": [],
            "self_state": [],
            "authorship": [],
            "mesh": [],
            "metrics": [],
            "skills": [],
            "continuation": [],
            "user": [formatted],
        }

        if is_compact:
            system_parts.append(self._context_budget_line(record, soul_mode))
        if system_parts:
            slots["system_notice"].append(self._format_system_notice(system_parts, is_compact))
        if persona.memory_text:
            slots["memory"].append(
                f"## Current memory ({persona.memory_path.name})\n\n{persona.memory_text}"
            )
        if promoted["text"]:
            slots["promoted"].append(f"## Promoted memory\n\n{promoted['text']}")

        recall_block, chat_block = self._combined_chat_memory_block(recall, short_term, chat_mem)
        if recall_block:
            slots["recall"].append(recall_block)
        if chat_block:
            slots["chat_memory"].append(chat_block)

        # Record mtimes for the files we just loaded so follow-ups can tell if
        # persona or chat memory has changed.
        if record_persona:
            self._record_file(chat_id, persona.memory_path)
        if record_chat and chat_memory_path is not None:
            self._record_file(chat_id, chat_memory_path)

        self.plugin_manager.fill_prompt_slots(
            chat_id,
            slots,
            is_first=is_first,
            rehydrated=build_ctx.rehydrated,
            last_blocks=self._last_blocks[chat_id],
            last_prompt_time=last_prompt_time,
            force_slots=force_slots,
            compact=is_compact,
        )

        metrics_context = self.metrics_context_for_prompt(chat_id, compact=metrics_compact)
        if metrics_context:
            slots["metrics"].append(metrics_context)

        if build_ctx.continuation_anchor:
            slots["continuation"].append(build_ctx.continuation_anchor)
        if build_ctx.interrupted_turn:
            slots["continuation"].append(build_ctx.interrupted_turn)

        skill_context = self._skill_context(chat_id, skill_names, compact=True, message=formatted)
        if skill_context:
            slots["skills"].append(skill_context)

        force_slots = force_slots or set()
        self._apply_prompt_blocks(slots, is_compact, force_slots=force_slots)

        # Remember when this prompt was built so plugins can use mtime-based
        # change detection on the next follow-up.
        self._last_prompt_time[chat_id] = time.time()

        prompt = self._render_slots(slots)

        if wake_budget is not None and is_compact:
            estimated = self._estimate_prompt_tokens(prompt, record)
            wake_budget_line = self._wake_context_budget_line(record, soul_mode, estimated)
            if wake_budget_line:
                # Add the wake budget/pressure line and re-render.
                system_parts.append(wake_budget_line)
                slots["system_notice"] = [self._format_system_notice(system_parts, is_compact)]
                prompt = self._render_slots(slots)

            # If the prompt still exceeds the wake budget, drop lower-tier slots.
            if self._estimate_prompt_tokens(prompt, record) > wake_budget:
                slots, prompt = self._trim_slots_to_budget(slots, record, wake_budget)
                estimated = self._estimate_prompt_tokens(prompt, record)
                wake_budget_line = self._wake_context_budget_line(record, soul_mode, estimated)
                if wake_budget_line and system_parts:
                    if system_parts[-1].startswith("[Wake context budget:"):
                        system_parts[-1] = wake_budget_line
                    else:
                        system_parts.append(wake_budget_line)
                    slots["system_notice"] = [self._format_system_notice(system_parts, is_compact)]
                    prompt = self._render_slots(slots)

        flags = {
            "persona_memory_exceeded": persona.memory_truncated,
            # The record's chat_memory_exceeded flag should track only the
            # on-disk chat memory file size, not the prompt's recall/context
            # truncation. Prompt truncation is reported in the system notice
            # and is not a persistent file-exceeded state.
            "chat_memory_exceeded": chat_status.get("exceeded", False),
        }

        pctx = PromptContext(
            prompt,
            notice,
            flags,
            slots,
            model=effective_model,
            force_new_session=force_new_session,
            compact=is_compact,
        )
        return self.plugin_manager.after_prompt_built(chat_id, pctx)

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
        interrupted_turn: str | None = None,
        skill_names: set[str] | None = None,
        mcp_names: list[str] | None = None,
        wake_event: WakeEvent | None = None,
        other_instance_running: bool = False,
        rehydrated: bool = False,
        rehydration_reason: RehydrationReason | None = None,
    ) -> PromptContext:
        """Build a first-turn prompt and any memory truncation notice.

        Stale-session rehydration (and other session-boundary turns) call this
        to rebuild the full prompt from persona + memory + the optional
        continuation anchor. This is not an incremental re-injection; it is a
        known cost of rehydration.
        """
        # A new ACP session starts here; reset the follow-up change cache.
        self._reset_cache(chat_id)

        resolved_reason = (
            rehydration_reason
            if rehydration_reason is not None
            else (RehydrationReason.STALE if rehydrated else RehydrationReason.NONE)
        )
        self.plugin_manager.on_waking(
            chat_id,
            record,
            time.time(),
            wake_event=wake_event,
            other_instance_running=other_instance_running,
            rehydration_reason=resolved_reason.value,
        )
        is_compact = resolved_reason in self.COMPACT_REASONS
        build_ctx = PromptBuildContext(
            chat_id=chat_id,
            record=record,
            model=model or self.config.engine.model,
            is_first=True,
            continuation_anchor=continuation_anchor,
            interrupted_turn=interrupted_turn,
            rehydrated=rehydrated,
            rehydration_reason=resolved_reason,
            compact=is_compact,
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
        tiered_compact = is_compact and self.config.harness.wake_context_token_budget > 0
        if is_compact:
            persona = PersonaPrompt(text=identity_anchor(self.config.persona))
        else:
            persona = compose_persona(self.config.persona)
        mgr = self.memory_factory(chat_id)
        is_fresh_memory_query = is_compact and self._wants_memory_recall(formatted)
        soul_mode = "fresh" if is_compact else "full"
        recall, short_term = self._load_recall_and_short_term(
            mgr, formatted, effective_model, is_compact, soul_mode
        )
        chat_status = mgr.chat_memory_status()
        promoted = mgr.promoted_memory(
            max_chars=self.config.harness.memory.max_compact_promoted_chars if is_compact else None
        )
        persona_mem_max = self.config.harness.memory.max_persona_memory_chars
        if is_compact:
            persona_mem_max = min(
                persona_mem_max,
                self.config.harness.memory.max_compact_persona_memory_chars,
            )
        pm = mgr.persona_memory(persona_mem_max)
        persona.memory_text = pm["text"]
        persona.memory_truncated = pm["truncated"]
        persona.memory_path = pm["path"]
        persona.limit = pm["limit"]
        persona.loaded = pm["loaded"]
        persona.total = pm["total"]
        notice = self.build_system_notice(persona, recall, chat_status)

        if is_compact and not is_fresh_memory_query:
            chat_mem = self._chat_memory_summary(chat_status)
        else:
            chat_mem = mgr.chat_memory_block(
                max_chars=(
                    self.config.harness.memory.max_compact_chat_memory_chars if is_compact else None
                )
            )

        rehydration_notice = self._rehydration_notice(build_ctx.rehydration_reason)
        wake_narrative = ""
        if rehydrated or (record is not None and self.lifecycle_log is not None):
            wake_narrative = self._wake_narrative(
                chat_id, self._last_wake_event(chat_id), record=record
            )

        system_parts = [p for p in [rehydration_notice, wake_narrative, notice] if p]
        force_slots = self.SOUL_SLOTS if is_compact else set()

        pctx = self._assemble_prompt(
            chat_id=chat_id,
            record=record,
            effective_model=effective_model,
            formatted=formatted,
            persona=persona,
            recall=recall,
            short_term=short_term,
            chat_status=chat_status,
            promoted=promoted,
            chat_mem=chat_mem,
            build_ctx=build_ctx,
            system_parts=system_parts,
            skill_names=skill_names,
            force_slots=force_slots,
            is_compact=is_compact,
            soul_mode=soul_mode,
            is_first=True,
            record_persona=True,
            record_chat=True,
            chat_memory_path=mgr.chat_memory_path,
            notice=notice,
            metrics_compact=is_compact,
            wake_budget=(
                self.config.harness.wake_context_token_budget
                if tiered_compact
                else None
            ),
        )
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
        interrupted_turn: str | None = None,
        skill_names: set[str] | None = None,
        rehydrated: bool = False,
        rehydration_reason: RehydrationReason | None = None,
        wake_event: WakeEvent | None = None,
        other_instance_running: bool = False,
    ) -> PromptContext:
        """Build a follow-up prompt, re-injecting the soul when context pressure rises.

        The ACP session holds the full persona and prior conversation, so follow-
        ups normally only need a linked identity anchor and the user message. When
        the context window is under pressure, the cheap identity slots (self_state,
        body, wake, mesh, self_narrative) are forced into the prompt. When the
        window is very full, the full soul (persona memory, chat memory, recall)
        is re-injected and a fresh ACP session is requested for the next turn.
        """
        formatted = self.format_user_message(
            user_message,
            reply_to,
            reply_to_is_bot,
            reply_to_message_id,
            chat_id,
        )
        soul_mode, force_new_session = self._soul_mode(chat_id, record, rehydrated, formatted)
        resolved_reason = (
            rehydration_reason
            if rehydration_reason is not None
            else (RehydrationReason.STALE if rehydrated else RehydrationReason.NONE)
        )
        if soul_mode == "fresh":
            resolved_reason = RehydrationReason.FRESH
        if wake_event is not None or rehydrated:
            self.plugin_manager.on_waking(
                chat_id,
                record,
                time.time(),
                wake_event=wake_event,
                other_instance_running=other_instance_running,
                rehydration_reason=resolved_reason.value,
            )
        is_compact = soul_mode == "fresh"
        build_ctx = PromptBuildContext(
            chat_id=chat_id,
            record=record,
            model=record.model if record else None,
            is_first=False,
            continuation_anchor=continuation_anchor,
            interrupted_turn=interrupted_turn,
            rehydrated=rehydrated,
            rehydration_reason=resolved_reason,
            compact=is_compact,
        )
        build_ctx = self.plugin_manager.before_build_prompt(chat_id, build_ctx)
        effective_model = build_ctx.model or self.config.engine.model

        anchor = identity_anchor(self.config.persona)
        mgr = self.memory_factory(chat_id)
        recall, short_term = self._load_recall_and_short_term(
            mgr, formatted, effective_model, is_compact, soul_mode
        )
        chat_status = mgr.chat_memory_status()

        # Promoted memory is a user-curated pocket that survives fresh compact mode.
        promoted = mgr.promoted_memory(
            max_chars=self.config.harness.memory.max_compact_promoted_chars if is_compact else None
        )

        # Load persona memory on full soul (rehydration) and when the file has
        # changed since the last prompt.  Fresh compact mode only loads changed
        # persona memory and caps it tightly.
        persona_memory_path = mgr.persona_memory_path
        record_persona = soul_mode == "full" or (
            soul_mode == "fresh" and self._file_changed(chat_id, persona_memory_path)
        )
        if record_persona:
            max_chars = self.config.harness.memory.max_persona_memory_chars
            if is_compact:
                max_chars = min(
                    max_chars,
                    self.config.harness.memory.max_compact_persona_memory_chars,
                )
            pm = mgr.persona_memory(max_chars)
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

        # Load on-disk chat memory on full/fresh soul and when the file has
        # changed.
        chat_memory_path = mgr.chat_memory_path
        record_chat = soul_mode in ("full", "fresh") or self._file_changed(
            chat_id, chat_memory_path
        )
        if record_chat:
            chat_mem = mgr.chat_memory_block(
                max_chars=(
                    self.config.harness.memory.max_compact_chat_memory_chars if is_compact else None
                )
            )
        else:
            chat_mem = None

        rehydration_notice = self._rehydration_notice(build_ctx.rehydration_reason)

        soul_notice = ""
        if force_new_session:
            if soul_mode == "fresh":
                wake_narrative = self._wake_narrative(
                    chat_id, self._last_wake_event(chat_id), record=record
                )
                soul_notice = (
                    "Fresh ACP session for context pressure. "
                    "Persona memory is compacted and long-term recall is skipped "
                    "unless the user asks about memory. If you need long-term facts, "
                    "call `memory_recall(query, tags)`."
                )
                if wake_narrative:
                    soul_notice += f" {wake_narrative}"
            else:
                soul_notice = (
                    "Context window is nearly full. This prompt includes the full soul "
                    "and a fresh ACP session will be started."
                )
        elif soul_mode == "small":
            soul_notice = (
                "Context pressure is high. Re-injecting the cheap soul slots "
                "(self_state, body, wake, mesh, self_narrative) to keep continuity."
            )
        elif soul_mode == "full":
            soul_notice = (
                "Re-injecting the full soul: persona memory, chat memory, recall, "
                "and identity slots."
            )

        system_parts = [p for p in [rehydration_notice, soul_notice, notice] if p]
        force_slots = self.SOUL_SLOTS if soul_mode in ("small", "full", "fresh") else None

        pctx = self._assemble_prompt(
            chat_id=chat_id,
            record=record,
            effective_model=effective_model,
            formatted=formatted,
            persona=persona,
            recall=recall,
            short_term=short_term,
            chat_status=chat_status,
            promoted=promoted,
            chat_mem=chat_mem,
            build_ctx=build_ctx,
            system_parts=system_parts,
            skill_names=skill_names,
            force_slots=force_slots,
            is_compact=is_compact,
            soul_mode=soul_mode,
            is_first=False,
            record_persona=record_persona,
            record_chat=record_chat,
            chat_memory_path=chat_memory_path,
            last_prompt_time=self._last_prompt_time.get(chat_id),
            force_new_session=force_new_session,
            notice=notice,
            metrics_compact=True,
        )
        if soul_mode == "full" and record is not None:
            self._last_full_soul_turn[chat_id] = record.turn_number
        return pctx
