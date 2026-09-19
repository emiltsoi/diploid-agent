"""Continuation and interruption anchors for prompt assembly.

Extracted from ``context/builder.py`` — builds the prompt's continuation slot:
interrupted-turn anchors, rehydration notices, continuation-trigger anchors,
and subagent dispatch continuation blocks.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from diploid_agent.config import Config
from diploid_agent.dispatch import Dispatch, DispatchStatus
from diploid_agent.models import PartialTurn, SessionRecord
from diploid_agent.persona_composer import _trim_to_section
from diploid_agent.plugins.contexts import RehydrationReason
from diploid_agent.text import compact_duration, human_duration


class PromptAnchors:
    """Build continuation anchors and rehydration notices for prompts."""

    def __init__(self, config: Config) -> None:
        self.config = config

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
                age = compact_duration(time.time() - partial.last_side_effect_at)
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
                    age = compact_duration(time.time() - at)
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

    @staticmethod
    def _normalize_trigger(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text).strip().lower()

    def is_continuation_message(self, user_message: str) -> bool:
        """Return True if the user message is a continuation trigger."""
        normalized = self._normalize_trigger(user_message)
        if not normalized:
            return False
        return normalized in {
            self._normalize_trigger(t) for t in self.config.engine.continuation_triggers
        }

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
        duration = human_duration(max(0.0, end - start))
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
