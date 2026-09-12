"""Wake-context narration for rehydrated and compact prompts.

Extracted from ``context/builder.py`` — renders the continuity note from the
last wake-relevant lifecycle event plus the compact wake-state and wake-budget
one-liners injected into the system notice.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from diploid_agent.acp_client.lifecycle import AcpLifecycleLog
from diploid_agent.config import Config
from diploid_agent.models import SessionRecord
from diploid_agent.text import compact_duration


class WakeContext:
    """Render wake narratives and wake-budget lines for prompt assembly."""

    def __init__(
        self,
        config: Config,
        lifecycle_log_fn: Callable[[], AcpLifecycleLog | None],
    ) -> None:
        self.config = config
        # Late-bound: ContextBuilder.lifecycle_log is settable post-construction.
        self._lifecycle_log_fn = lifecycle_log_fn

    @property
    def lifecycle_log(self) -> AcpLifecycleLog | None:
        return self._lifecycle_log_fn()

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
                    notes.append(f"I was silent for {compact_duration(silent)}.")
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
