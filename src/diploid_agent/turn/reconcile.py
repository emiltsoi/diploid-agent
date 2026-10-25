"""TurnReconcile: fold externally-driven turns back into chat state.

``session/load``/``session/resume`` replays the session's full history as
``session/update`` notifications while no prompt is in flight. The ACP client
buffers that replay (``drain_session_replay``); this component compares it to
the diploid-side transcript and adopts any turns driven by another client —
e.g. a GUI front-end sharing custody of the session — so the record, the
transcript mirror, and the plugins all see one history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from diploid_agent.models import SessionRecord
from diploid_agent.turn.base import TurnComponent

logger = logging.getLogger(__name__)


@dataclass
class ExternalTurn:
    """One turn reconstructed from replayed session history."""

    user: str
    assistant: str


@dataclass
class ReconcileReport:
    """Outcome of a post-resume reconciliation."""

    external_turns: list[ExternalTurn] = field(default_factory=list)
    aligned: bool = True

    @property
    def count(self) -> int:
        return len(self.external_turns)


class TurnReconcile(TurnComponent):
    """Adopt turns driven outside diploid after an ACP session resume."""

    @staticmethod
    def _chunk_text(update: dict[str, Any]) -> str:
        content = update.get("content")
        if isinstance(content, list):
            return "".join(b.get("text", "") for b in content if isinstance(b, dict))
        if isinstance(content, dict):
            return str(content.get("text", ""))
        return ""

    @classmethod
    def reconstruct_turns(cls, updates: list[dict[str, Any]]) -> list[ExternalTurn]:
        """Group replayed updates into (user, assistant) turns.

        ``user_message_chunk`` marks the start of a turn; subsequent
        ``agent_message_chunk`` text accumulates as that turn's reply.
        Thought chunks and tool-call updates are part of the turn but carry
        no transcript text. Consecutive user chunks extend the same user
        message; a user message with no agent reply still counts as a turn
        (the external driver may have been interrupted).
        """
        turns: list[ExternalTurn] = []
        current: ExternalTurn | None = None
        for update in updates:
            kind = update.get("sessionUpdate")
            if kind == "user_message_chunk":
                if current is not None and current.assistant:
                    turns.append(current)
                    current = None
                if current is None:
                    current = ExternalTurn(user="", assistant="")
                current.user += cls._chunk_text(update)
            elif kind == "agent_message_chunk":
                if current is None:
                    current = ExternalTurn(user="", assistant="")
                current.assistant += cls._chunk_text(update)
        if current is not None:
            turns.append(current)
        return turns

    @staticmethod
    def _known_users(chat_id: str, backend: Any) -> list[str]:
        """User-side contents already in the local transcript mirror."""
        try:
            entries = backend.load_transcript()
        except Exception:  # noqa: BLE001 — reconciliation must never break resume
            return []
        return [
            str(e.get("content", ""))
            for e in entries
            if isinstance(e, dict) and e.get("role") == "user"
        ]

    def reconcile_after_resume(
        self,
        chat_id: str,
        session_id: str,
        record: SessionRecord,
    ) -> ReconcileReport | None:
        """Diff the resume replay against the transcript; adopt foreign turns.

        Returns a report (possibly empty) or None when nothing was replayed.
        """
        engine = getattr(self.runtime, "engine", None)
        drain = getattr(engine, "drain_session_replay", None)
        if drain is None:
            return None
        try:
            updates = self.runtime.call_engine_unlocked(drain, session_id)
        except Exception:
            logger.exception("session replay drain failed for %s", session_id)
            return None
        if not updates:
            return None

        replayed = self.reconstruct_turns(updates)
        known_users = self._known_users(chat_id, self.runtime._memory_manager(chat_id).backend)

        # Align on the last diploid-driven user message: everything replayed
        # after it was driven elsewhere. Fallback to the count delta when the
        # tail can't be matched (truncated content, edits).
        start = -1
        if known_users:
            last_known = known_users[-1]
            for i in range(len(replayed) - 1, -1, -1):
                if replayed[i].user == last_known:
                    start = i
                    break
        if start >= 0:
            external = replayed[start + 1 :]
            # Duplicate user text can align the tail on a later occurrence and
            # silently skip foreign turns sandwiched before it — surface any
            # prefix turns the transcript has never seen.
            known_set = set(known_users)
            unaccounted = [t.user[:120] for t in replayed[:start] if t.user not in known_set]
            if unaccounted:
                self._log_divergence(
                    chat_id,
                    session_id,
                    "unaccounted_prefix",
                    {"skipped_users": unaccounted, "aligned_at": start},
                )
        elif len(replayed) > len(known_users):
            # Tail unmatched — count-delta fallback. The replay and transcript
            # disagree on where our history ends; note it even though the
            # slice may still be right.
            self._log_divergence(
                chat_id,
                session_id,
                "tail_unmatched_fallback",
                {"replayed": len(replayed), "known_users": len(known_users)},
            )
            external = replayed[len(known_users) :]
        else:
            # Replay shows fewer/equal turns than the transcript and the tail
            # didn't match — the divergence this feature exists to detect.
            self._log_divergence(
                chat_id,
                session_id,
                "tail_unmatched",
                {"replayed": len(replayed), "known_users": len(known_users)},
            )
            return ReconcileReport(external_turns=[], aligned=not known_users)

        if not external:
            return ReconcileReport()

        backend = self.runtime._memory_manager(chat_id).backend
        for turn in external:
            try:
                backend.append_transcript(turn.user, turn.assistant, external=True)
            except Exception:
                logger.exception("failed to adopt external turn for %s", chat_id)

        record.external_turns = (record.external_turns or 0) + len(external)
        if self.runtime.lifecycle_log is not None:
            self.runtime.lifecycle_log.write(
                "reconcile.external_turns",
                chat_id=chat_id,
                session_id=session_id,
                detail={"adopted": len(external), "replayed": len(replayed)},
            )
        try:
            self.runtime._plugins.on_external_turns(
                chat_id, [{"user": t.user, "assistant": t.assistant} for t in external]
            )
        except Exception:
            logger.exception("on_external_turns hook failed for %s", chat_id)

        return ReconcileReport(external_turns=external, aligned=start >= 0)

    def _log_divergence(
        self, chat_id: str, session_id: str, reason: str, detail: dict[str, Any]
    ) -> None:
        detail = {"reason": reason, **detail}
        logger.warning("session replay divergence for %s (%s): %s", session_id, reason, detail)
        if self.runtime.lifecycle_log is not None:
            self.runtime.lifecycle_log.write(
                "reconcile.diverged",
                chat_id=chat_id,
                session_id=session_id,
                detail=detail,
            )
