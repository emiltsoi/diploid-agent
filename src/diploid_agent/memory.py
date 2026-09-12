"""Memory manager and re-exports for the conversational harness.

The concrete backend implementations live in `memory_backends`; shared data
models live in `memory_models`. This module remains the public import site.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diploid_agent.memory_backends import (
    FileMemoryBackend,
    HindsightMemoryBackend,
    MemoryBackend,
    _trim_to_last_section,
    _trim_to_section,
)
from diploid_agent.memory_models import MemoryItem, RecallResult
from diploid_agent.memory_promoted import PromotedMemory
from diploid_agent.memory_retention import TurnRetainBuffer
from diploid_agent.memory_short_term import ShortTermMemory, summary_request

logger = logging.getLogger(__name__)

__all__ = [
    "FileMemoryBackend",
    "HindsightMemoryBackend",
    "MemoryBackend",
    "MemoryItem",
    "MemoryManager",
    "RecallResult",
]


class MemoryManager:
    """Coordinates transcript, retention, summarization, and recall."""

    def __init__(
        self,
        config: Any,  # MemoryConfig
        persona: Any,  # PersonaConfig
        sessions_root: Path,
        chat_id: str,
        devin_client: Any,
        metrics: Any | None = None,
    ):
        self.memory_config = config
        self.persona = persona
        self.sessions_root = Path(sessions_root).expanduser()
        self.chat_id = chat_id
        self.devin_client = devin_client
        self.metrics = metrics

        if config.backend == "hindsight":
            hc = config.hindsight
            bank = hc.bank or persona.name
            self.backend: MemoryBackend = HindsightMemoryBackend(
                base_url=hc.base_url,
                bank=bank,
                chat_id=chat_id,
                sessions_root=self.sessions_root,
                api_key=hc.api_key,
                timeout=hc.timeout,
                max_recall_tokens=hc.max_recall_tokens,
                recall_min_scores=hc.recall_min_scores,
                prefer_observations=hc.prefer_observations,
                async_writes=hc.async_writes,
                fallback_to_file=hc.fallback_to_file,
                spool_path=hc.spool_path,
                max_chat_memory_chars=config.max_chat_memory_chars,
                metrics=metrics,
            )
        else:
            self.backend = FileMemoryBackend(
                sessions_root=self.sessions_root,
                chat_id=chat_id,
                max_chat_memory_chars=config.max_chat_memory_chars,
            )

        self._retain_buffer = TurnRetainBuffer(
            self._transcript_path.parent / "turn-retain-buffer.jsonl",
            self._retain_items,
            bundle_turns=config.retain_bundle_turns,
            chat_id=chat_id,
            persona_name=persona.name,
        )
        self._promoted = PromotedMemory(
            config,
            persona,
            self.sessions_root,
            chat_id,
            backend_fn=lambda: self.backend,
        )
        self._short_term = ShortTermMemory(
            config,
            chat_id,
            self.sessions_root,
            load_transcript=self._load_transcript,
            devin_client=devin_client,
            retain=self.retain,
        )
        self._maybe_migrate_legacy_files()
        self._retain_buffer.load()

    @property
    def _turn_buffer(self) -> list[dict[str, Any]]:
        return self._retain_buffer.entries

    def _retain_items(self, items: list[MemoryItem]) -> None:
        """Retain items via the *current* backend (tests swap ``manager.backend``)."""
        self.backend.retain(items)

    def _maybe_migrate_legacy_files(self) -> None:
        """Rename legacy transcript/memory files and move short-term summary cache into .cache/."""
        self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_transcript = self._transcript_path.with_name("transcript.jsonl")
        if legacy_transcript.exists() and not self._transcript_path.exists():
            legacy_transcript.rename(self._transcript_path)
        fb = self._file_backend
        if fb:
            legacy_memory = fb._memory_path.with_name("MEMORY.md")
            if legacy_memory.exists() and not fb._memory_path.exists():
                legacy_memory.rename(fb._memory_path)
        self._short_term.migrate_cache()
        self._short_term.prune_cache()

    @property
    def _file_backend(self) -> FileMemoryBackend | None:
        if isinstance(self.backend, FileMemoryBackend):
            return self.backend
        if isinstance(self.backend, HindsightMemoryBackend) and self.backend._fallback:
            return self.backend._fallback
        return None

    @property
    def _transcript_path(self) -> Path:
        safe = self.chat_id.replace("/", "_")
        return self.sessions_root / safe / "chat_transcript.jsonl"

    def chat_memory_status(self) -> dict[str, Any]:
        """Return the current chat memory file size and cap.

        This is used to detect whether a memory file has grown beyond its
        context budget even when the recall query did not match any entries.
        """
        path = self.chat_memory_path
        if not path:
            return {
                "path": None,
                "limit": self.memory_config.max_chat_memory_chars,
                "total": 0,
                "exceeded": False,
            }
        total = len(path.read_text()) if path.exists() else 0
        limit = self.memory_config.max_chat_memory_chars
        return {"path": path, "limit": limit, "total": total, "exceeded": total > limit}

    def chat_memory_block(self, max_chars: int | None = None) -> str | None:
        """Return the most recent on-disk chat memory, capped to `max_chars`.

        When the file is trimmed and a ``<name>_archive.md`` sibling exists,
        append a pointer so the agent always knows the older content is
        archived rather than lost — trimming keeps the tail, so a pointer
        inside the file's head would itself be trimmed away.
        """
        fb = self._file_backend
        if not fb:
            return None
        text = fb._load_memory_text()
        if not text:
            return None
        cap = max_chars or self.memory_config.max_chat_memory_chars
        if len(text) <= cap:
            return text
        trimmed = _trim_to_last_section(text, cap)
        archive = fb._memory_path.with_name(
            f"{fb._memory_path.stem}_archive{fb._memory_path.suffix}"
        )
        if archive.exists():
            trimmed += (
                f"\n\n[Older sections are archived in {archive.name} — "
                "read that file for the full history.]"
            )
        return trimmed

    @property
    def chat_memory_path(self) -> Path | None:
        """Path to the local chat memory file, if any."""
        fb = self._file_backend
        if fb:
            return fb._memory_path
        return None

    @property
    def persona_memory_path(self) -> Path:
        """Path to the persona's memory file."""
        return self._promoted.persona_path

    def persona_memory(self, max_chars: int | None = None) -> dict[str, Any]:
        """Load and optionally cap the persona's MEMORY.md for the prompt."""
        return self._promoted.persona_memory(max_chars)

    @property
    def promoted_memory_path(self) -> Path:
        """Path to the user-curated promoted memory file for this chat."""
        return self._promoted.promoted_path

    def promoted_memory(self, max_chars: int | None = None) -> dict[str, Any]:
        """Load the promoted memory pocket, always capped tightly."""
        return self._promoted.promoted_memory(max_chars)

    def _load_transcript(self) -> list[dict[str, Any]]:
        path = self._transcript_path
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _tidy_promoted_memory(self) -> None:
        self._promoted.tidy()

    def _append_transcript(
        self,
        user_message: str,
        reply: str,
        notice: str | None = None,
        system_note: str | None = None,
    ) -> None:
        path = self._transcript_path
        path.parent.mkdir(parents=True, exist_ok=True)
        assistant_content = reply if reply else (notice or "")
        with open(path, "a") as f:
            if system_note:
                f.write(json.dumps({"role": "system", "content": system_note}) + "\n")
            f.write(json.dumps({"role": "user", "content": user_message}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": assistant_content}) + "\n")

    def append_mesh_note(self, text: str) -> None:
        """Append a mesh/system note to the current chat's transcript."""
        self.backend.append_system_note(text)

    def _short_term_context(self, model: str | None = None) -> str:
        return self._short_term.context(model)

    def summarize_and_retain_recent_turns(
        self,
        n: int | None = None,
        model: str | None = None,
    ) -> str:
        """Summarize the oldest `n` short-term pairs and retain a compaction item.

        Returns the summary text and writes it to the `.cache` short-term summary
        path so `compaction_context` can load it without running the model again.
        """
        return self._short_term.summarize_and_retain(n=n, model=model)

    def compaction_context(self, model: str | None = None) -> str:
        """Return a short-term block for a fresh reset without triggering summarization.

        Loads the most recently cached short-term summary and appends the raw
        minimum fresh turns.  If no summary has been cached yet, falls back to
        the raw recent window or the fresh window.
        """
        return self._short_term.compaction_context(model)

    def recall_context(
        self,
        user_message: str,
        model: str | None = None,
        *,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        include_short_term: bool = True,
    ) -> RecallResult:
        """Return a memory block for the prompt, combining short-term + recall.

        The long-term recall is loaded first and capped, then the short-term
        transcript is appended so recent context is always visible. The
        `truncated` flag is set if the long-term recall had to be trimmed.

        Set `include_short_term=False` to get the long-term recall slice only,
        which is useful when the caller already provides a compact short-term
        summary and wants to keep the two under separate headings.
        """
        short = self._short_term_context(model) if include_short_term else ""
        cap = self.memory_config.max_chat_memory_chars
        max_recall_chars = min(
            max_chars if max_chars is not None else (self.memory_config.max_recall_chars or cap),
            cap,
        )

        query = user_message or "relevant context"
        tags: list[str] = [f"chat:{self.chat_id}"]
        recall_text = self.backend.recall(
            query,
            tags=tags,
            max_tokens=max_tokens or self.memory_config.hindsight.max_recall_tokens,
        )

        # Reserve space for the short-term context; trim the long-term recall
        # first so the most recent conversation is always visible.
        original_recall_len = len(recall_text)
        short_len = len(short) + (2 if short and recall_text else 0)
        # The long-term recall is capped both by the overall chat-memory budget
        # and by the explicit recall character cap.
        recall_cap = min(max_recall_chars, max(0, cap - short_len))
        truncated = False
        if recall_text and original_recall_len > recall_cap:
            recall_text = _trim_to_section(recall_text, recall_cap)
            truncated = True

        if short and recall_text:
            combined = f"{recall_text}\n\n{short}"
        else:
            combined = recall_text or short

        sep = 2 if short and original_recall_len > 0 else 0
        total = original_recall_len + len(short) + sep
        loaded = len(combined)

        return RecallResult(
            text=combined,
            truncated=truncated,
            memory_path=self.chat_memory_path,
            limit=cap,
            loaded=loaded,
            total=total,
        )

    def retain(
        self,
        content: str,
        tags: list[str] | None = None,
        context: str | None = None,
    ) -> None:
        """Retain a user-supplied observation in the active backend."""
        item_tags = list(tags or [])
        chat_tag = f"chat:{self.chat_id}"
        if chat_tag not in item_tags:
            item_tags.append(chat_tag)
        if "memory" not in item_tags:
            item_tags.append("memory")

        if "promoted" in item_tags or self._promoted.should_auto_promote(content, item_tags):
            if "promoted" not in item_tags:
                item_tags.append("promoted")
            self._promoted.append(content)

        item = MemoryItem(
            content=content.strip(),
            timestamp=datetime.now(UTC).isoformat(),
            document_id=f"retain-{self.chat_id}-{uuid.uuid4().hex[:12]}",
            context=context,
            metadata={
                "chat_id": self.chat_id,
                "persona": self.persona.name,
                "kind": "retained",
            },
            tags=item_tags,
        )
        self.backend.retain([item])

    def record_turn(
        self,
        user_message: str,
        reply: str,
        model: str,
        turn_number: int,
        session_number: int = 0,
        extra_items: list[MemoryItem] | None = None,
        notice: str | None = None,
        system_note: str | None = None,
        final_segment: str | None = None,
    ) -> None:
        """Append to local transcript and retain to the active backend."""
        assistant_content = reply if reply else (notice or "")
        self._append_transcript(user_message, reply, notice=notice, system_note=system_note)

        retain_content = assistant_content
        if (
            final_segment is not None
            and self.memory_config.retain_final_segment
            and len(final_segment.strip()) >= self.memory_config.retain_min_final_chars
        ):
            retain_content = final_segment.lstrip("\n")

        pair_content = f"User: {user_message}\n\nAssistant: {retain_content}"
        self._retain_buffer.append(
            pair_content,
            turn_number=turn_number,
            session_number=session_number,
            model=model,
        )

        if extra_items:
            self.backend.retain(extra_items)

        # Pre-compute the smart short-term compaction summary for the next turn,
        # so a `fresh` reset can load it without running the model synchronously.
        # Only run the model when the window is actually overflowing; otherwise
        # each turn would pay for an unnecessary summarization call.
        if self._short_term.should_precompute(turn_number, model):
            self._short_term.summarize_and_retain(
                n=self.memory_config.short_term_turns - self.memory_config.min_short_term_turns,
                model=model,
            )
            if self.metrics is not None:
                self.metrics.inc("compaction_summary_calls_total")

        if (
            self.memory_config.n_turns_summarization
            and turn_number % self.memory_config.n_turns_summarization == 0
        ):
            self._summarize(model, turn_number=turn_number, session_number=session_number)

    def _summarize(
        self,
        model: str,
        turn_number: int = 0,
        session_number: int = 0,
    ) -> None:
        """Summarize the last N turns and retain to the active backend and the file mirror."""

        transcript = self._load_transcript()
        n = self.memory_config.n_turns_summarization
        n = min(n * 2, len(transcript)) if n else 0
        if n <= 0:
            return

        recent = transcript[-n:]
        lines = []
        for entry in recent:
            role = entry.get("role", "user") or "unknown"
            lines.append(f"{role.capitalize()}: {entry.get('content', '')}")
        transcript_text = "\n\n".join(lines)

        prompt = (
            "Summarize the following conversation into a concise bullet list "
            "of facts, preferences, and decisions. Do not include pleasantries. "
            "Do not invent information.\n\n"
            f"{transcript_text}"
        )

        try:
            cwd = self.sessions_root / self.chat_id.replace("/", "_") / ".summarize"
            request = summary_request(prompt, cwd, model, self.memory_config)
            result = self.devin_client.prompt(request)
            reply = result.reply
            summary_item = MemoryItem(
                content=reply,
                timestamp=datetime.now(UTC).isoformat(),
                document_id=f"summary-{self.chat_id}-{session_number:06d}-{turn_number:06d}",
                session_number=session_number,
                metadata={
                    "chat_id": self.chat_id,
                    "persona": self.persona.name,
                    "model": model,
                    "turn": turn_number,
                    "session": session_number,
                },
                tags=[
                    "memory",
                    "summary",
                    f"chat:{self.chat_id}",
                    f"session:{session_number}",
                    f"persona:{self.persona.name}",
                ],
            )
            self.backend.retain([summary_item])
            fb = self._file_backend
            if fb and fb is not self.backend:
                fb.retain([summary_item])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summarization failed: %s", exc)

    def memory_content(self) -> str:
        """Return the raw per-chat memory for /memory display."""
        fb = self._file_backend
        if fb:
            return fb._load_memory_text() or "No memory saved for this chat yet."
        return "Memory is stored in Hindsight; use recall to inspect."

    def stats(self) -> dict[str, Any]:
        return self.backend.stats()

    def promote(self, fact: str) -> None:
        """Append a fact to the chat's curated promoted memory pocket.

        Promoted facts are always loaded in compact/fresh mode so the user can
        curate a small "me" pocket that the compactor cannot throw away.
        """
        self._promoted.append(fact)

    def promote_to_persona(self, fact: str) -> None:
        """Append a fact to the persona's MEMORY.md and, for Hindsight, index it."""
        self._promoted.promote_to_persona(fact)

    def close(self) -> None:
        """Release any resources held by the backend."""
        try:
            self._retain_buffer.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retain buffer flush on close failed: %s", exc)
        self.backend.close()
