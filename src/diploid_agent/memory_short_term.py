"""Short-term memory: the recent-turns window and its summary cache.

``ShortTermMemory`` turns the tail of the chat transcript into a prompt
block. When the window exceeds the character budget the oldest turns are
summarized (via a background engine prompt) into a content-addressed cache
file under ``.cache/``, so a ``fresh`` reset can reload the summary without
paying for another model call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.engine import TurnRequest
from diploid_agent.memory_backends import _trim_to_section

logger = logging.getLogger(__name__)


def summary_request(prompt: str, cwd: Path, model: str | None, config: Any) -> TurnRequest:
    """Build the background TurnRequest shared by all memory summarization."""
    return TurnRequest(
        prompt=prompt,
        cwd=cwd,
        model=model,
        soft_timeout=config.summary_soft_timeout,
        timeout=config.summary_timeout,
        background=True,
    )


class ShortTermMemory:
    """Recent-turns windowing plus the content-addressed summary cache."""

    def __init__(
        self,
        config: Any,  # MemoryConfig
        chat_id: str,
        sessions_root: Path,
        *,
        load_transcript: Callable[[], list[dict[str, Any]]],
        devin_client: Any,
        retain: Callable[..., None],
    ) -> None:
        self._config = config
        self._chat_id = chat_id
        self._sessions_root = sessions_root
        self._load_transcript = load_transcript
        self._devin_client = devin_client
        self._retain = retain

    # -------------------------------------------------------------- cache dir

    @property
    def _cache_dir(self) -> Path:
        safe = self._chat_id.replace("/", "_")
        return self._sessions_root / safe / ".cache"

    def migrate_cache(self) -> None:
        """Move legacy ``.short-term-summary-*.md`` files into ``.cache/``."""
        cache_dir = self._cache_dir
        chat_dir = cache_dir.parent
        for path in chat_dir.glob(".short-term-summary-*.md"):
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.rename(cache_dir / path.name)
            except OSError:
                pass

    def prune_cache(self) -> None:
        """Remove .cache/*.md entries older than short_term_summary_cache_days."""
        cache_dir = self._cache_dir
        if not cache_dir.exists():
            return
        max_age = self._config.short_term_summary_cache_days * 86400
        now = time.time()
        for path in cache_dir.glob("*.md"):
            try:
                if now - path.stat().st_mtime > max_age:
                    path.unlink()
            except OSError:
                pass

    def _summary_path(self, entries: list[dict[str, Any]]) -> Path:
        content = json.dumps(entries, sort_keys=True)
        h = hashlib.md5(content.encode()).hexdigest()[:12]
        return self._cache_dir / f"short-term-summary-{h}.md"

    # --------------------------------------------------------------- window

    def window(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (recent, fresh, older) transcript windows for short-term handling."""
        transcript = self._load_transcript()
        n = self._config.short_term_turns * 2
        recent = transcript[-n:] if n > 0 else transcript
        min_pairs = max(0, self._config.min_short_term_turns * 2)
        if len(recent) <= min_pairs:
            fresh = recent
            older: list[dict[str, Any]] = []
        else:
            fresh = recent[-min_pairs:]
            older = recent[:-min_pairs]
        return recent, fresh, older

    @staticmethod
    def format_recent_turns(entries: list[dict[str, Any]]) -> str:
        lines = ["Recent conversation:"]
        for entry in entries:
            role = entry.get("role", "unknown").capitalize()
            lines.append(f"{role}: {entry.get('content', '')}")
        return "\n\n".join(lines)

    def context(self, model: str | None = None) -> str:
        """Return the short-term prompt block, summarizing older turns if needed."""
        if not self._config.include_short_term:
            return ""
        recent, fresh, older = self.window()
        if not recent:
            return ""

        raw_text = self.format_recent_turns(recent)

        if self._config.short_term_strategy != "smart":
            return raw_text

        max_chars = self._config.max_short_term_chars
        if len(raw_text) <= max_chars:
            return raw_text

        fresh_text = self.format_recent_turns(fresh)

        # If even the minimum fresh window does not fit, truncate as a last resort.
        if not older:
            return (
                _trim_to_section(raw_text, max_chars)
                + "\n\n[... short-term context truncated because even the minimum "
                "fresh turns exceed the budget ...]"
            )

        # If even the minimum fresh window is larger than the short-term budget,
        # truncate it. We cannot keep any older turns at this point.
        if len(fresh_text) >= max_chars:
            return (
                _trim_to_section(fresh_text, max_chars)
                + "\n\n[... short-term context truncated because even the minimum "
                "fresh turns exceed the budget ...]"
            )

        summary = self.summarize_and_retain(
            n=self._config.short_term_turns - self._config.min_short_term_turns,
            model=model,
        )
        if not summary:
            # Fallback: just use the fresh turns if summarization failed.
            return fresh_text

        prefix = "Summary of earlier short-term turns:\n\n"
        combined = f"{prefix}{summary}\n\n{fresh_text}"
        if len(combined) <= max_chars:
            return combined

        # Trim the summary so the fresh turns remain intact.
        summary_cap = max(0, max_chars - len(fresh_text) - len(prefix) - 2)
        if summary_cap == 0:
            return fresh_text
        trimmed_summary = _trim_to_section(summary, summary_cap)
        return f"{prefix}{trimmed_summary}\n[... older turns truncated ...]\n\n{fresh_text}"

    def should_precompute(
        self,
        turn_number: int,
        model: str | None,
    ) -> bool:
        """Return True when it is worth paying for a pre-computed summary."""
        if not self._config.precompute_short_term_summary:
            return False
        if self._config.short_term_strategy != "smart":
            return False
        if turn_number < self._config.precompute_short_term_summary_min_turns:
            return False

        recent, _, _ = self.window()
        if not recent:
            return False

        raw_text = self.format_recent_turns(recent)
        return len(raw_text) > self._config.max_short_term_chars

    def summarize_and_retain(
        self,
        n: int | None = None,
        model: str | None = None,
    ) -> str:
        """Summarize the oldest `n` short-term pairs and retain a compaction item.

        Returns the summary text and writes it to the `.cache` short-term summary
        path so `compaction_context` can load it without running the model again.
        """
        if not self._config.include_short_term:
            return ""
        if self._config.short_term_strategy != "smart":
            return ""

        recent, fresh, older = self.window()
        if not older:
            return ""

        raw_text = self.format_recent_turns(recent)
        fresh_text = self.format_recent_turns(fresh)
        max_chars = self._config.max_short_term_chars
        if len(raw_text) <= max_chars:
            return ""
        if len(fresh_text) >= max_chars:
            return ""

        max_pairs = (
            n
            if n is not None
            else (self._config.short_term_turns - self._config.min_short_term_turns)
        )
        max_entries = max(0, max_pairs * 2)
        if max_entries and len(older) > max_entries:
            older = older[:max_entries]

        summary = self._load_or_summarize(older, model)
        if not summary:
            return ""

        self._retain(
            summary,
            tags=["compaction", f"chat:{self._chat_id}"],
        )
        return summary

    def compaction_context(self, model: str | None = None) -> str:
        """Return a short-term block for a fresh reset without triggering summarization.

        Loads the most recently cached short-term summary and appends the raw
        minimum fresh turns.  If no summary has been cached yet, falls back to
        the raw recent window or the fresh window.
        """
        if not self._config.include_short_term:
            return ""
        recent, fresh, older = self.window()
        if not recent:
            return ""

        raw_text = self.format_recent_turns(recent)
        fresh_text = self.format_recent_turns(fresh)
        max_chars = self._config.max_short_term_chars

        # Small enough that no summarization has happened yet.
        if len(raw_text) <= max_chars:
            return raw_text

        # Try to load a cached summary for the current older window.
        summary = ""
        if older:
            path = self._summary_path(older)
            if path.exists():
                summary = path.read_text()
            else:
                # Fall back to the most recent summary in the cache directory.
                cache_dir = path.parent
                if cache_dir.exists():
                    paths = sorted(
                        cache_dir.glob("short-term-summary-*.md"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for p in paths:
                        summary = p.read_text()
                        if summary:
                            break

        if not summary:
            return fresh_text

        prefix = "Summary of earlier short-term turns:\n\n"
        combined = f"{prefix}{summary}\n\n{fresh_text}"
        if len(combined) <= max_chars:
            return combined

        summary_cap = max(0, max_chars - len(fresh_text) - len(prefix) - 2)
        if summary_cap == 0:
            return fresh_text
        trimmed_summary = _trim_to_section(summary, summary_cap)
        return f"{prefix}{trimmed_summary}\n[... older turns truncated ...]\n\n{fresh_text}"

    def _load_or_summarize(
        self,
        entries: list[dict[str, Any]],
        model: str | None,
    ) -> str:
        """Return a cached summary of the older short-term entries, or generate one."""
        path = self._summary_path(entries)
        if path.exists():
            return path.read_text()

        lines = []
        for entry in entries:
            role = entry.get("role", "unknown").capitalize()
            lines.append(f"{role}: {entry.get('content', '')}")
        text = "\n\n".join(lines)

        prompt = (
            "Summarize the following conversation turns into a concise, dense "
            "bullet list of key facts, questions, and decisions. Do not add "
            "pleasantries or invent information.\n\n"
            f"{text}"
        )

        try:
            safe = self._chat_id.replace("/", "_")
            cwd = self._sessions_root / safe / ".summarize"
            request = summary_request(prompt, cwd, model, self._config)
            result = self._devin_client.prompt(request)
            summary = result.reply.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Short-term summary failed: %s", exc)
            summary = _trim_to_section(text, self._config.max_short_term_chars)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary)
        return summary
