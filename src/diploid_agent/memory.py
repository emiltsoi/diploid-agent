"""Memory backends for the conversational harness.

Provides a pluggable interface for retaining conversation turns and recalling
context. Two primary backends are supported:

- FileMemoryBackend: local JSONL transcript + Markdown memory file.
- HindsightMemoryBackend: HTTP-backed Hindsight server with local spool and
  file fallback.

The MemoryManager coordinates retention, summarization, recall, and prompt
formatting. It does not prune memory files mechanically; instead, it loads only
the first N characters of each memory source into a new session's prompt and
injects a `## System notice` when truncation occurs, leaving the agent free to
edit its own memory files.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from diploid_agent.engine import TurnRequest

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A single retain item."""

    content: str
    timestamp: str = ""
    document_id: str = ""
    session_number: int = 0
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
        if not self.document_id:
            self.document_id = f"mem-{uuid.uuid4().hex[:12]}"

    def to_hindsight(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "context": self.context or "",
            "metadata": {k: str(v) for k, v in (self.metadata or {}).items()},
            "document_id": self.document_id,
            "tags": self.tags,
            "update_mode": "replace",
        }


@dataclass
class RecallResult:
    """The result of recalling chat memory for a new session."""

    text: str
    truncated: bool
    memory_path: Path | None
    limit: int
    loaded: int
    total: int


class MemoryBackend(abc.ABC):
    """Pluggable memory store."""

    @abc.abstractmethod
    def health(self) -> bool:
        """Return True if the backend is currently reachable."""

    @abc.abstractmethod
    def retain(self, items: list[MemoryItem]) -> None:
        """Persist one or more items. Must not block the conversation."""

    @abc.abstractmethod
    def recall(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        max_tokens: int = 1500,
    ) -> str:
        """Return relevant context as a string, capped loosely."""

    @abc.abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return backend statistics."""

    def close(self) -> None:
        """Release any resources held by the backend."""


def _trim_to_section(text: str, limit: int) -> str:
    """Return the first `limit` characters, rounded down to a section break."""
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    last_blank = candidate.rfind("\n\n")
    if last_blank > 0:
        return text[:last_blank]
    last_newline = candidate.rfind("\n")
    if last_newline > 0:
        return text[:last_newline]
    return candidate


def _trim_to_last_section(text: str, limit: int) -> str:
    """Return the last `limit` characters, rounded to a leading section break."""
    if len(text) <= limit:
        return text
    for m in reversed(list(re.finditer(r"\n## ", text))):
        if len(text) - m.start() <= limit:
            return text[m.start():].lstrip("\n")
    return text[-limit:].lstrip("\n")


class FileMemoryBackend(MemoryBackend):
    """Local-file memory: transcript JSONL + MEMORY.md summaries.

    Recall is a simple keyword search over the transcript and the memory file.
    This is the fallback and the default. It is not semantic, but it never
    blocks and never requires a network.
    """

    def __init__(
        self,
        sessions_root: Path,
        chat_id: str,
        max_chat_memory_chars: int = 8192,
    ):
        self.sessions_root = Path(sessions_root).expanduser()
        self.chat_id = chat_id
        self.max_chat_memory_chars = max_chat_memory_chars
        self._maybe_migrate_legacy_files()

    def _maybe_migrate_legacy_files(self) -> None:
        """Rename legacy transcript/memory files to the durable chat-ledger names."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        legacy_transcript = self._session_dir / "transcript.jsonl"
        if legacy_transcript.exists() and not self._transcript_path.exists():
            legacy_transcript.rename(self._transcript_path)
        legacy_memory = self._session_dir / "MEMORY.md"
        if legacy_memory.exists() and not self._memory_path.exists():
            legacy_memory.rename(self._memory_path)

    @property
    def _session_dir(self) -> Path:
        safe = self.chat_id.replace("/", "_")
        return self.sessions_root / safe

    @property
    def _transcript_path(self) -> Path:
        return self._session_dir / "chat_transcript.jsonl"

    @property
    def _memory_path(self) -> Path:
        return self._session_dir / "chat_MEMORY.md"

    def health(self) -> bool:
        return True

    def load_transcript(self) -> list[dict[str, Any]]:
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

    def append_transcript(self, user_message: str, assistant_reply: str) -> None:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        with open(self._transcript_path, "a") as f:
            f.write(json.dumps({"role": "user", "content": user_message}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": assistant_reply}) + "\n")

    def retain(self, items: list[MemoryItem]) -> None:
        """Append memory/summary items to MEMORY.md; turns go to transcript."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        if not items:
            return
        blocks: list[str] = []
        for item in items:
            if "turn" in item.tags:
                continue
            ts = item.timestamp
            tag_str = ", ".join(item.tags)
            blocks.append(f"## {ts} ({tag_str})\n\n{item.content}\n")
        if blocks:
            with open(self._memory_path, "a") as f:
                f.write("\n".join(blocks) + "\n")

    def _load_memory_text(self) -> str:
        path = self._memory_path
        if not path.exists():
            return ""
        return path.read_text()

    def _keyword_score(self, query: str, text: str) -> float:
        words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        if not words:
            return 0.0
        text_lower = text.lower()
        return sum(1 for w in words if w in text_lower) / len(words)

    def recall(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        max_tokens: int = 1500,
    ) -> str:
        # max_tokens is a no-op for the file backend; we use the char cap
        # set on construction.
        query = query or "relevant context"

        candidates: list[tuple[str, float]] = []

        for entry in self.load_transcript():
            role = entry.get("role", "unknown").capitalize()
            text = f"{role}: {entry.get('content', '')}"
            score = self._keyword_score(query, text)
            if score > 0:
                candidates.append((text, score))

        memory_text = self._load_memory_text()
        if memory_text:
            for block in memory_text.split("\n## "):
                if block.strip():
                    score = self._keyword_score(query, block)
                    if score > 0:
                        candidates.append(("Memory:\n" + block, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        selected: list[str] = []
        total = 0
        for text, _ in candidates:
            if total + len(text) > self.max_chat_memory_chars:
                break
            selected.append(text)
            total += len(text) + 2

        if not selected:
            return ""

        return "\n\n".join(["Memory from previous turns:"] + selected)

    def stats(self) -> dict[str, Any]:
        transcript = self.load_transcript()
        memory_size = self._memory_path.stat().st_size if self._memory_path.exists() else 0
        return {
            "backend": "file",
            "transcript_turns": len(transcript) // 2,
            "memory_bytes": memory_size,
        }


class HindsightMemoryBackend(MemoryBackend):
    """Hindsight server backend with local spool and file fallback."""

    def __init__(
        self,
        base_url: str,
        bank: str,
        chat_id: str,
        sessions_root: Path,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_recall_tokens: int = 1500,
        recall_min_scores: dict[str, float] | None = None,
        prefer_observations: bool = True,
        async_writes: bool = True,
        fallback_to_file: bool = True,
        spool_path: Path | None = None,
        max_chat_memory_chars: int = 8192,
        metrics: Any | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.bank = bank
        self.chat_id = chat_id
        self.api_key = api_key
        self.timeout = timeout
        self.max_recall_tokens = max_recall_tokens
        self.recall_min_scores = recall_min_scores or {}
        self.prefer_observations = prefer_observations
        self.async_writes = async_writes
        self.fallback_to_file = fallback_to_file
        self.max_chat_memory_chars = max_chat_memory_chars
        self.metrics = metrics

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._auth_headers(),
        )

        safe = self.chat_id.replace("/", "_")
        session_dir = Path(sessions_root).expanduser() / safe
        session_dir.mkdir(parents=True, exist_ok=True)
        self._spool_path = spool_path or session_dir / "hindsight-pending-retain.jsonl"
        self._dead_letter_path = self._spool_path.with_name("hindsight-dead-letter.jsonl")
        self._spool_lock = threading.Lock()
        self._dead_letter_lock = threading.Lock()
        self._fallback = (
            FileMemoryBackend(sessions_root, chat_id, max_chat_memory_chars)
            if fallback_to_file
            else None
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["authorization"] = self.api_key
        return headers

    def _bank_url(self, *parts: str) -> str:
        return f"{self.base_url}/v1/default/{'/'.join(parts)}".rstrip("/")

    def health(self) -> bool:
        try:
            resp = httpx.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.debug("Hindsight health check failed: %s", exc)
            return False

    def _flush_spool(self) -> None:
        if not self._spool_path.exists():
            return
        if not self.health():
            return

        with self._spool_lock, open(self._spool_path, "r+") as f:
            lines = f.readlines()
            if not lines:
                return

            # Process in batches of 20.
            flushed: set[int] = set()
            for i in range(0, len(lines), 20):
                batch_lines = lines[i : i + 20]
                batch_entries: list[tuple[int, dict[str, Any]]] = []
                for offset, line in enumerate(batch_lines):
                    try:
                        batch_entries.append((i + offset, json.loads(line)))
                    except json.JSONDecodeError:
                        continue
                if not batch_entries:
                    continue

                # Validate each spooled payload and dead-letter anything that has
                # become permanently unprocessable (e.g. empty content after edits).
                valid_entries: list[tuple[int, dict[str, Any]]] = []
                for idx, payload in batch_entries:
                    ok, reason = self._validate_hindsight_item(payload)
                    if ok:
                        valid_entries.append((idx, payload))
                    else:
                        flushed.add(idx)
                        self._dead_letter(payload, reason=f"validation: {reason}")
                        logger.warning(
                            "Rejecting spooled Hindsight item %s: %s",
                            payload.get("document_id"),
                            reason,
                        )

                if not valid_entries:
                    continue

                valid_payloads = [payload for _, payload in valid_entries]
                try:
                    self._post_payloads(valid_payloads)
                    # _post_payloads returns without raising on 4xx or success.
                    # Either way the batch should not be retried.
                    flushed.update(idx for idx, _ in valid_entries)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Hindsight flush batch failed: %s", exc)
                    break

            if flushed:
                remaining = [line for i, line in enumerate(lines) if i not in flushed]
                f.seek(0)
                f.writelines(remaining)
                f.truncate()

    def _spool(self, items: list[MemoryItem]) -> None:
        with self._spool_lock, open(self._spool_path, "a") as f:
            f.writelines(json.dumps(item.to_hindsight()) + "\n" for item in items)

    def _dead_letter(self, item: dict[str, Any], *, reason: str) -> None:
        """Write an unprocessable item to a dead-letter spool for inspection."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": reason,
            "item": item,
        }
        with self._dead_letter_lock, open(self._dead_letter_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    @staticmethod
    def _validate_hindsight_item(item: dict[str, Any]) -> tuple[bool, str]:
        """Return (valid, reason_or_empty) for a Hindsight payload item."""
        content = item.get("content")
        if not content or not str(content).strip():
            return False, "empty content"
        if len(str(content)) > 100_000:
            return False, "content too large"
        return True, ""

    def _partition_items(
        self, items: list[MemoryItem]
    ) -> tuple[list[MemoryItem], list[MemoryItem]]:
        """Partition items into valid and rejected for Hindsight."""
        valid: list[MemoryItem] = []
        rejected: list[MemoryItem] = []
        for item in items:
            payload = item.to_hindsight()
            ok, reason = self._validate_hindsight_item(payload)
            if ok:
                valid.append(item)
            else:
                rejected.append(item)
                logger.warning(
                    "Rejecting Hindsight item %s: %s",
                    payload.get("document_id"),
                    reason,
                )
                self._dead_letter(payload, reason=f"validation: {reason}")
        return valid, rejected

    def _post_payloads(self, payloads: list[dict[str, Any]]) -> None:
        """POST Hindsight payloads, handling 4xx, 5xx, and network errors."""
        if not payloads:
            return

        body = {"items": payloads, "async": self.async_writes}
        try:
            resp = self._client.post(
                self._bank_url("banks", self.bank, "memories"),
                json=body,
                timeout=self.timeout,
            )
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                # Payload rejected by the server. Move the batch to the dead-letter
                # spool so it is not retried forever.
                logger.error(
                    "Hindsight rejected retain batch: %s - %s", resp.status_code, resp.text
                )
                if self.metrics is not None:
                    self.metrics.inc(
                        "hindsight_retain_failures_total", reason=f"{resp.status_code}"
                    )
                for payload in payloads:
                    self._dead_letter(payload, reason=f"{resp.status_code}")
                return
            if resp.status_code < 300 and resp.json().get("success"):
                if self.metrics is not None:
                    self.metrics.inc("hindsight_retain_total")
                return
            # 5xx, 429, or a 2xx with success=False are all transient.
            logger.warning("Hindsight retain returned %s, will retry", resp.status_code)
            raise RuntimeError(f"Hindsight retain returned {resp.status_code}")
        except httpx.HTTPStatusError as exc:
            if 500 <= exc.response.status_code < 600:
                # Transient server error; leave in spool for retry.
                logger.warning("Hindsight retain failed (5xx): %s", exc)
                if self.metrics is not None:
                    self.metrics.inc("hindsight_retain_failures_total", reason="5xx")
                raise
            if exc.response.status_code == 429:
                logger.warning("Hindsight rate limit: %s", exc)
                if self.metrics is not None:
                    self.metrics.inc("hindsight_retain_failures_total", reason="429")
                raise
            # Treat other 4xx as a permanent payload error.
            logger.error("Hindsight rejected retain batch: %s", exc)
            if self.metrics is not None:
                self.metrics.inc(
                    "hindsight_retain_failures_total",
                    reason=f"{exc.response.status_code}",
                )
            for payload in payloads:
                self._dead_letter(payload, reason=f"{exc.response.status_code}")
        except httpx.RequestError:
            logger.warning("Hindsight retain request failed (network)")
            if self.metrics is not None:
                self.metrics.inc("hindsight_retain_failures_total", reason="network")
            raise

    def retain(self, items: list[MemoryItem]) -> None:
        if not items:
            return

        # Validate and dead-letter any malformed items before spooling the valid ones.
        valid, rejected = self._partition_items(items)
        if rejected and self.metrics is not None:
            self.metrics.inc("hindsight_retain_rejected_total")

        # Always spool first so the data is durable locally, then try to flush.
        if valid:
            self._spool(valid)
        try:
            self._flush_spool()
        except Exception as exc:  # noqa: BLE001
            # Spool will be retried on the next call.
            logger.debug("Hindsight flush spool failed (will retry): %s", exc)

    def recall(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        max_tokens: int = 1500,
    ) -> str:
        if not self.health():
            if self._fallback:
                return self._fallback.recall(query, tags=tags, max_tokens=max_tokens)
            return ""

        body: dict[str, Any] = {
            "query": query,
            "max_tokens": max_tokens,
            "prefer_observations": self.prefer_observations,
            "types": ["world", "experience", "observation"],
        }
        if tags:
            body["tags"] = tags
            body["tags_match"] = "any"
        if self.recall_min_scores:
            body["min_scores"] = self.recall_min_scores

        try:
            resp = self._client.post(
                self._bank_url("banks", self.bank, "memories", "recall"),
                json=body,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            snippets = [r.get("text", "") for r in results if r.get("text")]
            text = "\n\n".join(snippets)
            if text:
                return f"Memory from previous turns:\n\n{text}"
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hindsight recall failed: %s", exc)
            if self._fallback:
                return self._fallback.recall(query, tags=tags, max_tokens=max_tokens)
            return ""

    def stats(self) -> dict[str, Any]:
        try:
            resp = self._client.get(
                self._bank_url("banks", self.bank, "stats"),
                timeout=self.timeout,
            )
            if resp.status_code < 300:
                return {"backend": "hindsight", **resp.json()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hindsight stats failed: %s", exc)
        return {"backend": "hindsight", "reachable": False}

    def close(self) -> None:
        self._client.close()


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

        self._maybe_migrate_legacy_files()

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
        self._migrate_short_term_summary_cache()
        self._prune_short_term_summary_cache()

    def _migrate_short_term_summary_cache(self) -> None:
        chat_dir = self._transcript_path.parent
        cache_dir = chat_dir / ".cache"
        for path in chat_dir.glob(".short-term-summary-*.md"):
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.rename(cache_dir / path.name)
            except OSError:
                pass

    def _prune_short_term_summary_cache(self) -> None:
        """Remove .cache/*.md entries older than short_term_summary_cache_days."""
        cache_dir = self._transcript_path.parent / ".cache"
        if not cache_dir.exists():
            return
        max_age = self.memory_config.short_term_summary_cache_days * 86400
        now = time.time()
        for path in cache_dir.glob("*.md"):
            try:
                if now - path.stat().st_mtime > max_age:
                    path.unlink()
            except OSError:
                pass

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
        """Return the most recent on-disk chat memory, capped to `max_chars`."""
        fb = self._file_backend
        if not fb:
            return None
        text = fb._load_memory_text()
        if not text:
            return None
        cap = max_chars or self.memory_config.max_chat_memory_chars
        if len(text) <= cap:
            return text
        return _trim_to_last_section(text, cap)

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
        return self.persona.profile_root / self.persona.memory_filename

    def persona_memory(self, max_chars: int | None = None) -> dict[str, Any]:
        """Load and optionally cap the persona's MEMORY.md for the prompt."""
        from diploid_agent.persona_composer import _trim_to_section

        path = self.persona_memory_path
        text = ""
        total = 0
        loaded = 0
        limit = max_chars or 0
        truncated = False

        if path.exists():
            raw = path.read_text()
            total = len(raw)
            if max_chars and total > max_chars:
                text = _trim_to_section(raw, max_chars)
                loaded = len(text)
                truncated = True
            else:
                text = raw
                loaded = total

        return {
            "text": text,
            "path": path if total > 0 else None,
            "truncated": truncated,
            "limit": limit,
            "loaded": loaded,
            "total": total,
        }

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

    def _append_transcript(self, user_message: str, reply: str, notice: str | None = None) -> None:
        path = self._transcript_path
        path.parent.mkdir(parents=True, exist_ok=True)
        assistant_content = reply if reply else (notice or "")
        with open(path, "a") as f:
            f.write(json.dumps({"role": "user", "content": user_message}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": assistant_content}) + "\n")

    def _short_term_context(self, model: str | None = None) -> str:
        if not self.memory_config.include_short_term:
            return ""
        transcript = self._load_transcript()
        n = self.memory_config.short_term_turns * 2
        recent = transcript[-n:] if n > 0 else transcript
        if not recent:
            return ""

        lines = ["Recent conversation:"]
        for entry in recent:
            role = entry.get("role", "unknown").capitalize()
            lines.append(f"{role}: {entry.get('content', '')}")
        raw_text = "\n\n".join(lines)

        if self.memory_config.short_term_strategy != "smart":
            return raw_text

        max_chars = self.memory_config.max_short_term_chars
        if len(raw_text) <= max_chars:
            return raw_text

        # Smart strategy: keep the most recent min_short_term_turns raw and
        # summarize the older part of the short-term window to fit the budget.
        min_pairs = max(0, self.memory_config.min_short_term_turns * 2)
        if len(recent) <= min_pairs:
            # Even the minimum fresh window does not fit; truncate as a last
            # resort and mark it.
            return (
                _trim_to_section(raw_text, max_chars)
                + "\n\n[... short-term context truncated because even the minimum "
                "fresh turns exceed the budget ...]"
            )

        fresh = recent[-min_pairs:]
        older = recent[:-min_pairs]

        fresh_lines = ["Recent conversation:"]
        for entry in fresh:
            role = entry.get("role", "unknown").capitalize()
            fresh_lines.append(f"{role}: {entry.get('content', '')}")
        fresh_text = "\n\n".join(fresh_lines)

        # If even the minimum fresh window is larger than the short-term budget,
        # truncate it. We cannot keep any older turns at this point.
        if len(fresh_text) >= max_chars:
            return (
                _trim_to_section(fresh_text, max_chars)
                + "\n\n[... short-term context truncated because even the minimum "
                "fresh turns exceed the budget ...]"
            )

        summary = self._load_or_summarize_short_term(older, model)
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

    def _short_term_summary_path(self, entries: list[dict[str, Any]]) -> Path:
        safe = self.chat_id.replace("/", "_")
        content = json.dumps(entries, sort_keys=True)
        h = hashlib.md5(content.encode()).hexdigest()[:12]
        return self.sessions_root / safe / ".cache" / f"short-term-summary-{h}.md"

    def _load_or_summarize_short_term(
        self,
        entries: list[dict[str, Any]],
        model: str | None,
    ) -> str:
        """Return a cached summary of the older short-term entries, or generate one."""
        path = self._short_term_summary_path(entries)
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
            safe = self.chat_id.replace("/", "_")
            cwd = self.sessions_root / safe / ".summarize"
            request = TurnRequest(
                prompt=prompt,
                cwd=cwd,
                model=model,
                soft_timeout=30.0,
            )
            result = self.devin_client.prompt(request)
            summary = result.reply.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Short-term summary failed: %s", exc)
            summary = _trim_to_section(text, self.memory_config.max_short_term_chars)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary)
        return summary

    def recall_context(self, user_message: str, model: str | None = None) -> RecallResult:
        """Return a memory block for the prompt, combining short-term + recall.

        The long-term recall is loaded first and capped, then the short-term
        transcript is appended so recent context is always visible. The
        `truncated` flag is set if the long-term recall had to be trimmed.
        """
        short = self._short_term_context(model)
        cap = self.memory_config.max_chat_memory_chars
        max_recall_chars = self.memory_config.max_recall_chars or cap

        query = user_message or "relevant context"
        tags: list[str] = [f"chat:{self.chat_id}"]
        recall_text = self.backend.recall(
            query,
            tags=tags,
            max_tokens=self.memory_config.hindsight.max_recall_tokens,
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
    ) -> None:
        """Append to local transcript and retain to the active backend."""
        assistant_content = reply if reply else (notice or "")
        self._append_transcript(user_message, reply, notice=notice)

        pair_content = f"User: {user_message}\n\nAssistant: {assistant_content}"
        item = MemoryItem(
            content=pair_content,
            timestamp=datetime.now(UTC).isoformat(),
            document_id=f"turn-{self.chat_id}-{session_number:06d}-{turn_number:06d}",
            session_number=session_number,
            metadata={
                "role": "pair",
                "chat_id": self.chat_id,
                "persona": self.persona.name,
                "model": model,
                "turn": turn_number,
                "session": session_number,
            },
            tags=[
                "turn",
                f"chat:{self.chat_id}",
                f"session:{session_number}",
                f"persona:{self.persona.name}",
            ],
        )
        self.backend.retain([item])

        if extra_items:
            self.backend.retain(extra_items)

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
        """For the file backend: summarize the last N turns and store as memory.

        Hindsight does its own consolidation, so this is a no-op for
        hindsight unless the user explicitly configures it.
        """
        if self.memory_config.backend == "hindsight":
            return

        fb = self._file_backend
        if not fb:
            return

        transcript = fb.load_transcript()
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
            request = TurnRequest(prompt=prompt, cwd=cwd, model=model)
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

    def promote_to_persona(self, fact: str) -> None:
        """Append a fact to the persona's MEMORY.md and, for Hindsight, index it.

        Memory files are not mechanically pruned. The agent has agency to edit
        them using its own file tools when the system notice says they exceed
        the prompt budget.
        """
        path = self.persona_memory_path
        path.parent.mkdir(parents=True, exist_ok=True)
        new_block = f"- {fact.strip()}\n"
        with open(path, "a") as f:
            f.write(new_block)

        if isinstance(self.backend, HindsightMemoryBackend):
            item = MemoryItem(
                content=fact.strip(),
                timestamp=datetime.now(UTC).isoformat(),
                document_id=f"promote-{self.chat_id}-{uuid.uuid4().hex[:12]}",
                metadata={
                    "chat_id": self.chat_id,
                    "persona": self.persona.name,
                    "kind": "promoted",
                },
                tags=[
                    "memory",
                    "persona",
                    "promoted",
                    f"chat:{self.chat_id}",
                    f"persona:{self.persona.name}",
                ],
            )
            self.backend.retain([item])

    def close(self) -> None:
        """Release any resources held by the backend."""
        self.backend.close()
