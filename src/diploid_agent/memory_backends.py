"""Pluggable memory backends."""

from __future__ import annotations

import abc
import json
import logging
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from diploid_agent.memory_models import MemoryItem

logger = logging.getLogger(__name__)


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

    def append_system_note(self, text: str) -> None:
        """Append a system/mesh note to the transcript with no assistant reply.

        Backends that do not support direct transcript notes may leave this as a
        no-op; FileMemoryBackend and HindsightMemoryBackend override it.
        """

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
            return text[m.start() :].lstrip("\n")
    return text[-limit:].lstrip("\n")


class FileMemoryBackend(MemoryBackend):
    """Local-file memory: transcript JSONL + MEMORY.md summaries.

    Recall is a simple keyword search over the transcript, the memory file, and
    its `chat_MEMORY_archive.md` sibling if one exists. This is the fallback and
    the default. It is not semantic, but it never blocks and never requires a
    network.
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

    @property
    def _archive_path(self) -> Path:
        return self._memory_path.with_name(
            f"{self._memory_path.stem}_archive{self._memory_path.suffix}"
        )

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

    def append_system_note(self, text: str) -> None:
        """Append a single system/mesh note to the transcript."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        with open(self._transcript_path, "a") as f:
            f.write(json.dumps({"role": "system", "content": text}) + "\n")

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

    def _load_archive_text(self) -> str:
        path = self._archive_path
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

        archive_text = self._load_archive_text()
        if archive_text:
            for block in archive_text.split("\n## "):
                if block.strip():
                    score = self._keyword_score(query, block) * 0.9
                    if score > 0:
                        candidates.append(("Memory (archive):\n" + block, score))

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
        archive_size = self._archive_path.stat().st_size if self._archive_path.exists() else 0
        return {
            "backend": "file",
            "transcript_turns": len(transcript) // 2,
            "memory_bytes": memory_size,
            "archive_bytes": archive_size,
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

    def append_system_note(self, text: str) -> None:
        """Append a note locally and queue it for hindsight if available."""
        if self._fallback is not None:
            self._fallback.append_system_note(text)

    def close(self) -> None:
        self._client.close()
