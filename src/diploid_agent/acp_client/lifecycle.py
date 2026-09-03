"""Append-only audit log for ACP transport and session lifecycle events."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from diploid_agent.acp_client.errors import AcpTransportError

logger = logging.getLogger(__name__)


@dataclass
class AcpLifecycleEvent:
    """One entry in the ACP lifecycle audit log."""

    event: str
    timestamp: str
    chat_id: str | None = None
    session_id: str | None = None
    model: str | None = None
    reason: str | None = None
    detail: dict[str, Any] | None = None


class AcpLifecycleLog:
    """Per-harness append-only JSONL log for ACP lifecycle events.

    Writes are lock-protected. Events should be small and fire-and-forget so
    they do not block the ACP control path.
    """

    def __init__(self, path: Path, max_lines: int = 10000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._max_lines = max(1, max_lines)

    def _read_tail(self, n: int) -> list[str]:
        """Return the last ``n`` non-empty lines without loading the whole file."""
        lines: deque[str] = deque(maxlen=n)
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(line)
        return list(lines)

    def _trim_if_needed(self) -> None:
        """Keep the log at or below ``_max_lines`` by trimming from the head."""
        if not self.path.exists():
            return
        with self._lock:
            # Reading max_lines + 1 lets us detect whether any trimming is needed.
            lines = self._read_tail(self._max_lines + 1)
            if len(lines) <= self._max_lines:
                return
            trimmed = lines[-self._max_lines :]
            with self.path.open("w", encoding="utf-8") as f:
                f.writelines(trimmed)

    def write(
        self,
        event: str,
        *,
        chat_id: str | None = None,
        session_id: str | None = None,
        model: str | None = None,
        reason: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append a lifecycle event to the log and trim if it grew too large."""
        entry = AcpLifecycleEvent(
            event=event,
            timestamp=datetime.now(UTC).isoformat(),
            chat_id=chat_id,
            session_id=session_id,
            model=model,
            reason=reason,
            detail=detail,
        )
        line = json.dumps(asdict(entry), default=str, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line)
        self._trim_if_needed()

    def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return the most recent log entries without loading the whole file."""
        if not self.path.exists():
            return []
        with self._lock:
            lines = self._read_tail(limit)
        return [json.loads(line) for line in lines if line.strip()]

    def recent_events_for(
        self,
        chat_id: str,
        event_types: list[str] | None = None,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Return the most recent log entries for a specific chat.

        Filters by ``chat_id`` and, optionally, by ``event`` type.  Because the
        log is stored as plain JSONL, this scans the tail of the file.
        """
        if not self.path.exists():
            return []
        types = frozenset(event_types) if event_types is not None else None
        # Read a larger tail so the filtered result is likely to contain enough
        # entries; if not, callers get fewer results.
        with self._lock:
            lines = self._read_tail(limit * 10)
        events = [json.loads(line) for line in lines if line.strip()]
        events.reverse()
        results: list[dict[str, Any]] = []
        for event in events:
            if event.get("chat_id") != chat_id:
                continue
            if types is not None and event.get("event") not in types:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        results.reverse()
        return results


class AcpRestartHistory:
    """Persistent, per-harness restart attempt history for ACP transport backoff.

    Stores monotonic timestamps and reasons in a JSONL file so the backoff
    counter survives process restarts.  In-memory fallback when no path is
    provided (tests).  Older files that only contain a timestamp are loaded as
    ``transport_error``.
    """

    def __init__(self, path: Path | None, window: float) -> None:
        self.path = Path(path) if path is not None else None
        self._window = max(1.0, window)
        self._lock = threading.RLock()
        self._history: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load existing restart timestamps from disk and prune stale ones."""
        if self.path is None or not self.path.exists():
            return
        try:
            with self._lock, self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    ts = float(data.get("timestamp", 0))
                    if ts > 0:
                        self._history.append(
                            {
                                "timestamp": ts,
                                "reason": data.get("reason") or "transport_error",
                            }
                        )
            self._prune()
            self._prune_file()
        except (OSError, ValueError, TypeError):
            logger.exception("Failed to load ACP restart history from %s", self.path)

    def _prune(self, now: float | None = None) -> None:
        """Drop timestamps outside the configured backoff window."""
        now = now if now is not None else time.time()
        cutoff = now - self._window
        self._history = [entry for entry in self._history if entry["timestamp"] > cutoff]

    def _prune_file(self) -> None:
        """Rewrite the on-disk history so it matches the in-memory pruned list."""
        if self.path is None:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                for entry in self._history:
                    f.write(json.dumps(entry, default=str) + "\n")

    def record(self, reason: str = "transport_error") -> None:
        """Record that a restart attempt is happening, pruning old entries."""
        now = time.time()
        entry = {"timestamp": now, "reason": reason}
        with self._lock:
            self._prune(now)
            self._history.append(entry)
            self._prune_file()

    def count(self, reason: str | None = None) -> int:
        """Return the number of restart attempts in the current window."""
        with self._lock:
            self._prune()
            if reason is None:
                return len(self._history)
            return sum(1 for entry in self._history if entry["reason"] == reason)

    def check(self, max_restarts: int, reason: str = "transport_error") -> None:
        """Raise AcpTransportError if we have restarted too many times recently."""
        if max_restarts <= 0:
            return
        with self._lock:
            self._prune()
            n = sum(1 for entry in self._history if entry["reason"] == reason)
            if n >= max_restarts:
                raise AcpTransportError(
                    "acp.restart",
                    msg=(
                        f"ACP transport has been restarted {n} times "
                        f"in the last {self._window}s; giving up"
                    ),
                )
