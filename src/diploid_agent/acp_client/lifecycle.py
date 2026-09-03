"""Append-only audit log for ACP transport and session lifecycle events."""

from __future__ import annotations

import json
import logging
import threading
import time
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

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

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
        """Append a lifecycle event to the log."""
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

    def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return the most recent log entries."""
        if not self.path.exists():
            return []
        with self._lock, self.path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]


class AcpRestartHistory:
    """Persistent, per-harness restart attempt history for ACP transport backoff.

    Stores monotonic timestamps in a JSONL file so the backoff counter survives
    process restarts.  In-memory fallback when no path is provided (tests).
    """

    def __init__(self, path: Path | None, window: float) -> None:
        self.path = Path(path) if path is not None else None
        self._window = max(1.0, window)
        self._lock = threading.Lock()
        self._history: list[float] = []
        self._load()

    def _load(self) -> None:
        """Load existing restart timestamps from disk."""
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
                        self._history.append(ts)
        except (OSError, ValueError, TypeError):
            logger.exception("Failed to load ACP restart history from %s", self.path)

    def _prune(self, now: float | None = None) -> None:
        """Drop timestamps outside the configured backoff window."""
        now = now if now is not None else time.time()
        cutoff = now - self._window
        self._history = [t for t in self._history if t > cutoff]

    def record(self) -> None:
        """Record that a restart attempt is happening, pruning old entries."""
        now = time.time()
        with self._lock:
            self._prune(now)
            self._history.append(now)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"timestamp": now}, default=str) + "\n")

    def count(self) -> int:
        """Return the number of restart attempts in the current window."""
        with self._lock:
            self._prune()
            return len(self._history)

    def check(self, max_restarts: int) -> None:
        """Raise AcpTransportError if we have restarted too many times recently."""
        if max_restarts <= 0:
            return
        with self._lock:
            self._prune()
            if len(self._history) >= max_restarts:
                raise AcpTransportError(
                    "acp.restart",
                    msg=(
                        f"ACP transport has been restarted {len(self._history)} times "
                        f"in the last {self._window}s; giving up"
                    ),
                )
