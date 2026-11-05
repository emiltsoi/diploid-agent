"""Per-chat durable event log for multi-client sync.

Every chat gets a ``chat_events.jsonl`` in its session dir. Each line is
``{seq, ts, kind, chat_id, payload}`` with ``seq`` a per-chat monotonic
counter — so a client can long-poll ``GET /events/{chat_id}?since=N`` and
receive exactly the events it has not seen, including across restarts.

The pop-queue outbox stays single-consumer (Telegram's delivery worker).
This log is the shared read side: N clients read it, nobody consumes it.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Events are turn-granular, not chunk-granular — a few KB per day.
_RING_MAX = 500


class EventLog:
    """Append-only per-chat event log with a hot tail cache."""

    def __init__(self, chat_dir_fn: Callable[[str], Path]) -> None:
        self._chat_dir_fn = chat_dir_fn
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._rings: dict[str, deque[dict[str, Any]]] = {}
        self._loaded: set[str] = set()

    def _events_path(self, chat_id: str) -> Path:
        return self._chat_dir_fn(chat_id) / "chat_events.jsonl"

    def _ring(self, chat_id: str) -> deque[dict[str, Any]]:
        if chat_id in self._loaded:
            return self._rings[chat_id]
        ring: deque[dict[str, Any]] = deque(maxlen=_RING_MAX)
        path = self._events_path(chat_id)
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in lines[-_RING_MAX:]:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and isinstance(event.get("seq"), int):
                    ring.append(event)
        self._rings[chat_id] = ring
        self._loaded.add(chat_id)
        return ring

    def next_seq(self, chat_id: str) -> int:
        """The seq the next appended event for this chat will carry."""
        with self._lock:
            ring = self._ring(chat_id)
            return (ring[-1]["seq"] + 1) if ring else 1

    def append(self, chat_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append an event to the chat's log; returns the stored event."""
        with self._cond:
            ring = self._ring(chat_id)
            event = {
                "seq": (ring[-1]["seq"] + 1) if ring else 1,
                "ts": time.time(),
                "kind": kind,
                "chat_id": chat_id,
                "payload": payload,
            }
            ring.append(event)
            path = self._events_path(chat_id)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except OSError:
                # The ring still carries it; a client polling before the
                # next append sees the event. Persistence resumes next write,
                # but the file keeps a permanent interior seq hole — clients
                # must tolerate gaps, ``truncated`` only detects head loss.
                pass
            self._cond.notify_all()
            return event

    def since(
        self,
        chat_id: str,
        since: int = 0,
        wait: float = 0.0,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Return events with ``seq > since`` for a chat.

        Long-polls up to ``wait`` seconds for new events. ``next_seq`` is
        always the seq the next event will carry, so a client that fell
        behind the ring resumes cleanly: ``since < next_seq - len(events)``
        means truncation happened and ``truncated`` is true.
        """
        deadline = time.monotonic() + wait if wait > 0 else 0.0
        with self._cond:
            while True:
                ring = self._ring(chat_id)
                events = [e for e in ring if e["seq"] > since][:limit]
                next_seq = (ring[-1]["seq"] + 1) if ring else 1
                truncated = bool(ring) and since < ring[0]["seq"] - 1
                if events or wait <= 0:
                    return {
                        "chat_id": chat_id,
                        "events": events,
                        "next_seq": next_seq,
                        "truncated": truncated,
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "chat_id": chat_id,
                        "events": [],
                        "next_seq": next_seq,
                        "truncated": truncated,
                    }
                self._cond.wait(timeout=remaining)
