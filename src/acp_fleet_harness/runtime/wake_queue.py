"""Persistent, multi-process safe wake queue for proactive harness wake events."""

from __future__ import annotations

import fcntl
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from acp_fleet_harness.models import WakeEvent

logger = logging.getLogger(__name__)


class WakeQueue:
    """JSONL-backed queue of wake events with cross-process locking.

    Events start as ``ready=False``. Callers mark them ready with ``.ready()``,
    the waker pops and claims due events with ``.pop_due()``, and the
    ``/wake`` endpoint consumes them with ``.complete()``. ``.complete()`` and
    ``.ready()`` are idempotent; ``.fail()`` reschedules a claimed event for
    later retry.

    Mutating operations acquire a file lock, re-read the JSONL backing file,
    apply the change, and atomically replace the file. The lock is held on a
    dedicated ``.lock`` file so the JSONL itself can be replaced without
    invalidating the lock.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path).expanduser()
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._in_memory: dict[str, WakeEvent] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        self._in_memory = {}
        if not self._path.exists():
            return
        try:
            text = self._path.read_text()
        except OSError:
            logger.warning("Could not read wake queue at %s", self._path)
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                event = WakeEvent.from_dict(data)
                self._in_memory[event.id] = event
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("Skipping malformed wake queue line")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(asdict(e), default=str) + "\n" for e in self._in_memory.values()]
        tmp = self._path.with_suffix(self._path.suffix + ".new")
        tmp.write_text("".join(lines))
        tmp.replace(self._path)

    @contextmanager
    def _transaction(self):
        """Acquire the cross-process lock, re-read the file, yield, then save."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self._lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._load()
                yield
                self._save()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _unique_id(self) -> str:
        return f"wake-{uuid.uuid4().hex[:12]}"

    def enqueue(self, event: WakeEvent) -> WakeEvent:
        if not event.id:
            event.id = self._unique_id()
        with self._transaction():
            self._in_memory[event.id] = event
        return event

    def get(self, event_id: str) -> WakeEvent | None:
        with self._transaction():
            return self._in_memory.get(event_id)

    def ready(self, event_id: str, now: float | None = None) -> WakeEvent | None:
        with self._transaction():
            event = self._in_memory.get(event_id)
            if event is None or event.ready:
                return event
            event.ready = True
            event.scheduled_at = now if now is not None else time.time()
            event.leased_until = None
            return event

    def pending(
        self,
        chat_id: str | None = None,
        now: float | None = None,
    ) -> list[WakeEvent]:
        with self._transaction():
            events = list(self._in_memory.values())
        if chat_id is not None:
            events = [e for e in events if e.chat_id == chat_id]
        if now is not None:
            events = [e for e in events if not e.ready or e.scheduled_at <= now]
        return sorted(events, key=lambda e: (-e.priority, e.scheduled_at, e.created_at))

    def pop_due(
        self,
        now: float | None = None,
        lease_seconds: float = 300.0,
    ) -> list[WakeEvent]:
        if now is None:
            now = time.time()
        with self._transaction():
            due = [
                e
                for e in self._in_memory.values()
                if e.ready
                and e.scheduled_at <= now
                and (e.leased_until is None or e.leased_until <= now)
            ]
            due = sorted(due, key=lambda e: (-e.priority, e.scheduled_at, e.created_at))
            for event in due:
                event.leased_until = now + lease_seconds
            return due

    def complete(self, event_id: str) -> WakeEvent | None:
        with self._transaction():
            event = self._in_memory.pop(event_id, None)
            return event

    def fail(
        self,
        event_id: str,
        retry_after: float,
        now: float | None = None,
    ) -> WakeEvent | None:
        if now is None:
            now = time.time()
        with self._transaction():
            event = self._in_memory.get(event_id)
            if event is None:
                return None
            event.attempts += 1
            event.scheduled_at = now + retry_after
            event.leased_until = None
            return event

    def pending_count(self) -> int:
        with self._transaction():
            return sum(
                1
                for e in self._in_memory.values()
                if e.ready and (e.leased_until is None or e.leased_until <= time.time())
            )

    def due_count(self, now: float | None = None) -> int:
        if now is None:
            now = time.time()
        with self._transaction():
            return sum(
                1
                for e in self._in_memory.values()
                if e.ready
                and e.scheduled_at <= now
                and (e.leased_until is None or e.leased_until <= now)
            )

    def cancel(
        self,
        chat_id: str | None = None,
        reason: str | None = None,
        now: float | None = None,
    ) -> int:
        """Remove matching ready events and return how many were removed.

        Only events that are not currently leased are cancelled; a wake that is
        actively being processed is left alone.
        """
        if now is None:
            now = time.time()
        with self._transaction():
            to_remove = [
                event_id
                for event_id, event in self._in_memory.items()
                if event.ready
                and (chat_id is None or event.chat_id == chat_id)
                and (reason is None or event.reason == reason)
                and (event.leased_until is None or event.leased_until <= now)
            ]
            for event_id in to_remove:
                self._in_memory.pop(event_id)
            return len(to_remove)
