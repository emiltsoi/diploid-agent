"""Persistent, multi-process safe wake queue for proactive harness wake events."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from diploid_agent._compat import atomic_replace, fcntl
from diploid_agent.models import WakeEvent
from diploid_agent.sanitize import sanitize_json_value

logger = logging.getLogger(__name__)

# Reasons whose pending events draw on the shared self-wake budget: agent
# self-wakes and cron turn deliveries are one accounting pool, so neither
# can stack armed interrupts on top of the other.
WAKE_BUDGET_REASON_PREFIXES = ("self_wake", "cron:")


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
        self._quarantine_path = self._path.with_suffix(".quarantine.jsonl")
        self._in_memory: dict[str, WakeEvent] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        self._in_memory = {}
        if not self._path.exists():
            return
        try:
            text = self._path.read_text(encoding="utf-8")
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
        tmp.write_text("".join(lines), encoding="utf-8")
        atomic_replace(tmp, self._path)

    @contextmanager
    def _transaction(self):
        """Acquire the cross-process lock, re-read the file, yield, then save."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self._lock_path, "a+", encoding="utf-8") as lock_file:
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
        # Record-time scrub: a lone surrogate in a wire payload would crash the
        # utf-8 save on every retry — the vesper wake-queue wedge class.
        event.payload = sanitize_json_value(event.payload)
        with self._transaction():
            self._in_memory[event.id] = event
        return event

    def merge_into_ready(
        self,
        event: WakeEvent,
        *,
        key: str,
        key_field: str = "plan_id",
        list_field: str = "tasks",
    ) -> WakeEvent | None:
        """Fold `event` into an already-ready, unleased event sharing its
        ``reason``, ``chat_id``, and ``payload[key_field] == key``.

        Pending bookkeeping wakes for one plan coalesce into a single digest
        event instead of one ping per task; mesh informational deliveries
        coalesce per peer the same way (``key_field="sender"``,
        ``list_field="messages"``). The named payload list is concatenated;
        every other payload key takes the newer event's value. Returns the
        absorbing event (caller must not enqueue `event`), or ``None``.
        """
        now = time.time()
        with self._transaction():
            candidates = [
                e
                for e in self._in_memory.values()
                if e.chat_id == event.chat_id
                and e.reason == event.reason
                and e.ready
                and (e.leased_until is None or e.leased_until <= now)
                and e.payload.get(key_field) == key
            ]
            if not candidates:
                return None
            # Newest armed digest wins if two ever coexist (e.g. a parked
            # arrival created a second digest while the first was leased);
            # the merge then stretches only that digest's own window.
            target = max(candidates, key=lambda e: e.created_at)
            merged = dict(target.payload)
            items = list(merged.get(list_field) or [])
            items.extend(event.payload.get(list_field) or [])
            merged.update({k: v for k, v in event.payload.items() if k != list_field})
            merged[list_field] = items
            target.payload = sanitize_json_value(merged)
            return target

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

    def fire_now(self, event_id: str, now: float | None = None) -> WakeEvent | None:
        """Force an armed event to fire on the next tick.

        Used by flush-on-priority: an urgent delivery from a peer with a
        parked digest fires the digest without waiting out its window.
        Deliberately not ``fail()`` — no attempts increment, no retry
        backoff. Leased events are left alone (they pop on their own or
        fail forward); ``None`` when the event is absent or not ready.
        """
        if now is None:
            now = time.time()
        with self._transaction():
            event = self._in_memory.get(event_id)
            if event is None or not event.ready:
                return None
            if event.leased_until is not None and event.leased_until > now:
                return event
            event.scheduled_at = now
            return event

    def fire_matching(
        self,
        *,
        chat_id: str,
        reason: str,
        key_field: str,
        key: str,
        now: float | None = None,
    ) -> list[WakeEvent]:
        """``fire_now`` for a class of events rather than one id.

        Flush-on-priority can't name a digest by id — digest ids are
        msg_id-suffixed so a parked arrival can never overwrite an
        in-flight (leased) digest under a deterministic id. Forces every
        ready, unleased event matching ``chat_id``/``reason``/
        ``payload[key_field] == key`` to fire next tick.
        """
        if now is None:
            now = time.time()
        with self._transaction():
            fired = []
            for e in self._in_memory.values():
                if (
                    e.chat_id == chat_id
                    and e.reason == reason
                    and e.ready
                    and (e.leased_until is None or e.leased_until <= now)
                    and e.payload.get(key_field) == key
                ):
                    e.scheduled_at = now
                    fired.append(e)
            return fired

    def complete(self, event_id: str) -> WakeEvent | None:
        with self._transaction():
            event = self._in_memory.pop(event_id, None)
            return event

    def fail(
        self,
        event_id: str,
        retry_after: float,
        now: float | None = None,
        max_retries: int | None = None,
        last_error: str | None = None,
    ) -> WakeEvent | None:
        """Reschedule a claimed event for retry, or dead-letter it.

        ``max_retries`` bounds the retry count: when the next failure would
        exceed it, the event is appended to the sibling quarantine file and
        removed from the live queue instead of being rescheduled. ``None``
        preserves the legacy unbounded behavior; ``0`` quarantines on the
        first failure. ``last_error`` is recorded in the dead letter so the
        row explains itself.
        """
        if now is None:
            now = time.time()
        with self._transaction():
            event = self._in_memory.get(event_id)
            if event is None:
                return None
            event.attempts += 1
            if max_retries is not None and event.attempts > max_retries:
                event.leased_until = None
                self._append_quarantine(event, now=now, last_error=last_error)
                self._in_memory.pop(event_id)
                logger.error(
                    "Wake event %s quarantined after %d attempts (chat %s, reason %s): %s",
                    event.id,
                    event.attempts,
                    event.chat_id,
                    event.reason,
                    last_error,
                )
                return event
            event.scheduled_at = now + retry_after
            event.leased_until = None
            return event

    def _append_quarantine(
        self,
        event: WakeEvent,
        *,
        now: float,
        last_error: str | None,
    ) -> None:
        row = asdict(event)
        row["quarantined_at"] = now
        row["last_error"] = last_error
        with open(self._quarantine_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def quarantined(self) -> list[dict]:
        """Read the dead-letter file. Returns [] when nothing quarantined."""
        try:
            text = self._quarantine_path.read_text(encoding="utf-8")
        except OSError:
            return []
        rows = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed quarantine line")
        return rows

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

    def cancel_event(
        self,
        event_id: str,
        chat_id: str | None = None,
        now: float | None = None,
    ) -> WakeEvent | None:
        """Remove one ready, unleased event by id and return it.

        ``chat_id``, when given, scopes the removal so a caller can only
        retract events belonging to its own chat. Currently leased events are
        left alone so a wake being processed is not cancelled mid-flight.
        """
        if now is None:
            now = time.time()
        with self._transaction():
            event = self._in_memory.get(event_id)
            if (
                event is None
                or not event.ready
                or (chat_id is not None and event.chat_id != chat_id)
                or (event.leased_until is not None and event.leased_until > now)
            ):
                return None
            return self._in_memory.pop(event_id)

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

    def cancel_older_than(
        self,
        before: float,
        reason: str | None = None,
        now: float | None = None,
    ) -> int:
        """Remove ready events created before ``before`` and return the count.

        Currently leased events are left alone so a wake that is being
        processed is not cancelled mid-flight.
        """
        if now is None:
            now = time.time()
        with self._transaction():
            to_remove = [
                event_id
                for event_id, event in self._in_memory.items()
                if event.ready
                and event.created_at < before
                and (reason is None or event.reason == reason)
                and (event.leased_until is None or event.leased_until <= now)
            ]
            for event_id in to_remove:
                self._in_memory.pop(event_id)
            return len(to_remove)
