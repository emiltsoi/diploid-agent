"""In-memory event bus for the runtime."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Event(BaseModel):
    """One event on the bus."""

    id: str = Field(default_factory=lambda: f"event-{uuid.uuid4().hex[:12]}")
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class EventBus:
    """Thread-safe publish/subscribe event queue.

    Subscribers run on a single background thread so events are delivered
    one at a time. The bus is a Phase 1 skeleton; it is intentionally simple.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Event], None]] = []
        self._queue: Queue[Event] = Queue()
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._queue.put(Event(type="stop", payload={}))
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    @property
    def running(self) -> bool:
        """Return whether the event bus dispatch thread is running."""
        return self._running

    def post(self, event: Event) -> None:
        self._queue.put(event)

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.remove(callback)

    def _dispatch_loop(self) -> None:
        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if not self._running or event.type == "stop":
                break
            with self._lock:
                subscribers = list(self._subscribers)
            for callback in subscribers:
                try:
                    callback(event)
                except Exception:
                    logger.exception("Subscriber failed for event %s", event.id)
