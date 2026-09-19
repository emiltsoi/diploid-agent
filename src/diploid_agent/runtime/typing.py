"""Typing heartbeat for active tasks."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diploid_agent.plan.models import Task

logger = logging.getLogger(__name__)


class RuntimeTyping:
    """Drive ``notifier.typing(...)`` while background tasks are running."""

    def __init__(self, notifier_fn: Callable[[], Any]) -> None:
        self._notifier_fn = notifier_fn
        self._counts: dict[str, int] = {}
        self._threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._turn_chats: set[str] = set()
        self._lock = threading.Lock()

    @property
    def _notifier(self) -> Any:
        return self._notifier_fn()

    def _heartbeat(self, chat_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self._notifier.typing(chat_id)
            except Exception:
                logger.exception("Typing heartbeat for %s failed", chat_id)
            if stop_event.wait(4.0):
                break

    def on_task_started(self, _plan_id: str, task: Task) -> None:
        if task.chat_id is None:
            return
        with self._lock:
            self._increment(task.chat_id)

    def on_task_done(
        self,
        _plan_id: str,
        task: Task,
        _outcome: tuple[str, str, int],
    ) -> None:
        if task.chat_id is None:
            return
        with self._lock:
            self._decrement(task.chat_id)

    def on_turn_started(self, chat_id: str) -> None:
        """Type while a wake-driven turn runs (mesh/wake/cron/continuation).

        Only call for turns with no poller-side typing context — user-message
        turns already type via the poller. ``_turn_chats`` pairs starts with
        finishes so a turn that never started cannot eat a task's count.
        """
        with self._lock:
            self._turn_chats.add(chat_id)
            self._increment(chat_id)

    def on_turn_finished(self, chat_id: str) -> None:
        with self._lock:
            if chat_id not in self._turn_chats:
                return
            self._turn_chats.discard(chat_id)
            self._decrement(chat_id)

    def _increment(self, chat_id: str) -> None:
        count = self._counts.get(chat_id, 0)
        self._counts[chat_id] = count + 1
        if count == 0:
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._heartbeat,
                args=(chat_id, stop_event),
                daemon=True,
                name=f"typing-{chat_id}",
            )
            self._threads[chat_id] = (thread, stop_event)
            thread.start()

    def _decrement(self, chat_id: str) -> None:
        count = self._counts.get(chat_id, 0)
        if count <= 0:
            return
        count -= 1
        self._counts[chat_id] = count
        if count == 0:
            entry = self._threads.pop(chat_id, None)
            if entry is not None:
                _, stop_event = entry
                stop_event.set()

    def stop(self) -> None:
        """Stop all typing heartbeats."""
        with self._lock:
            threads = list(self._threads.values())
            self._counts.clear()
            self._threads.clear()
            self._turn_chats.clear()
        for _, stop_event in threads:
            stop_event.set()
        for thread, _ in threads:
            thread.join(timeout=1.0)
