"""Typing heartbeat for active tasks."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diploid_agent.plan.models import Task
    from diploid_agent.runtime.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)


class RuntimeTyping:
    """Drive ``notifier.typing(...)`` while background tasks are running."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime
        self._counts: dict[str, int] = {}
        self._threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._lock = threading.Lock()

    def _heartbeat(self, chat_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self._runtime.notifier.typing(chat_id)
            except Exception:
                logger.exception("Typing heartbeat for %s failed", chat_id)
            if stop_event.wait(4.0):
                break

    def on_task_started(self, _plan_id: str, task: Task) -> None:
        if task.chat_id is None:
            return
        with self._lock:
            count = self._counts.get(task.chat_id, 0)
            self._counts[task.chat_id] = count + 1
            if count == 0:
                stop_event = threading.Event()
                thread = threading.Thread(
                    target=self._heartbeat,
                    args=(task.chat_id, stop_event),
                    daemon=True,
                    name=f"typing-{task.chat_id}",
                )
                self._threads[task.chat_id] = (thread, stop_event)
                thread.start()

    def on_task_done(
        self,
        _plan_id: str,
        task: Task,
        _outcome: tuple[str, str, int],
    ) -> None:
        if task.chat_id is None:
            return
        with self._lock:
            count = self._counts.get(task.chat_id, 0)
            if count <= 0:
                return
            count -= 1
            self._counts[task.chat_id] = count
            if count == 0:
                entry = self._threads.pop(task.chat_id, None)
                if entry is not None:
                    _, stop_event = entry
                    stop_event.set()

    def stop(self) -> None:
        """Stop all typing heartbeats."""
        with self._lock:
            threads = list(self._threads.values())
            self._counts.clear()
            self._threads.clear()
        for _, stop_event in threads:
            stop_event.set()
        for thread, _ in threads:
            thread.join(timeout=1.0)
