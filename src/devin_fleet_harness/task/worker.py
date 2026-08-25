"""Thread-pool worker for background task execution."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class WorkerPool:
    """Resizable thread-pool keyed by task id."""

    def __init__(self, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self._lock = threading.Lock()
        self._max_workers = max_workers
        self._running = True
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="task-worker-",
        )
        self._futures: dict[str, Future[Any]] = {}

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def resize(self, max_workers: int) -> None:
        """Recreate the executor with a new worker limit.

        Already-running tasks continue on the old executor. New submissions
        use the new one.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        with self._lock:
            if max_workers == self._max_workers:
                return
            self._prune_done()
            self._executor.shutdown(wait=False, cancel_futures=False)
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="task-worker-",
            )
            self._max_workers = max_workers
            self._running = True

    def _prune_done(self) -> None:
        """Drop completed futures to avoid unbounded memory growth."""
        self._futures = {
            task_id: future for task_id, future in self._futures.items() if not future.done()
        }

    def submit(self, task_id: str, fn: Callable[[], Any]) -> None:
        with self._lock:
            self._prune_done()
            future = self._futures.get(task_id)
            if future is not None and not future.done():
                logger.warning("Task %s is already running; ignoring duplicate submit", task_id)
                return
            self._futures[task_id] = self._executor.submit(fn)

    def running(self, task_id: str) -> bool:
        with self._lock:
            future = self._futures.get(task_id)
            return future is not None and not future.done()

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._running = False
            self._prune_done()
            self._executor.shutdown(wait=wait, cancel_futures=False)
