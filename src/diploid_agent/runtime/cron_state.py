"""Persistent, multi-process safe cron job state store.

Mirrors ``WakeQueue``: JSONL backing file, cross-process flock on a
dedicated ``.lock`` file, read-modify-atomic-replace transactions.
"""

from __future__ import annotations

import fcntl
import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CronJobState(BaseModel):
    """Runtime state for one cron job."""

    job_id: str
    source_file: str = ""
    last_run_at: float | None = None
    next_due_at: float | None = None
    last_status: str | None = None  # ok | failed | skipped
    running_task_id: str | None = None
    running_plan_id: str | None = None
    consecutive_failures: int = 0
    disabled: bool = False  # auto-disabled after consecutive failures
    last_finished_at: float | None = None
    last_summary: str = ""
    last_task_id: str | None = None  # survives restart for reconciliation
    queued_due: bool = False  # overlap=queue: a due fire is owed
    updated_at: float = Field(default_factory=time.time)


class CronStateStore:
    """JSONL-backed per-job state with cross-process locking."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path).expanduser()
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._in_memory: dict[str, CronJobState] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        self._in_memory = {}
        if not self._path.exists():
            return
        try:
            text = self._path.read_text()
        except OSError:
            logger.warning("Could not read cron state at %s", self._path)
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                state = CronJobState.model_validate(json.loads(line))
                self._in_memory[state.job_id] = state
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("Skipping malformed cron state line")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [state.model_dump_json() + "\n" for state in self._in_memory.values()]
        tmp = self._path.with_suffix(self._path.suffix + ".new")
        tmp.write_text("".join(lines))
        tmp.replace(self._path)

    @contextmanager
    def _transaction(self, save: bool = True):
        """Lock, re-read, yield; write back only when ``save`` is set."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self._lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._load()
                yield
                if save:
                    self._save()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def get(self, job_id: str) -> CronJobState | None:
        with self._transaction(save=False):
            return self._in_memory.get(job_id)

    def ensure(self, job_id: str, source_file: str = "") -> CronJobState:
        with self._transaction(save=False):
            state = self._in_memory.get(job_id)
        if state is None:
            state = CronJobState(job_id=job_id, source_file=source_file)
            return self.update(state)
        return state

    def update(self, state: CronJobState) -> CronJobState:
        state.updated_at = time.time()
        with self._transaction():
            self._in_memory[state.job_id] = state
            return state

    def all(self) -> dict[str, CronJobState]:
        with self._transaction(save=False):
            return dict(self._in_memory)

    def remove_missing(self, job_ids: set[str]) -> None:
        """Drop state for job ids no longer present in any config file."""
        with self._transaction():
            for job_id in list(self._in_memory):
                if job_id not in job_ids and self._in_memory[job_id].running_task_id is None:
                    del self._in_memory[job_id]
