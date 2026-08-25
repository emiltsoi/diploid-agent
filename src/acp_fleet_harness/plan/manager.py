"""File-backed plan manager with thread-safe JSONL persistence."""

from __future__ import annotations

import fcntl
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from acp_fleet_harness.plan.models import Plan, PlanStatus, Task, TaskStatus

logger = logging.getLogger(__name__)


class PlanManager:
    """JSONL-backed plan and task store.

    Mutating operations re-read the backing file, apply the change, and
    atomically rewrite it under a cross-process file lock, matching the
    pattern used by ``WakeQueue`` and ``DispatchStore``.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "plans.jsonl"
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._plans: dict[str, Plan] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        self._plans = {}
        if not self._path.exists():
            return
        try:
            text = self._path.read_text()
        except OSError:
            logger.warning("Could not read plan store at %s", self._path)
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                plan = Plan.from_dict(data)
                self._plans[plan.id] = plan
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning("Skipping malformed plan store line")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(plan.to_dict(), default=str) + "\n" for plan in self._plans.values()]
        tmp = self._path.with_suffix(self._path.suffix + ".new")
        tmp.write_text("".join(lines))
        tmp.replace(self._path)

    @contextmanager
    def _transaction(self):
        """Acquire the cross-process lock, re-read, yield, then save."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self._lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                self._load()
                yield
                self._save()
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _unique_plan_id(self) -> str:
        return f"plan-{uuid.uuid4().hex[:12]}"

    def _unique_task_id(self) -> str:
        return f"task-{uuid.uuid4().hex[:12]}"

    def _get_task_in_plan(self, plan: Plan, task_id: str) -> Task | None:
        for task in plan.tasks:
            if task.id == task_id:
                return task
        return None

    def _resolve_statuses(self, plan: Plan) -> None:
        """Recompute PENDING/BLOCKED/READY for all tasks based on deps."""
        by_id = {t.id: t for t in plan.tasks}
        status_map = {t.id: t.status for t in plan.tasks}
        changed = True
        while changed:
            changed = False
            for task in plan.tasks:
                if task.status not in (
                    TaskStatus.PENDING,
                    TaskStatus.READY,
                    TaskStatus.BLOCKED,
                ):
                    continue

                missing = [d for d in task.depends_on if d not in by_id]
                dep_statuses = [status_map.get(d) for d in task.depends_on if d in by_id]
                any_failed = any(s in (TaskStatus.FAILED, TaskStatus.BLOCKED) for s in dep_statuses)
                all_done = all(s == TaskStatus.DONE for s in dep_statuses) and not missing

                if missing or any_failed:
                    if task.status != TaskStatus.BLOCKED:
                        task.status = TaskStatus.BLOCKED
                        status_map[task.id] = task.status
                        changed = True
                elif (all_done and task.depends_on) or not task.depends_on:
                    if task.status != TaskStatus.READY:
                        task.status = TaskStatus.READY
                        status_map[task.id] = task.status
                        changed = True
                else:
                    status_map[task.id] = task.status

    def _update_plan_status(self, plan: Plan) -> None:
        if not plan.tasks:
            plan.status = PlanStatus.DRAFT
            return
        if all(t.status == TaskStatus.DONE for t in plan.tasks):
            plan.status = PlanStatus.COMPLETED
        elif any(t.status == TaskStatus.FAILED for t in plan.tasks) and not any(
            t.status in (TaskStatus.RUNNING, TaskStatus.READY) for t in plan.tasks
        ):
            plan.status = PlanStatus.FAILED
        elif any(t.status in (TaskStatus.RUNNING, TaskStatus.READY) for t in plan.tasks):
            plan.status = PlanStatus.ACTIVE
        else:
            plan.status = PlanStatus.DRAFT

    def create_plan(
        self,
        name: str,
        description: str = "",
        chat_id: str | None = None,
        tasks: list[Task] | None = None,
    ) -> Plan:
        tasks = tasks or []
        for task in tasks:
            if not task.id:
                task.id = self._unique_task_id()
        now = time.time()
        plan = Plan(
            id=self._unique_plan_id(),
            name=name,
            description=description,
            chat_id=chat_id,
            tasks=tasks,
            created_at=now,
            updated_at=now,
        )
        self._resolve_statuses(plan)
        self._update_plan_status(plan)
        with self._transaction():
            self._plans[plan.id] = plan
        return plan

    def get_plan(self, plan_id: str) -> Plan | None:
        with self._transaction():
            return self._plans.get(plan_id)

    def list_plans(self, chat_id: str | None = None) -> list[Plan]:
        with self._transaction():
            plans = list(self._plans.values())
        if chat_id is not None:
            plans = [p for p in plans if p.chat_id == chat_id]
        return sorted(plans, key=lambda p: p.created_at, reverse=True)

    def add_task(self, plan_id: str, task: Task) -> Task | None:
        if not task.id:
            task.id = self._unique_task_id()
        with self._transaction():
            plan = self._plans.get(plan_id)
            if plan is None:
                return None
            by_id = {t.id for t in plan.tasks}
            if task.id in by_id:
                return None
            plan.tasks.append(task)
            self._resolve_statuses(plan)
            self._update_plan_status(plan)
            plan.updated_at = time.time()
            return task

    def get_task(self, plan_id: str, task_id: str) -> Task | None:
        with self._transaction():
            plan = self._plans.get(plan_id)
            if plan is None:
                return None
            return self._get_task_in_plan(plan, task_id)

    def get_ready_tasks(self, plan_id: str) -> list[Task]:
        with self._transaction():
            plan = self._plans.get(plan_id)
            if plan is None:
                return []
            return [t for t in plan.tasks if t.status == TaskStatus.READY]

    def start_task(self, plan_id: str, task_id: str, now: float | None = None) -> Task | None:
        if now is None:
            now = time.time()
        with self._transaction():
            plan = self._plans.get(plan_id)
            if plan is None:
                return None
            task = self._get_task_in_plan(plan, task_id)
            if task is None or task.status != TaskStatus.READY:
                return None
            task.status = TaskStatus.RUNNING
            task.started_at = now
            self._update_plan_status(plan)
            plan.updated_at = now
            return task

    def complete_task(
        self,
        plan_id: str,
        task_id: str,
        result: str = "",
        log: str = "",
    ) -> Task | None:
        with self._transaction():
            plan = self._plans.get(plan_id)
            if plan is None:
                return None
            task = self._get_task_in_plan(plan, task_id)
            if task is None:
                return None
            task.status = TaskStatus.DONE
            task.result = result
            task.log = log
            task.completed_at = time.time()
            self._resolve_statuses(plan)
            self._update_plan_status(plan)
            plan.updated_at = time.time()
            return task

    def fail_task(self, plan_id: str, task_id: str, log: str = "") -> Task | None:
        with self._transaction():
            plan = self._plans.get(plan_id)
            if plan is None:
                return None
            task = self._get_task_in_plan(plan, task_id)
            if task is None:
                return None
            task.status = TaskStatus.FAILED
            task.log = log
            task.completed_at = time.time()
            self._resolve_statuses(plan)
            self._update_plan_status(plan)
            plan.updated_at = time.time()
            return task
