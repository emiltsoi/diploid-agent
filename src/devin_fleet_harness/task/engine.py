"""Task engine: schedules and executes plan tasks using a worker pool."""

from __future__ import annotations

import logging
import os
import subprocess
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from devin_fleet_harness.config import Config, TaskConfig
from devin_fleet_harness.engine import AgentEngine, DevinAcpEngine, TurnRequest
from devin_fleet_harness.plan.manager import PlanManager
from devin_fleet_harness.plan.models import Plan, Task, TaskStatus, TaskType
from devin_fleet_harness.runtime.event_bus import Event, EventBus
from devin_fleet_harness.task.worker import WorkerPool

logger = logging.getLogger(__name__)


class TaskEngine:
    """Background task scheduler and executor for living plans.

    Reads its operational settings from a live ``TaskConfig`` so that worker
    count, timeouts, enabled task types, and ACP overrides can change without
    a full harness restart.
    """

    def __init__(
        self,
        plan_manager: PlanManager,
        event_bus: EventBus,
        *,
        engine: AgentEngine | None = None,
        model: str | None = None,
        acp_timeout: float | None = None,
        config: Config | None = None,
        task_config: TaskConfig | None = None,
        max_workers: int = 4,
        shell_timeout: float = 60.0,
        on_task_start: Callable[[str, Task], None] | None = None,
        on_task_done: Callable[[str, Task, tuple[str, str, int]], None] | None = None,
    ) -> None:
        self.plan_manager = plan_manager
        self.event_bus = event_bus
        self.engine = engine
        self._model = model
        self._acp_timeout = acp_timeout
        self.config = config
        if task_config is None:
            task_config = TaskConfig(
                workers=max_workers,
                shell_timeout=shell_timeout,
                enabled_types=TaskConfig().enabled_types,
                acp_timeout=acp_timeout,
                acp_model=model,
            )
        self._task_config = task_config
        self._pool = WorkerPool(max_workers=self._task_config.workers)
        self._on_task_start = on_task_start
        self._on_task_done = on_task_done

    @property
    def task_config(self) -> TaskConfig:
        return self._task_config

    def reconfigure(self) -> None:
        """Apply settings that need explicit wiring (e.g., pool size)."""
        self._pool.resize(self._task_config.workers)

    def start_task(self, plan_id: str, task_id: str | None = None) -> Task:
        """Start a ready task. If no task_id is provided, pick the first ready task."""
        if task_id is None:
            ready = self.plan_manager.get_ready_tasks(plan_id)
            if not ready:
                raise ValueError(f"No ready tasks in plan {plan_id}")
            task = ready[0]
            task_id = task.id
        else:
            task = self.plan_manager.get_task(plan_id, task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found in plan {plan_id}")
            if task.status != TaskStatus.READY:
                raise ValueError(f"Task {task_id} is not ready (status={task.status})")

        started = self.plan_manager.start_task(plan_id, task_id, now=time.time())
        if started is None:
            raise ValueError(f"Could not start task {task_id}")

        self._pool.resize(self._task_config.workers)
        self._pool.submit(started.id, lambda: self._run_task(plan_id, started.id))
        return started

    def _run_task(self, plan_id: str, task_id: str) -> None:
        task = self.plan_manager.get_task(plan_id, task_id)
        if task is None:
            logger.warning("Task %s not found during execution", task_id)
            return

        result: str = ""
        log: str = ""
        exit_code: int = -1

        if task.type.value not in self._task_config.enabled_types:
            result = f"Task type {task.type.value} is disabled"
            log = ""
            exit_code = -1
            failed = self.plan_manager.fail_task(plan_id, task_id, log=result)
            if failed:
                self.event_bus.post(
                    Event(
                        type="task.failed",
                        payload={
                            "plan_id": plan_id,
                            "task_id": task_id,
                            "log": result,
                        },
                    )
                )
                self._start_ready_dependents(plan_id)
        else:
            if self._on_task_start is not None:
                try:
                    self._on_task_start(plan_id, task)
                except Exception:
                    logger.exception("Task %s on_task_start failed", task_id)

            try:
                result, log, exit_code = self._execute(task)
                if exit_code == 0:
                    completed = self.plan_manager.complete_task(
                        plan_id, task_id, result=result, log=log
                    )
                    if completed:
                        self.event_bus.post(
                            Event(
                                type="task.completed",
                                payload={
                                    "plan_id": plan_id,
                                    "task_id": task_id,
                                    "result": result,
                                },
                            )
                        )
                        self._start_ready_dependents(plan_id)
                else:
                    failed = self.plan_manager.fail_task(plan_id, task_id, log=log or result)
                    if failed:
                        self.event_bus.post(
                            Event(
                                type="task.failed",
                                payload={
                                    "plan_id": plan_id,
                                    "task_id": task_id,
                                    "log": log or result,
                                },
                            )
                        )
                        self._start_ready_dependents(plan_id)
            except Exception as exc:
                logger.exception("Task %s failed", task_id)
                result = ""
                log = f"{exc}\n{traceback.format_exc()}"
                exit_code = -1
                self.plan_manager.fail_task(plan_id, task_id, log=log)
                self.event_bus.post(
                    Event(
                        type="task.failed",
                        payload={
                            "plan_id": plan_id,
                            "task_id": task_id,
                            "error": str(exc),
                        },
                    )
                )

        if self._on_task_done is not None:
            try:
                self._on_task_done(plan_id, task, (result, log, exit_code))
            except Exception:
                logger.exception("Task %s on_task_done failed", task_id)

    def _execute(self, task: Task) -> tuple[str, str, int]:
        if task.type == TaskType.SHELL:
            return self._run_shell(task)
        if task.type == TaskType.ACP:
            return self._run_acp(task)
        if task.type == TaskType.NOOP:
            return "noop", "", 0
        return f"Unknown task type {task.type}", "", -1

    def _run_shell(self, task: Task) -> tuple[str, str, int]:
        cwd = task.cwd if task.cwd is not None else Path(os.getcwd())
        timeout = self._task_config.shell_timeout
        try:
            proc = subprocess.run(
                task.command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            return "", f"Timed out after {timeout}s", -1
        except (OSError, ValueError) as exc:
            return "", str(exc), -1

    def _run_acp(self, task: Task) -> tuple[str, str, int]:
        prompt_text = task.prompt or task.command
        if not prompt_text:
            return "ACP task has no prompt or command", "", -1

        cwd = task.cwd if task.cwd is not None else Path(os.getcwd())

        # ACP tasks are background subagents and must not share the chat's
        # AcpClient / devin acp process. Create a fresh engine per task.
        if isinstance(self.engine, DevinAcpEngine):
            if self.config is None:
                return "ACP task config not available", "", -1
            api_key = None
            if self.config.secrets:
                api_key = self.config.secrets.windsurf_api_key
            engine = DevinAcpEngine(config=self.config.devin, api_key=api_key)
        elif self.engine is not None:
            engine = self.engine
        else:
            return "ACP engine not configured", "", -1

        model = (
            task.acp_model
            if task.acp_model is not None
            else self._task_config.acp_model
            if self._task_config.acp_model is not None
            else self._model
            if self._model is not None
            else self.config.devin.model
            if self.config is not None
            else None
        )
        soft_timeout = (
            self._task_config.acp_timeout
            if self._task_config.acp_timeout is not None
            else self._acp_timeout
            if self._acp_timeout is not None
            else self.config.devin.soft_timeout
            if self.config is not None
            else None
        )

        request = TurnRequest(
            prompt=prompt_text,
            cwd=cwd,
            model=model,
            mcp_servers=None,
            soft_timeout=soft_timeout,
        )
        try:
            result = engine.prompt(request)
            return result.reply, "", 0
        except (RuntimeError, TimeoutError) as exc:
            logger.exception("ACP task %s timed out", task.id)
            return (str(exc), "", -1)
        except Exception as exc:
            logger.exception("ACP task %s failed", task.id)
            return (str(exc), "", -1)
        finally:
            if isinstance(engine, DevinAcpEngine):
                engine.close()

    def _start_ready_dependents(self, plan_id: str) -> None:
        """Auto-start any tasks that became ready after the last state change."""
        for task in self.plan_manager.get_ready_tasks(plan_id):
            if not self._pool.running(task.id):
                try:
                    self.start_task(plan_id, task.id)
                except ValueError:
                    pass

    def get_status(self, plan_id: str) -> Plan | None:
        return self.plan_manager.get_plan(plan_id)

    def is_running(self) -> bool:
        return self._pool.is_running()

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)
