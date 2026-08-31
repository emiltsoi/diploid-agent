"""Task engine: schedules and executes plan tasks using a worker pool."""

from __future__ import annotations

import logging
import os
import subprocess
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.config import Config, TaskConfig
from diploid_agent.engine import AcpEngine, AgentEngine, TurnRequest
from diploid_agent.engine.factory import build_engine
from diploid_agent.plan.manager import PlanManager
from diploid_agent.plan.models import Plan, Task, TaskStatus, TaskType
from diploid_agent.runtime.event_bus import Event, EventBus
from diploid_agent.task.worker import WorkerPool

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
        on_service_restart: Callable[[str, str], None] | None = None,
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
        self._on_service_restart = on_service_restart

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
                result, log, exit_code, extra = self._execute(task)
                extra_kwargs = extra or {}
                if exit_code == 0:
                    completed = self.plan_manager.complete_task(
                        plan_id, task_id, result=result, log=log, **extra_kwargs
                    )
                    if completed:
                        self.event_bus.post(
                            Event(
                                type="task.completed",
                                payload={
                                    "plan_id": plan_id,
                                    "task_id": task_id,
                                    "result": result,
                                    "log": log,
                                    **extra_kwargs,
                                },
                            )
                        )
                        self._start_ready_dependents(plan_id)
                else:
                    failed = self.plan_manager.fail_task(
                        plan_id, task_id, log=log or result, **extra_kwargs
                    )
                    if failed:
                        self.event_bus.post(
                            Event(
                                type="task.failed",
                                payload={
                                    "plan_id": plan_id,
                                    "task_id": task_id,
                                    "log": log or result,
                                    **extra_kwargs,
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

    def _execute(self, task: Task) -> tuple[str, str, int, dict[str, Any] | None]:
        if task.type == TaskType.SHELL:
            return self._run_shell(task)
        if task.type in (TaskType.ACP, TaskType.SUBAGENT):
            return self._run_acp(task)
        if task.type == TaskType.NOOP:
            return "noop", "", 0, None
        return f"Unknown task type {task.type}", "", -1, None

    def _run_shell(self, task: Task) -> tuple[str, str, int, dict[str, Any] | None]:
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
            return proc.stdout, proc.stderr, proc.returncode, None
        except subprocess.TimeoutExpired:
            return "", f"Timed out after {timeout}s", -1, {"stop_reason": "timeout", "timed_out": True}
        except (OSError, ValueError) as exc:
            return "", str(exc), -1, {"stop_reason": "failed"}

    def _run_acp(self, task: Task) -> tuple[str, str, int, dict[str, Any] | None]:
        prompt_text = task.prompt or task.command
        if not prompt_text:
            return "ACP task has no prompt or command", "", -1, {"stop_reason": "failed"}

        cwd = task.cwd if task.cwd is not None else Path(os.getcwd())

        # ACP tasks are background subagents and must not share the chat's
        # AcpClient / devin acp process. Create a fresh engine per task, unless
        # an explicit non-ACP engine has been injected (e.g. FakeAgentEngine in
        # tests).
        if self.config is None:
            return "ACP engine not configured", "", -1, {"stop_reason": "failed"}

        if self.engine is None or isinstance(self.engine, AcpEngine):
            api_key = self.config.secrets.windsurf_api_key if self.config.secrets else None
            service_name = f"{self.config.persona.name}.service" if self.config.persona else None
            engine = build_engine(
                self.config.engine,
                api_key=api_key,
                service_name=service_name,
                on_service_restart=self._on_service_restart,
            )
        else:
            engine = self.engine

        model = (
            task.acp_model
            if task.acp_model is not None
            else self._task_config.acp_model
            if self._task_config.acp_model is not None
            else self._model
            if self._model is not None
            else self.config.engine.model
        )
        soft_timeout = (
            task.acp_timeout
            if task.acp_timeout is not None
            else self._task_config.acp_timeout
            if self._task_config.acp_timeout is not None
            else self._acp_timeout
            if self._acp_timeout is not None
            else self.config.engine.soft_timeout
        )

        request = TurnRequest(
            prompt=prompt_text,
            cwd=cwd,
            model=model,
            mcp_servers=task.mcp_servers,
            soft_timeout=soft_timeout,
        )
        try:
            turn_result = engine.prompt(request)
        except TimeoutError as exc:
            logger.exception("ACP task %s timed out", task.id)
            return (str(exc), "", -1, {"stop_reason": "timeout", "timed_out": True})
        except (RuntimeError, Exception) as exc:
            logger.exception("ACP task %s failed", task.id)
            return (str(exc), "", -1, {"stop_reason": "failed"})
        finally:
            if engine is not self.engine:
                engine.close()

        stop_reason: str | None = None
        cancelled = turn_result.cancelled and not turn_result.timed_out
        timed_out = turn_result.timed_out
        partial = turn_result.partial

        if timed_out:
            stop_reason = "timeout"
        elif cancelled:
            stop_reason = "cancelled"
        elif turn_result.stop_reason in ("timeout", "cancelled"):
            stop_reason = turn_result.stop_reason
            if stop_reason == "timeout":
                timed_out = True
            else:
                cancelled = True
            partial = True

        log = ""
        if stop_reason == "timeout":
            log = "Subagent stopped early: timed out"
        elif stop_reason == "cancelled":
            log = "Subagent stopped early: cancelled"

        extra: dict[str, Any] | None = None
        if stop_reason is not None or partial or cancelled or timed_out:
            extra = {
                "stop_reason": stop_reason,
                "cancelled": cancelled,
                "partial": partial,
                "timed_out": timed_out,
            }

        return turn_result.reply, log, 0, extra

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
