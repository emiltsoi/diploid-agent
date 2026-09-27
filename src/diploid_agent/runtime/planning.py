"""Plan/dispatch wake helpers and continuation message builders."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from diploid_agent.dispatch import Dispatch
from diploid_agent.models import WakeEvent
from diploid_agent.plan.models import Plan, PlanStatus, Task, TaskStatus

if TYPE_CHECKING:
    from diploid_agent.context import ContextBuilder
    from diploid_agent.runtime.wake_queue import WakeQueue


logger = logging.getLogger(__name__)


class RuntimePlanning:
    """Plan task wakes, plan conclusions, and continuation anchors."""

    def __init__(
        self,
        wake_queue: WakeQueue,
        plan_conclusion_enqueued: set[str],
        context_builder: ContextBuilder,
    ) -> None:
        self.wake_queue = wake_queue
        self._plan_conclusion_enqueued = plan_conclusion_enqueued
        self.context_builder = context_builder

    def _enqueue_plan_task_wake(self, plan: Plan, task: Task) -> None:
        """Enqueue a non-silent wake that reports one task's completion or failure."""
        if plan.chat_id is None:
            return
        # Quiet tasks (delivery: silent cron jobs) finish without a wake — a
        # clean bookkeeping run must not page the user. Anything not clean —
        # failed, timed out, partial — still wakes: that is the monitor's job.
        if (
            task.quiet
            and task.status == TaskStatus.DONE
            and not task.timed_out
            and not task.partial
            and not task.cancelled
        ):
            return
        total = len(plan.tasks)
        done = sum(1 for t in plan.tasks if t.status in (TaskStatus.DONE, TaskStatus.INCOMPLETE))
        failed = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
        completed_count = done + failed
        detail = (
            task.result
            if task.status in (TaskStatus.DONE, TaskStatus.INCOMPLETE) and task.result
            else task.log
        )
        payload = {
            "plan_id": plan.id,
            "plan_name": plan.name,
            "plan_status": plan.status.value,
            "task_id": task.id,
            "task_name": task.name,
            "task_status": task.status.value,
            "task_detail": ((detail or "").replace("\n", " ").strip())[:500],
            "tasks": [
                {
                    "task_id": task.id,
                    "task_name": task.name,
                    "task_status": task.status.value,
                    "task_detail": ((detail or "").replace("\n", " ").strip())[:500],
                }
            ],
            "completed_count": completed_count,
            "total_count": total,
            "retry_after": 5.0,
        }
        event = WakeEvent(
            id="",
            chat_id=plan.chat_id,
            reason="plan_task_update",
            priority=1,
            scheduled_at=time.time(),
            payload=payload,
            silent=False,
            created_at=time.time(),
            ready=True,
        )
        # Coalesce: several task wakes pending for the same plan fold into one
        # digest event instead of one ping per task.
        if not self.wake_queue.merge_into_ready(event, key=plan.id):
            self.wake_queue.enqueue(event)

    def _maybe_enqueue_plan_conclusion(self, plan: Plan) -> None:
        """Enqueue a single non-silent wake with the plan's final conclusion."""
        if plan.chat_id is None:
            return
        if plan.status not in (PlanStatus.COMPLETED, PlanStatus.FAILED):
            return
        if plan.id in self._plan_conclusion_enqueued:
            return
        self._plan_conclusion_enqueued.add(plan.id)

        task_lines: list[str] = []
        for t in plan.tasks:
            line = f"- {t.name}: {t.status.value}"
            if t.status in (TaskStatus.DONE, TaskStatus.INCOMPLETE) and t.result:
                line += f" ({t.result[:100].replace(chr(10), ' ').strip()})"
            elif t.status == TaskStatus.FAILED and t.log:
                line += f" ({t.log[:100].replace(chr(10), ' ').strip()})"
            task_lines.append(line)

        payload = {
            "plan_id": plan.id,
            "plan_name": plan.name,
            "plan_status": plan.status.value,
            "tasks_summary": "\n".join(task_lines),
            "retry_after": 5.0,
        }
        self.wake_queue.enqueue(
            WakeEvent(
                id="",
                chat_id=plan.chat_id,
                reason="plan_completed",
                priority=1,
                scheduled_at=time.time(),
                payload=payload,
                silent=False,
                created_at=time.time(),
                ready=True,
            )
        )

    def _build_plan_task_update_message(self, payload: dict[str, Any]) -> str:
        """Format a user message for a single task status update."""
        plan_name = payload.get("plan_name", "unknown")
        plan_status = payload.get("plan_status", "unknown")
        completed = payload.get("completed_count", 0)
        total = payload.get("total_count", 0)
        lines = [
            "[system wake: plan task update]",
            "",
            f"Plan: {plan_name}",
            f"Plan status: {plan_status} ({completed}/{total} tasks finished)",
        ]
        tasks = payload.get("tasks") or []
        if len(tasks) > 1:
            # Coalesced bookkeeping wakes arrive as one digest, not one ping
            # per task.
            lines.append("Tasks:")
            for t in tasks:
                line = f"- {t.get('task_name', 'unknown')}: {t.get('task_status', 'unknown')}"
                if t.get("task_detail"):
                    line += f" ({t['task_detail']})"
                lines.append(line)
            lines.append("")
            lines.append("Summarize these task updates for the user in one line.")
            return "\n".join(lines)
        task_name = payload.get("task_name", "unknown")
        task_status = payload.get("task_status", "unknown")
        task_detail = payload.get("task_detail", "")
        lines.extend([f"Task: {task_name}", f"Task status: {task_status}"])
        if task_detail:
            lines.append(f"Detail: {task_detail}")
        lines.append("")
        lines.append("Briefly report this task's status to the user.")
        return "\n".join(lines)

    def _build_plan_completed_message(self, payload: dict[str, Any]) -> str:
        """Format a user message asking the model to deliver the final conclusion."""
        plan_name = payload.get("plan_name", "unknown")
        plan_status = payload.get("plan_status", "unknown")
        tasks_summary = payload.get("tasks_summary", "")
        lines = [
            "[system wake: plan completed]",
            "",
            f"Plan: {plan_name}",
            f"Final status: {plan_status}",
        ]
        if tasks_summary:
            lines.extend(["", "Task summary:", tasks_summary])
        lines.append("")
        lines.append("Please give a concise final conclusion.")
        return "\n".join(lines)

    def _build_dispatch_continuation(self, dispatch: Dispatch) -> str:
        """Build a continuation anchor for a completed background dispatch."""
        return self.context_builder.build_dispatch_continuation(dispatch)
