"""Shared helpers for the HTTP transport route modules."""

from __future__ import annotations

from typing import Any

from diploid_agent.plan.models import Plan, Task
from diploid_agent.transport.http.models import ChatResponse, PlanResponse, TaskResponse


def _to_response(result: Any) -> ChatResponse:
    """Convert a ChatResult or bare reply string into a ChatResponse."""
    if isinstance(result, str):
        return ChatResponse(reply=result)
    return ChatResponse(
        reply=result.reply,
        notice=result.notice,
        continuation=result.continuation,
        dispatch_id=result.dispatch_id,
        session_id=result.session_id,
        session_number=result.session_number,
        turn_number=result.turn_number,
        metrics=result.metrics,
    )


def _task_to_response(task: Task) -> TaskResponse:
    return TaskResponse(
        id=task.id,
        name=task.name,
        type=task.type.value,
        status=task.status.value,
        result=task.result,
        log=task.log,
        acp_model=task.acp_model,
    )


def _plan_to_response(plan: Plan) -> PlanResponse:
    return PlanResponse(
        id=plan.id,
        name=plan.name,
        status=plan.status.value,
        chat_id=plan.chat_id,
        tasks=[_task_to_response(t) for t in plan.tasks],
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )
