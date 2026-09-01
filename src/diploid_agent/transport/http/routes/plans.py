from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, status

from diploid_agent.config import (
    Config,
)
from diploid_agent.plan.models import Task
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *
from diploid_agent.transport.http.utils import _plan_to_response, _task_to_response


def register_plans(app: FastAPI, runtime: RuntimeAPI, command_handler: CommandHandler, config: Config, _require_api_key: Callable[[str | None], None]) -> None:
    @app.post("/plan/create", response_model=PlanResponse, dependencies=[Depends(_require_api_key)])
    def plan_create(req: PlanCreateRequest) -> PlanResponse:
        tasks = []
        for t in req.tasks:
            task_data = t.model_dump()
            task_data["chat_id"] = req.chat_id
            tasks.append(Task.model_validate(task_data))
        try:
            raw = command_handler.call(
                method="plan_create",
                name=req.name,
                description=req.description,
                chat_id=req.chat_id,
                tasks=tasks,
                catch=False,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _plan_to_response(raw)

    @app.post(
        "/plan/task/start", response_model=TaskResponse, dependencies=[Depends(_require_api_key)]
    )
    def plan_task_start(req: PlanTaskStartRequest) -> TaskResponse:
        try:
            raw = command_handler.call(
                method="plan_task_start",
                plan_id=req.plan_id,
                task_id=req.task_id,
                requires_chat_id=False,
                catch=False,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _task_to_response(raw)

    @app.post(
        "/plan/task/done", response_model=TaskResponse, dependencies=[Depends(_require_api_key)]
    )
    def plan_task_done(req: PlanTaskDoneRequest) -> TaskResponse:
        try:
            raw = command_handler.call(
                method="plan_task_done",
                plan_id=req.plan_id,
                task_id=req.task_id,
                result=req.result,
                log=req.log,
                requires_chat_id=False,
                catch=False,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return _task_to_response(raw)

    @app.get("/plan/list", response_model=list[PlanResponse])
    def plan_list() -> list[PlanResponse]:
        try:
            raw = command_handler.call(
                method="plan_list",
                requires_chat_id=False,
                catch=False,
            )
        except AttributeError:
            raw = []
        plans = raw if isinstance(raw, list) else []
        return [_plan_to_response(p) for p in plans]

    @app.get("/plan/{plan_id}", response_model=PlanResponse)
    def plan_get(plan_id: str) -> PlanResponse:
        try:
            raw = command_handler.call(
                method="plan_get",
                plan_id=plan_id,
                requires_chat_id=False,
                catch=False,
            )
        except AttributeError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Plan retrieval not supported",
            )
        if raw is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Plan {plan_id} not found",
            )
        return _plan_to_response(raw)
