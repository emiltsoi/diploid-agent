from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Query

from diploid_agent.config import (
    Config,
)
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *
from diploid_agent.transport.http.utils import _to_response


def register_sessions(app: FastAPI, runtime: RuntimeAPI, command_handler: CommandHandler, config: Config, _require_api_key: Callable[[str | None], None]) -> None:
    @app.post(
        "/new/{chat_id}", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def new_session(chat_id: str) -> ChatResponse:
        raw = command_handler.call(
            method="new_session",
            chat_id=chat_id,
            http_path="/new/{chat_id}",
            catch=False,
        )
        return _to_response(raw)

    @app.get("/sessions/{chat_id}")
    def sessions(chat_id: str) -> dict[str, Any]:
        return command_handler.call(
            method="list_sessions",
            chat_id=chat_id,
            http_path="/sessions/{chat_id}",
            http_method="GET",
            catch=False,
        )

    @app.get("/subagents/{chat_id}")
    def subagents(chat_id: str) -> dict[str, Any]:
        return command_handler.call(
            method="subagent_status",
            chat_id=chat_id,
            http_path="/subagents/{chat_id}",
            http_method="GET",
            catch=False,
        )

    @app.get("/outbox/{chat_id}", response_model=OutboxResponse)
    def outbox(chat_id: str, wait: float = Query(0.0, ge=0, le=60)) -> OutboxResponse:
        raw = command_handler.call(
            method="outbox_pop",
            chat_id=chat_id,
            http_path="/outbox/{chat_id}",
            http_method="GET",
            wait=wait,
            catch=False,
        )
        if raw is None:
            return OutboxResponse(chat_id=chat_id, result=None)
        return OutboxResponse(chat_id=chat_id, result=_to_response(raw))

    @app.post("/resume", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def resume(req: ResumeRequest) -> ChatResponse:
        raw = command_handler.call(
            method="resume_session",
            chat_id=req.chat_id,
            session_number=req.session_number,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/branch", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def branch(req: BranchRequest) -> ChatResponse:
        raw = command_handler.call(
            method="branch_session",
            chat_id=req.chat_id,
            session_number=req.session_number,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/subagent",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def subagent(req: SubagentRequest) -> ChatResponse:
        raw = command_handler.call(
            method="subagent_start",
            chat_id=req.chat_id,
            prompt=req.prompt,
            context=req.context,
            model=req.model,
            cwd=Path(req.cwd) if req.cwd else None,
            acp_timeout=req.acp_timeout,
            catch=False,
        )
        return _to_response(raw)
