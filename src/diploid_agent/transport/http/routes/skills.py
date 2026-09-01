from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI

from diploid_agent.config import (
    Config,
)
from diploid_agent.models import ChatResult
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *
from diploid_agent.transport.http.utils import _to_response


def register_skills(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.get("/skill/{chat_id}")
    def skill_get(chat_id: str) -> ChatResponse:
        raw = command_handler.call(
            method="skill_list",
            chat_id=chat_id,
            http_path="/skill/{chat_id}",
            http_method="GET",
            catch=False,
        )
        if isinstance(raw, dict):
            return ChatResponse(reply=raw.get("reply", ""))
        return ChatResponse(reply=raw)

    @app.post("/skill", dependencies=[Depends(_require_api_key)])
    def skill_post(req: SkillCommandRequest) -> ChatResponse:
        if req.command == "list":
            raw = command_handler.call(
                method="skill_list",
                chat_id=req.chat_id,
                http_path="/skill/{chat_id}",
                http_method="GET",
                catch=False,
            )
        elif req.command == "enable" and req.name:
            raw = command_handler.call(
                method="skill_enable",
                chat_id=req.chat_id,
                name=req.name,
                catch=False,
            )
        elif req.command == "disable" and req.name:
            raw = command_handler.call(
                method="skill_disable",
                chat_id=req.chat_id,
                name=req.name,
                catch=False,
            )
        elif req.command == "create" and req.name and req.content:
            raw = command_handler.call(
                method="skill_create",
                chat_id=req.chat_id,
                name=req.name,
                content=req.content,
                catch=False,
            )
        else:
            return ChatResponse(
                reply="Usage: command=list|enable|disable|create, name and content required for create"
            )
        if isinstance(raw, ChatResult):
            return _to_response(raw)
        if isinstance(raw, dict):
            return ChatResponse(reply=raw.get("reply", ""))
        return ChatResponse(reply=str(raw))
