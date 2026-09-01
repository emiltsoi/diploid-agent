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


def register_models(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.get("/models")
    def models() -> dict[str, list[str]]:
        raw = command_handler.call(
            method="list_models",
            http_path="/models",
            http_method="GET",
            requires_chat_id=False,
            catch=False,
        )
        if isinstance(raw, list):
            return {"models": raw}
        return raw

    @app.post(
        "/switch-model", response_model=ChatResponse, dependencies=[Depends(_require_api_key)]
    )
    def switch_model(req: SwitchModelRequest) -> ChatResponse:
        raw = command_handler.call(
            method="switch_model",
            chat_id=req.chat_id,
            model=req.model,
            catch=False,
        )
        return _to_response(raw)

    @app.get("/mcp/{chat_id}")
    def mcp_get(chat_id: str) -> ChatResponse:
        raw = command_handler.call(
            method="mcp_list",
            chat_id=chat_id,
            http_path="/mcp/{chat_id}",
            http_method="GET",
            catch=False,
        )
        if isinstance(raw, dict):
            return ChatResponse(reply=raw.get("reply", ""))
        return ChatResponse(reply=raw)

    @app.post("/mcp", dependencies=[Depends(_require_api_key)])
    def mcp_post(req: McpCommandRequest) -> ChatResponse:
        if req.command == "list":
            raw = command_handler.call(
                method="mcp_list",
                chat_id=req.chat_id,
                http_path="/mcp/{chat_id}",
                http_method="GET",
                catch=False,
            )
        elif req.command == "enable" and req.name:
            raw = command_handler.call(
                method="mcp_enable",
                chat_id=req.chat_id,
                name=req.name,
                catch=False,
            )
        elif req.command == "disable" and req.name:
            raw = command_handler.call(
                method="mcp_disable",
                chat_id=req.chat_id,
                name=req.name,
                catch=False,
            )
        else:
            return ChatResponse(
                reply="Usage: command=list|enable|disable, name required for enable/disable"
            )
        if isinstance(raw, ChatResult):
            return _to_response(raw)
        if isinstance(raw, dict):
            return ChatResponse(reply=raw.get("reply", ""))
        return ChatResponse(reply=str(raw))
