from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI

from diploid_agent.config import (
    Config,
)
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *
from diploid_agent.transport.http.utils import _to_response


def register_state(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.post("/state", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def state_event(req: StateEventRequest) -> ChatResponse:
        raw = command_handler.call(
            method="plugin_event",
            chat_id=req.chat_id,
            plugin=req.plugin,
            event=req.event,
            catch=False,
            **(req.params or {}),
        )
        return _to_response(raw)

    @app.get("/memory/{chat_id}")
    def memory(chat_id: str) -> dict[str, object]:
        raw = command_handler.call(
            method="memory",
            chat_id=chat_id,
            http_path="/memory/{chat_id}",
            http_method="GET",
            catch=False,
        )
        if isinstance(raw, str):
            return {"chat_id": chat_id, "memory": raw}
        if isinstance(raw, dict):
            return {"chat_id": chat_id, "memory": raw.get("memory", "")}
        return {"chat_id": chat_id, "memory": ""}

    @app.post("/summarize/{chat_id}", dependencies=[Depends(_require_api_key)])
    def summarize(chat_id: str) -> ChatResponse:
        raw = command_handler.call(
            method="summarize",
            chat_id=chat_id,
            http_path="/summarize/{chat_id}",
            catch=False,
        )
        return _to_response(raw)

    @app.post("/recall", dependencies=[Depends(_require_api_key)])
    def recall(req: RecallRequest) -> ChatResponse:
        raw = command_handler.call(
            method="recall",
            chat_id=req.chat_id,
            query=req.query,
            tags=req.tags,
            max_tokens=req.max_tokens,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/retain", dependencies=[Depends(_require_api_key)])
    def retain(req: RetainRequest) -> ChatResponse:
        raw = command_handler.call(
            method="retain",
            chat_id=req.chat_id,
            content=req.content,
            tags=req.tags,
            context=req.context,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/promote", dependencies=[Depends(_require_api_key)])
    def promote(req: ChatRequest) -> ChatResponse:
        raw = command_handler.call(
            method="promote",
            chat_id=req.chat_id,
            fact=req.message,
            http_body={"message": req.message},
            catch=False,
        )
        return _to_response(raw)
