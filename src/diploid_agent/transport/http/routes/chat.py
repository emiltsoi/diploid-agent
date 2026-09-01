from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status

from diploid_agent.config import (
    Config,
)
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *
from diploid_agent.transport.http.utils import _to_response


def register_chat(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def chat(req: ChatRequest) -> ChatResponse:
        # The caller (e.g. the Telegram long-polling bot) is responsible for
        # delivering the reply, so suppress the runtime's own outbound notifier.
        result = runtime.process(
            req.chat_id,
            req.message,
            model=req.model,
            reply_to=req.reply_to,
            reply_to_is_bot=req.reply_to_is_bot,
            reply_to_message_id=req.reply_to_message_id,
            notify=False,
        )
        return _to_response(result)

    @app.post("/dispatch", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def dispatch(req: DispatchRequest) -> ChatResponse:
        raw = command_handler.call(
            method="dispatch",
            chat_id=req.chat_id,
            context=req.context,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/continue", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def continue_turn(req: ContinueRequest) -> ChatResponse:
        raw = command_handler.call(
            method="continue_turn",
            dispatch_id=req.dispatch_id,
            result=req.result,
            requires_chat_id=False,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/wake", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def wake(req: WakeRequest) -> ChatResponse:
        raw = command_handler.call(
            method="wake",
            chat_id=req.chat_id,
            event_id=req.event_id,
            reason=req.reason,
            silent=req.silent,
            catch=False,
        )
        if raw.reply == "Chat is busy; wake re-enqueued.":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=raw.reply,
            )
        if raw.reply == "Unknown or already completed wake event.":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=raw.reply,
            )
        if req.event_id is not None:
            runtime.wake_queue.complete(req.event_id)
        return _to_response(raw)

    @app.post("/stop", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def stop(req: StopRequest) -> ChatResponse:
        raw = command_handler.call(
            method="stop",
            chat_id=req.chat_id,
            catch=False,
        )
        return _to_response(raw)

    @app.post("/restart", response_model=ChatResponse, dependencies=[Depends(_require_api_key)])
    def restart(req: RestartRequest) -> ChatResponse:
        raw = command_handler.call(
            method="restart",
            chat_id=req.chat_id,
            catch=False,
        )
        return _to_response(raw)

    @app.post(
        "/graceful-restart",
        response_model=ChatResponse,
        dependencies=[Depends(_require_api_key)],
    )
    def graceful_restart(req: GracefulRestartRequest) -> ChatResponse:
        raw = command_handler.call(
            method="graceful_service_restart",
            chat_id=req.chat_id,
            service=req.service,
            reason=req.reason,
            catch=False,
        )
        return _to_response(raw)

    @app.get("/turn/{chat_id}")
    def turn_status(
        chat_id: str, wait: float = Query(0.0, ge=0, le=60, description="Long-poll wait in seconds")
    ) -> dict[str, Any]:
        """Return the partial state of an active turn for streaming clients."""
        return command_handler.call(
            method="turn_status",
            chat_id=chat_id,
            wait=wait,
            http_path="/turn/{chat_id}",
            http_method="GET",
            catch=False,
        )

    @app.get("/status/{chat_id}")
    def chat_status(chat_id: str) -> dict[str, object]:
        return command_handler.call(
            method="status",
            chat_id=chat_id,
            http_path="/status/{chat_id}",
            http_method="GET",
            catch=False,
        )
