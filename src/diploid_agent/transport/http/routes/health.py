from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI
from starlette.responses import PlainTextResponse

from diploid_agent.config import (
    Config,
)
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *


def register_health(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.get("/health")
    def health() -> dict[str, Any]:
        return command_handler.call(
            method="health",
            requires_chat_id=False,
            catch=False,
        )

    @app.get("/prometheus", dependencies=[Depends(_require_api_key)])
    def prometheus() -> PlainTextResponse:
        raw = command_handler.call(
            method="get_prometheus_metrics",
            requires_chat_id=False,
            catch=False,
        )
        return PlainTextResponse(raw)

    @app.get("/metrics", dependencies=[Depends(_require_api_key)])
    def metrics() -> dict[str, Any]:
        return command_handler.call(
            method="get_metrics",
            http_path="/metrics",
            http_method="GET",
            requires_chat_id=False,
            catch=False,
        )

    @app.get("/metrics/{chat_id}", dependencies=[Depends(_require_api_key)])
    def chat_metrics(chat_id: str) -> dict[str, Any]:
        return command_handler.call(
            method="get_metrics",
            chat_id=chat_id,
            http_path="/metrics/{chat_id}",
            http_method="GET",
            catch=False,
        )
