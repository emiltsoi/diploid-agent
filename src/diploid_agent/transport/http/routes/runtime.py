from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI

from diploid_agent.config import (
    Config,
)
from diploid_agent.models import RuntimeStatus
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *


def register_runtime(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.post(
        "/runtime/start",
        dependencies=[Depends(_require_api_key)],
    )
    def runtime_start() -> dict[str, bool]:
        command_handler.call(
            method="start",
            requires_chat_id=False,
            catch=False,
        )
        return {"ok": True}

    @app.post(
        "/runtime/stop",
        dependencies=[Depends(_require_api_key)],
    )
    def runtime_stop() -> dict[str, bool]:
        command_handler.call(
            method="shutdown",
            requires_chat_id=False,
            catch=False,
        )
        return {"ok": True}

    @app.get("/runtime/status", response_model=RuntimeStatusResponse)
    def runtime_status() -> RuntimeStatus:
        return command_handler.call(
            method="get_status",
            requires_chat_id=False,
            catch=False,
        )
