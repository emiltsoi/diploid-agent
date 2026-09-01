from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request, Response

from diploid_agent.config import (
    Config,
)
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *


def register_mesh(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.post("/ingress/{protocol}")
    async def ingress_route(protocol: str, request: Request) -> Response:
        """Generic pluggable ingress route."""
        return await runtime.handle_ingress(protocol, request)

    @app.post("/mesh/receive")
    async def mesh_receive(request: Request) -> Response:
        """Hermes-compatible mesh receive endpoint."""
        return await runtime.handle_ingress("mesh", request)

    @app.post("/plugins/openclaw-mesh/webhook")
    async def openclaw_mesh_webhook(request: Request) -> Response:
        """OpenClaw-compatible mesh receive alias."""
        return await runtime.handle_ingress("mesh", request)
