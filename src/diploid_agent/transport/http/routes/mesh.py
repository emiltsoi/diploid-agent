from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, FastAPI, Request, Response, status

from diploid_agent.config import (
    Config,
)
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import (
    MeshChatMapRequest,
    MeshNotifyRequest,
)


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

    @app.post(
        "/mesh/chat-map",
        dependencies=[Depends(_require_api_key)],
    )
    async def mesh_chat_map(req: MeshChatMapRequest) -> dict:
        """Update the mesh chat mapping for sessions or per-sender routing."""
        return command_handler.call(
            method="update_mesh_chat_map",
            chat_map=req.chat_map,
            chat_mapping=req.chat_mapping,
            fallback_chat_id=req.fallback_chat_id,
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/mesh/{chat_id}/notify",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(_require_api_key)],
    )
    async def mesh_notify(chat_id: str, req: MeshNotifyRequest) -> dict[str, str]:
        """Mirror a successfully sent mesh message to Telegram as a system notice."""
        runtime._float_mesh_to_telegram(
            chat_id,
            sender=req.sender,
            recipient=req.recipient,
            body=req.body,
            action=req.action,
            reply=req.reply,
            msg_id=req.msg_id,
        )
        return {"status": "accepted"}
