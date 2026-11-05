"""WebUI endpoints: session selector, event log, the page itself."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Query
from fastapi.responses import FileResponse

from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler


def register_events(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.get("/chats", dependencies=[Depends(_require_api_key)])
    def chats() -> dict[str, Any]:
        """List known chats for the webui session selector."""
        return command_handler.call(
            method="chats_list",
            http_path="/chats",
            http_method="GET",
            requires_chat_id=False,
            catch=False,
        )

    @app.get("/events/{chat_id}", dependencies=[Depends(_require_api_key)])
    def events(
        chat_id: str,
        since: int = Query(0, ge=0),
        wait: float = Query(0.0, ge=0, le=60, description="Long-poll wait in seconds"),
        limit: int = Query(500, ge=1, le=500),
    ) -> dict[str, Any]:
        """Long-poll the chat's event log for entries with seq > ``since``."""
        return command_handler.call(
            method="events_since",
            chat_id=chat_id,
            since=since,
            wait=wait,
            limit=limit,
            http_path="/events/{chat_id}",
            http_method="GET",
            catch=False,
        )

    # Deliberately unauthenticated: a browser navigation cannot set the
    # X-API-Key header, so a key-gated /ui would 403 before the page's own
    # setup form could collect the key. The file is static and carries no
    # data; every data endpoint above keeps the dependency.
    @app.get("/ui")
    def ui() -> FileResponse:
        """Serve the webui page."""
        page = Path(__file__).resolve().parents[3] / "webui" / "index.html"
        return FileResponse(page, media_type="text/html")
