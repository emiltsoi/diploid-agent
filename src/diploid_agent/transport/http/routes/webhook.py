from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool

from diploid_agent.config import (
    Config,
)
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *


def register_webhook(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.post("/webhook")
    async def telegram_webhook(request: Request) -> dict[str, object]:
        """Minimal Telegram webhook: extracts text and chat_id from update."""
        payload = await request.json()
        message = payload.get("message") or payload.get("edited_message") or {}
        chat = message.get("chat", {})
        chat_id = str(chat.get("id"))
        text = message.get("text", "")

        if not chat_id or not text:
            return {"ok": False, "error": "missing chat_id or text"}

        reply_to = message.get("reply_to_message", {})
        reply_to_text = reply_to.get("text", "") or reply_to.get("caption", "")
        reply_to_is_bot = reply_to.get("from", {}).get("is_bot")
        reply_to_message_id = reply_to.get("message_id")

        result = await run_in_threadpool(
            runtime.process,
            chat_id,
            text,
            reply_to=reply_to_text or None,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
        )
        return {
            "ok": True,
            "reply": result.reply,
            "notice": result.notice,
            "metrics": result.metrics,
        }
