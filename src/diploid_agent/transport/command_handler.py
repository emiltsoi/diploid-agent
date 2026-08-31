"""Unified command dispatch for HTTP and Telegram transports.

This module centralizes the *call* side of harness commands: given the name of
a `RuntimeAPI` method, the matching HTTP endpoint, and any arguments, it calls
the runtime directly when available and falls back to the HTTP harness when it
is not.  Formatting is intentionally left to each transport so the Telegram
poller and the HTTP endpoints can present results differently.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import httpx

from diploid_agent.models import ChatResult
from diploid_agent.transport.base import RuntimeAPI

logger = logging.getLogger(__name__)


def _coerce_chat_result(raw: Any) -> ChatResult:
    """Normalize a command result to a `ChatResult`."""
    if isinstance(raw, ChatResult):
        return raw
    if isinstance(raw, dict):
        return ChatResult(
            reply=raw.get("reply", ""),
            notice=raw.get("notice"),
            dispatch_id=raw.get("dispatch_id"),
        )
    return ChatResult(reply=str(raw))


class CommandHandler:
    """Dispatch a command to the runtime or the equivalent HTTP endpoint."""

    def __init__(
        self,
        *,
        runtime: RuntimeAPI | None = None,
        harness_url: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        client_provider: Callable[[], httpx.Client] | None = None,
    ) -> None:
        self.runtime = runtime
        self.harness_url = harness_url.rstrip("/") if harness_url else None
        self.api_key = api_key
        self.client = client
        self.client_provider = client_provider

    def _http_client(self) -> httpx.Client | None:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider()
        return None

    def call(
        self,
        *,
        method: str,
        chat_id: str | int | None = None,
        http_path: str = "",
        http_method: str = "POST",
        **kwargs: Any,
    ) -> Any:
        """Call a runtime method or the matching HTTP endpoint.

        For ``GET`` requests, ``http_path`` may contain a ``{chat_id}``
        placeholder that is filled from ``chat_id``.  For ``POST`` requests the
        body is built from ``kwargs`` plus ``chat_id``.
        """
        if self.runtime is not None:
            try:
                fn = getattr(self.runtime, method)
                call_kwargs = dict(kwargs)
                if chat_id is not None:
                    call_kwargs["chat_id"] = str(chat_id)
                return fn(**call_kwargs)
            except Exception:
                logger.exception("Runtime %s failed", method)
                return {"error": f"Sorry, I could not run {method}."}

        client = self._http_client()
        if self.harness_url is None or client is None:
            return {"error": "No runtime or harness URL configured."}

        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            if http_method == "GET":
                path = http_path.format(chat_id=str(chat_id)) if "{chat_id}" in http_path else http_path
                resp = client.get(f"{self.harness_url}{path}", headers=headers)
            else:
                body = dict(kwargs)
                if chat_id is not None:
                    body["chat_id"] = str(chat_id)
                resp = client.post(
                    f"{self.harness_url}{http_path}",
                    json=body,
                    headers=headers,
                )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("HTTP %s %s failed", http_method, http_path)
            return {"error": f"Sorry, the harness call for {method} failed."}

    def handle(self, command: str, chat_id: str | int, arg: str = "") -> ChatResult:
        """Telegram-style command: parse and return a `ChatResult`."""
        if command == "/subagent":
            if not arg.strip():
                return ChatResult(reply="Usage: /subagent <prompt>")
            raw = self.call(
                method="subagent_start",
                chat_id=chat_id,
                http_path="/subagent",
                prompt=arg,
            )
            return _coerce_chat_result(raw)

        return ChatResult(reply=f"Unknown command {command}. Try /help.")
