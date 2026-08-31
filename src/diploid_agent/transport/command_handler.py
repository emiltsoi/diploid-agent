"""Unified command handler for HTTP and Telegram transports.

This module centralizes the harness commands that were previously duplicated
between `transport/http.py` and `transport/telegram.py`.  It is the single
source of truth for command parsing, dispatch, and formatting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from diploid_agent.models import ChatResult
from diploid_agent.transport.base import RuntimeAPI

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """One slash-command supported by the harness."""

    name: str
    parser: Callable[[str], dict[str, Any]]
    formatter: Callable[[Any], ChatResult | dict[str, Any]]
    runtime_method: str = ""
    http_path: str = ""
    http_method: str = "POST"
    requires_chat_id: bool = True


class CommandHandler:
    """Parse and execute harness commands.

    When a ``runtime`` is available the handler calls it directly, otherwise it
    calls the equivalent HTTP endpoint.
    """

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
        self._commands: dict[str, Command] = {}
        self._register_default_commands()

    def _register_default_commands(self) -> None:
        self.register(
            Command(
                name="/subagent",
                parser=self._parse_subagent,
                formatter=self._format_chat_result,
                runtime_method="subagent_start",
                http_path="/subagent",
                http_method="POST",
            )
        )

    def register(self, command: Command) -> None:
        self._commands[command.name] = command

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def handle(
        self,
        command: str,
        chat_id: str | None = None,
        arg: str = "",
    ) -> ChatResult | dict[str, Any]:
        """Run a command and return a result suitable for the transport."""
        cmd = self._commands.get(command)
        if cmd is None:
            return ChatResult(reply="Unknown command. Try /help.")

        if command == "/help":
            return self._format_help()

        parsed = cmd.parser(arg)
        if "error" in parsed:
            return ChatResult(reply=parsed["error"])

        raw = self._dispatch(cmd, chat_id, parsed)
        if "error" in raw:
            return ChatResult(reply=raw["error"])

        return cmd.formatter(raw)

    def _http_client(self) -> httpx.Client | None:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider()
        return None

    def _dispatch(
        self,
        cmd: Command,
        chat_id: str | None,
        parsed: dict[str, Any],
    ) -> Any:
        if self.runtime is not None and cmd.runtime_method:
            method = getattr(self.runtime, cmd.runtime_method)
            try:
                kwargs = dict(parsed)
                if cmd.requires_chat_id and chat_id is not None:
                    kwargs["chat_id"] = str(chat_id)
                return method(**kwargs)
            except Exception:
                logger.exception("Command %s runtime call failed", cmd.name)
                return {"error": f"Sorry, I could not run {cmd.name}."}

        client = self._http_client()
        if self.harness_url is None or client is None:
            return {"error": "No runtime or harness URL configured."}

        body = dict(parsed)
        if cmd.requires_chat_id and chat_id is not None:
            body["chat_id"] = str(chat_id)

        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            if cmd.http_method == "GET":
                resp = client.get(f"{self.harness_url}{cmd.http_path}", headers=headers)
            else:
                resp = client.post(
                    f"{self.harness_url}{cmd.http_path}",
                    json=body,
                    headers=headers,
                )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("Command %s HTTP call failed", cmd.name)
            return {"error": f"Sorry, the harness call for {cmd.name} failed."}

    @staticmethod
    def _parse_subagent(arg: str) -> dict[str, Any]:
        if not arg.strip():
            return {"error": "Usage: /subagent <prompt>"}
        return {"prompt": arg}

    @staticmethod
    def _format_chat_result(raw: Any) -> ChatResult:
        if isinstance(raw, ChatResult):
            return raw
        if isinstance(raw, dict):
            return ChatResult(
                reply=raw.get("reply", ""),
                notice=raw.get("notice"),
                dispatch_id=raw.get("dispatch_id"),
            )
        return ChatResult(reply=str(raw))

    def _format_help(self) -> ChatResult:
        return ChatResult(reply="Available commands: " + ", ".join(self._commands.keys()))
