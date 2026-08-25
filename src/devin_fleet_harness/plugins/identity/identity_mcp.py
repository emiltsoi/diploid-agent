"""Stdio MCP server for the identity/self-narrative plugin."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from devin_fleet_harness.config import PluginConfig
from devin_fleet_harness.plugins.identity.identity import IdentityPlugin

DEFAULT_PROTOCOL_VERSION = "2024-11-05"


def _error_response(req_id: Any, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32602, "message": message},
    }


def _tool_result(req_id: Any, text: str, is_error: bool = False) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        },
    }


class IdentityMcpServer:
    """Minimal stdio MCP server backed by IdentityPlugin."""

    def __init__(self, chat_id: str, sessions_root: Path, state_file: str) -> None:
        config = PluginConfig(
            name="identity",
            state_file=state_file,
            prompt_slot="self_narrative",
            skill="identity",
        )
        self.plugin = IdentityPlugin(config, chat_id, sessions_root)

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "identity_update",
                "description": "Update the chat-scoped self-narrative.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "reason": {"type": "string", "default": "manual"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "identity_state",
                "description": "Return the current self-narrative prompt block.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "identity_history",
                "description": "Return recent self-narrative edits.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": [],
                },
            },
            {
                "name": "identity_clear",
                "description": "Clear the chat-scoped self-narrative.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def _handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            protocol_version = params.get("protocolVersion", DEFAULT_PROTOCOL_VERSION)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "devin-identity", "version": "0.1.0"},
                },
            }

        if method == "notifications/initialized":
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self._tools()}}

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name == "identity_update":
                content = arguments.get("content", "")
                reason = arguments.get("reason", "manual")
                return _tool_result(req_id, self.plugin._update(content, reason))
            if name == "identity_state":
                return _tool_result(req_id, self.plugin.prompt_block() or "No self-narrative set.")
            if name == "identity_history":
                limit = arguments.get("limit", 10)
                records = self.plugin._history(limit)
                return _tool_result(req_id, json.dumps(records, default=str, indent=2))
            if name == "identity_clear":
                return _tool_result(req_id, self.plugin._clear())
            return _error_response(req_id, f"Unknown tool: {name}")

        return _error_response(req_id, f"Unknown method: {method}")

    def run(self) -> None:
        while True:
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                break
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                _write(_error_response(None, f"Invalid JSON: {exc}"))
                continue

            if isinstance(request, list):
                response = [self._handle(r) for r in request]
                response = [r for r in response if r is not None]
                if response:
                    _write(response)
            else:
                response = self._handle(request)
                if response is not None:
                    _write(response)


def _write(message: dict[str, Any] | list[dict[str, Any]]) -> None:
    text = json.dumps(message, ensure_ascii=False)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP server for identity state.")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--sessions-root", default="sessions")
    parser.add_argument("--state-file", default="chat_SELF.md")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    sessions_root = Path(args.sessions_root).expanduser().resolve()
    server = IdentityMcpServer(
        chat_id=args.chat_id,
        sessions_root=sessions_root,
        state_file=args.state_file,
    )
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
