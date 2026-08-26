"""Reusable base class for stdio MCP servers."""

from __future__ import annotations

import json
import sys
from typing import Any

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


class StdioMcpServer:
    """Minimal stdio MCP server. Subclasses implement _tools() and _call_tool()."""

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version

    def _tools(self) -> list[dict[str, Any]]:
        return []

    def _call_tool(
        self, name: str, arguments: dict[str, Any], req_id: Any
    ) -> dict[str, Any]:
        return _error_response(req_id, f"Tool {name} not implemented")

    def _handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            protocol = params.get("protocolVersion", DEFAULT_PROTOCOL_VERSION)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": protocol,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": self.name, "version": self.version},
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
            return self._call_tool(name, arguments, req_id)

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


if __name__ == "__main__":
    StdioMcpServer("noop").run()
