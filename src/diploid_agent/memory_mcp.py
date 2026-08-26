"""Stdio MCP server for harness memory.

This server is launched once per ACP session with --chat-id and --harness-url.
It forwards memory tool calls to the harness HTTP endpoints.

Exposed tools:
- memory_recall(query, tags, max_tokens)
- memory_retain(content, tags, context)
- memory_promote(fact)
- memory_status()
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import httpx

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


class MemoryMcpServer:
    def __init__(self, chat_id: str, harness_url: str) -> None:
        self.chat_id = chat_id
        self.harness_url = harness_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.harness_url,
            timeout=30.0,
        )

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            if method == "GET":
                resp = self._client.get(path)
            else:
                resp = self._client.post(path, json=json)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "memory_recall",
                "description": "Recall memories matching a query and optional tags.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "max_tokens": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_retain",
                "description": "Retain a new observation with optional tags.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "context": {"type": "string"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "memory_promote",
                "description": "Promote a fact to the persona memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"fact": {"type": "string"}},
                    "required": ["fact"],
                },
            },
            {
                "name": "memory_status",
                "description": "Return memory backend status.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

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
                    "serverInfo": {"name": "diploid-memory", "version": "0.1.0"},
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
            if name in ("memory_recall", "memory_retain", "memory_promote", "memory_status"):
                return self._call_tool(name, arguments, req_id)
            return _error_response(req_id, f"Unknown tool: {name}")

        return _error_response(req_id, f"Unknown method: {method}")

    def _call_tool(self, name: str, arguments: dict[str, Any], req_id: Any) -> dict[str, Any]:
        if name == "memory_recall":
            body: dict[str, Any] = {
                "chat_id": self.chat_id,
                "query": arguments.get("query", ""),
                "tags": arguments.get("tags", []),
            }
            if "max_tokens" in arguments:
                body["max_tokens"] = int(arguments["max_tokens"])
            result = self._request("POST", "/recall", body)
            text = result.get("reply") or json.dumps(result, ensure_ascii=False)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "memory_retain":
            body = {
                "chat_id": self.chat_id,
                "content": arguments.get("content", ""),
                "tags": arguments.get("tags", []),
            }
            if "context" in arguments:
                body["context"] = arguments["context"]
            result = self._request("POST", "/retain", body)
            text = result.get("reply") or json.dumps(result, ensure_ascii=False)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "memory_promote":
            body = {"chat_id": self.chat_id, "message": arguments.get("fact", "")}
            result = self._request("POST", "/promote", body)
            text = result.get("reply") or json.dumps(result, ensure_ascii=False)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "memory_status":
            result = self._request("GET", f"/status/{self.chat_id}")
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return _tool_result(req_id, text, is_error="error" in result)

        return _error_response(req_id, f"Unknown tool: {name}")

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

    def close(self) -> None:
        self._client.close()


def _write(message: dict[str, Any] | list[dict[str, Any]]) -> None:
    text = json.dumps(message, ensure_ascii=False)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP server for harness memory.")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument(
        "--harness-url",
        default=os.environ.get("DEVIN_HARNESS_URL", "http://127.0.0.1:4003"),
    )
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    server = MemoryMcpServer(chat_id=args.chat_id, harness_url=args.harness_url)
    try:
        server.run()
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
