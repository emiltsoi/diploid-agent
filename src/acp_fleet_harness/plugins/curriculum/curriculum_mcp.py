"""Stdio MCP server for the curriculum plugin."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.plugins.curriculum.curriculum import CurriculumPlugin

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


class CurriculumMcpServer:
    """Minimal stdio MCP server backed by CurriculumPlugin."""

    def __init__(self, chat_id: str, sessions_root: Path, state_file: str) -> None:
        config = PluginConfig(
            name="curriculum",
            state_file=state_file,
            prompt_slot="persona_state",
        )
        self.plugin = CurriculumPlugin(config, chat_id, sessions_root)

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "curriculum_add_word",
                "description": "Add a word and its translation to the vocabulary.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "translation": {"type": "string"},
                    },
                    "required": ["word", "translation"],
                },
            },
            {
                "name": "curriculum_set_target_language",
                "description": "Set the target language for this chat.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "language": {"type": "string"},
                    },
                    "required": ["language"],
                },
            },
            {
                "name": "curriculum_set_unit",
                "description": "Set the current unit/topic for this chat.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "unit": {"type": "string"},
                    },
                    "required": ["unit"],
                },
            },
            {
                "name": "curriculum_state",
                "description": "Return the current curriculum prompt block.",
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
                    "serverInfo": {"name": "devin-curriculum", "version": "0.1.0"},
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
            if name == "curriculum_add_word":
                word = arguments.get("word", "")
                translation = arguments.get("translation", "")
                return _tool_result(req_id, self.plugin._add_word(word, translation))
            if name == "curriculum_set_target_language":
                return _tool_result(
                    req_id, self.plugin._set_target_language(arguments.get("language", ""))
                )
            if name == "curriculum_set_unit":
                return _tool_result(req_id, self.plugin._set_unit(arguments.get("unit", "")))
            if name == "curriculum_state":
                return _tool_result(req_id, self.plugin.prompt_block() or "No curriculum set.")
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
    parser = argparse.ArgumentParser(description="MCP server for curriculum state.")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--sessions-root", default="sessions")
    parser.add_argument("--state-file", default="chat_curriculum.json")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)

    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    sessions_root = Path(args.sessions_root).expanduser().resolve()
    server = CurriculumMcpServer(
        chat_id=args.chat_id,
        sessions_root=sessions_root,
        state_file=args.state_file,
    )
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
