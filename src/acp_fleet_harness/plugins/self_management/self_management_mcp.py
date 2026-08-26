"""Stdio MCP server for plugin self-management."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from acp_fleet_harness.mcp_stdio import StdioMcpServer, _error_response, _tool_result
from acp_fleet_harness.plugins.self_management.approvals import ApprovalStore


class SelfManagementMcpServer(StdioMcpServer):
    def __init__(
        self,
        chat_id: str,
        harness_url: str,
        sessions_root: Path,
        require_approval: bool = True,
        approval_timeout: float = 300.0,
    ) -> None:
        super().__init__("devin-self-management", "0.1.0")
        self.chat_id = chat_id
        self.harness_url = harness_url.rstrip("/")
        self.sessions_root = Path(sessions_root)
        self.require_approval = require_approval
        self.approval_timeout = approval_timeout
        self._client = httpx.Client(
            base_url=self.harness_url,
            timeout=60.0,
        )
        self._api_key = os.environ.get("HARNESS_API_KEY")
        self._approvals = ApprovalStore(
            self._chat_dir() / "chat_plugin_approvals.json",
            timeout_seconds=approval_timeout,
        )

    def _chat_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_")

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            if method == "GET":
                resp = self._client.get(path, headers=self._headers())
            elif method == "DELETE":
                resp = self._client.delete(path, headers=self._headers())
            else:
                resp = self._client.post(path, json=body, headers=self._headers())
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _check_approval(self, token: str | None) -> tuple[bool, str]:
        if not self.require_approval:
            return True, ""
        if not token:
            return False, "Approval token required. Call plugin_propose first."
        if not self._approvals.is_approved(token):
            return False, f"Token {token} is not approved or has expired."
        return True, ""

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "plugin_list",
                "description": "List currently loaded plugins.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "plugin_status",
                "description": "Return harness health and plugin status.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "plugin_sandbox",
                "description": "Validate a plugin module in isolation before adding it.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "module": {"type": "string"},
                        "plugin": {"type": "object"},
                    },
                    "required": ["module"],
                },
            },
            {
                "name": "plugin_create",
                "description": "Scaffold a new plugin module, validate it in the sandbox, and return a config ready for plugin_add.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "module": {"type": "string"},
                        "prompt_slot": {"type": "string"},
                        "state_file": {"type": "string"},
                        "mcp_server": {"type": "object"},
                        "config": {"type": "object"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "plugin_propose",
                "description": "Request approval for a plugin mutation. Returns a token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "plugin": {"type": "object"},
                    },
                    "required": ["operation"],
                },
            },
            {
                "name": "plugin_approve",
                "description": "Approve a pending plugin mutation token.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                },
            },
            {
                "name": "plugin_add",
                "description": "Add a plugin to the live config. Requires an approved token unless approval is disabled.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plugin": {"type": "object"},
                        "approval_token": {"type": "string"},
                        "dry_run": {"type": "boolean", "default": False},
                    },
                    "required": ["plugin"],
                },
            },
            {
                "name": "plugin_remove",
                "description": "Remove a plugin. Requires an approved token unless approval is disabled.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "approval_token": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "plugin_toggle",
                "description": "Enable or disable a plugin globally. Requires an approved token unless approval is disabled.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "enabled": {"type": "boolean"},
                        "approval_token": {"type": "string"},
                    },
                    "required": ["name", "enabled"],
                },
            },
            {
                "name": "plugin_rollback",
                "description": "Roll back the last N plugin config changes. Requires an approved token unless approval is disabled.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "steps": {"type": "integer", "default": 1},
                        "approval_token": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "plugin_incidents",
                "description": "Return recent plugin failures and recovery actions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plugin": {"type": "string"},
                    },
                    "required": [],
                },
            },
        ]

    def _call_tool(self, name: str, arguments: dict[str, Any], req_id: Any) -> dict[str, Any]:
        if name == "plugin_list":
            result = self._request("GET", f"/plugins/{self.chat_id}")
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "plugin_status":
            result = self._request("GET", "/health")
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "plugin_sandbox":
            body = {
                "module": arguments.get("module"),
                "plugin": arguments.get("plugin") or {},
            }
            result = self._request("POST", "/plugin/sandbox", body)
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "plugin_create":
            body = {
                "name": arguments.get("name"),
                "module": arguments.get("module"),
                "prompt_slot": arguments.get("prompt_slot", "persona_state"),
                "state_file": arguments.get("state_file"),
                "mcp_server": arguments.get("mcp_server"),
                "config": arguments.get("config"),
            }
            result = self._request("POST", "/plugins/create", body)
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "plugin_propose":
            token = self._approvals.propose(
                arguments.get("operation", "unknown"),
                arguments.get("plugin", {}),
            )
            return _tool_result(
                req_id,
                f"Proposed. Token: {token}. Say 'approve {token}' to allow execution.",
            )

        if name == "plugin_approve":
            token = arguments.get("token", "")
            if self._approvals.approve(token):
                return _tool_result(req_id, f"Token {token} approved.")
            return _error_response(req_id, f"Token {token} not found or expired.")

        approval_token = arguments.get("approval_token")

        if name == "plugin_add":
            if arguments.get("dry_run"):
                ok, msg = True, ""
            else:
                ok, msg = self._check_approval(approval_token)
            if not ok:
                return _error_response(req_id, msg)
            body = {
                "plugin": arguments.get("plugin", {}),
                "dry_run": arguments.get("dry_run", False),
            }
            result = self._request("POST", "/plugins", body)
            text = result.get("reply") or json.dumps(result, ensure_ascii=False)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "plugin_remove":
            ok, msg = self._check_approval(approval_token)
            if not ok:
                return _error_response(req_id, msg)
            plugin_name = arguments.get("name", "")
            result = self._request("DELETE", f"/plugins/{plugin_name}")
            text = result.get("reply") or json.dumps(result, ensure_ascii=False)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "plugin_toggle":
            ok, msg = self._check_approval(approval_token)
            if not ok:
                return _error_response(req_id, msg)
            plugin_name = arguments.get("name", "")
            body = {
                "name": plugin_name,
                "enabled": arguments.get("enabled", True),
            }
            result = self._request("POST", f"/plugins/{plugin_name}/toggle", body)
            text = result.get("reply") or json.dumps(result, ensure_ascii=False)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "plugin_rollback":
            ok, msg = self._check_approval(approval_token)
            if not ok:
                return _error_response(req_id, msg)
            body = {"steps": arguments.get("steps", 1)}
            result = self._request("POST", "/config/rollback", body)
            text = result.get("reply") or json.dumps(result, ensure_ascii=False)
            return _tool_result(req_id, text, is_error="error" in result)

        if name == "plugin_incidents":
            plugin = arguments.get("plugin")
            path = f"/plugin-incidents/{plugin}" if plugin else "/plugin-incidents"
            result = self._request("GET", path)
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return _tool_result(req_id, text, is_error="error" in result)

        return _error_response(req_id, f"Unknown tool: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP server for plugin self-management.")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--harness-url", default="http://127.0.0.1:4003")
    parser.add_argument("--sessions-root", default="sessions")
    parser.add_argument("--require-approval", type=lambda s: s.lower() in ("1", "true"), default="true")
    parser.add_argument("--approval-timeout", type=float, default=300.0)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args(argv)

    if args.log_file:
        import logging

        logging.basicConfig(
            filename=args.log_file,
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    server = SelfManagementMcpServer(
        chat_id=args.chat_id,
        harness_url=args.harness_url,
        sessions_root=Path(args.sessions_root).expanduser().resolve(),
        require_approval=args.require_approval,
        approval_timeout=args.approval_timeout,
    )
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
