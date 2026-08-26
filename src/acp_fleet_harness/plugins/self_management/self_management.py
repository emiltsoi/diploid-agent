"""Self-management plugin: exposes an MCP server for plugin lifecycle."""

from __future__ import annotations

from acp_fleet_harness.config import McpServerConfig
from acp_fleet_harness.plugins.base import StatePlugin


class SelfManagementPlugin(StatePlugin):
    """No prompt block; only provides the devin-self-management MCP server."""

    def mcp_server(self) -> McpServerConfig | None:
        return self.config.mcp_server
