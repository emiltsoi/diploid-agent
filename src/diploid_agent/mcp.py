"""MCP server configuration manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diploid_agent.config import Config, McpServerConfig


@dataclass
class _PlaceholderFormatter:
    """Format placeholders like {chat_id}, {sessions_root}, {harness_url}."""

    chat_id: str
    sessions_root: Path
    harness_url: str = ""

    def _safe_chat_dir(self) -> str:
        return str(self.sessions_root / self.chat_id.replace("/", "_"))

    def format(self, text: str) -> str:
        out: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == "{" and i + 1 < n and text[i + 1] == "{":
                out.append("{")
                i += 2
                continue
            if text[i] == "}" and i + 1 < n and text[i + 1] == "}":
                out.append("}")
                i += 2
                continue
            if text[i] == "{":
                end = text.find("}", i + 1)
                if end == -1:
                    out.append(text[i])
                    i += 1
                    continue
                key = text[i + 1 : end]
                if key == "chat_id":
                    out.append(self.chat_id)
                elif key == "sessions_root":
                    out.append(str(self.sessions_root))
                elif key == "chat_dir":
                    out.append(self._safe_chat_dir())
                elif key == "harness_url":
                    out.append(self.harness_url)
                else:
                    out.append(text[i : end + 1])
                i = end + 1
                continue
            out.append(text[i])
            i += 1
        return "".join(out)


@dataclass
class McpManager:
    """Resolve configured MCP servers per chat."""

    config: Config
    _chat_enabled: dict[str, set[str]] = field(default_factory=dict)

    def _server_by_name(self, name: str) -> McpServerConfig | None:
        for server in self.config.harness.mcp.servers:
            if server.name == name:
                return server
        return None

    @property
    def _sessions_root(self) -> Path:
        return Path(self.config.harness.sessions_root).expanduser().resolve()

    @property
    def _harness_url(self) -> str:
        return f"http://{self.config.harness.listen_host}:{self.config.harness.listen_port}"

    def _render_server(
        self,
        server: McpServerConfig,
        chat_id: str,
    ) -> dict[str, Any]:
        fmt = _PlaceholderFormatter(
            chat_id=chat_id,
            sessions_root=self._sessions_root,
            harness_url=self._harness_url,
        )
        env = [fmt.format(e) for e in server.env]
        # Pass the harness API key to child MCP processes so they can call back.
        harness_api_key = (
            self.config.secrets.harness_api_key if self.config.secrets else None
        )
        if harness_api_key:
            env.append(f"HARNESS_API_KEY={harness_api_key}")
        return {
            "name": server.name,
            "command": fmt.format(server.command),
            "args": [fmt.format(a) for a in server.args],
            "env": env,
        }

    def default_enabled_names(self) -> set[str]:
        return set(self.config.harness.mcp.default_enabled or [])

    def enabled_servers(
        self,
        chat_id: str,
        enabled_names: set[str] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the ACP `mcpServers` list for this chat."""
        if enabled_names is None:
            enabled_names = self._chat_enabled.get(chat_id, self.default_enabled_names())
        names = set(enabled_names)

        out: list[dict[str, Any]] = []
        for server in self.config.harness.mcp.servers:
            if server.disabled and server.name in names:
                continue
            if server.name in names:
                out.append(self._render_server(server, chat_id))
        return out

    def list_servers(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "command": s.command,
                "args": s.args,
                "env": s.env,
                "disabled": s.disabled,
            }
            for s in self.config.harness.mcp.servers
        ]

    def set_chat_enabled(self, chat_id: str, names: set[str] | list[str]) -> None:
        self._chat_enabled[chat_id] = set(names)
