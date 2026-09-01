"""Tests for MCP configuration manager."""

from pathlib import Path

from diploid_agent.config import Config, McpConfig, McpServerConfig
from diploid_agent.mcp import McpManager


def test_mcp_manager_returns_enabled_servers() -> None:
    config = Config(
        persona={"name": "test-pilot", "profile_root": "/tmp"},
        harness={
            "mcp": McpConfig(
                servers=[
                    McpServerConfig(
                        name="github",
                        command="npx",
                        args=["-y", "@modelcontextprotocol/server-github"],
                        env=[],
                    ),
                    McpServerConfig(
                        name="local",
                        command="/bin/echo",
                        args=["hello"],
                        env=[],
                        disabled=True,
                    ),
                ],
                default_enabled=["github"],
            )
        },
    )
    manager = McpManager(config)
    enabled = manager.enabled_servers(chat_id="12345", enabled_names=None)
    assert [s["name"] for s in enabled] == ["github"]


def test_mcp_manager_honors_chat_override() -> None:
    config = Config(
        persona={"name": "test-pilot", "profile_root": "/tmp"},
        harness={
            "mcp": McpConfig(
                servers=[
                    McpServerConfig(name="a", command="/bin/echo", args=["a"], env=[]),
                    McpServerConfig(name="b", command="/bin/echo", args=["b"], env=[]),
                ],
                default_enabled=["a"],
            )
        },
    )
    manager = McpManager(config)
    enabled = manager.enabled_servers(chat_id="12345", enabled_names={"a", "b"})
    assert [s["name"] for s in enabled] == ["a", "b"]


def test_mcp_manager_skips_disabled() -> None:
    config = Config(
        persona={"name": "test-pilot", "profile_root": "/tmp"},
        harness={
            "mcp": McpConfig(
                servers=[
                    McpServerConfig(
                        name="github",
                        command="npx",
                        args=["-y"],
                        env=[],
                        disabled=True,
                    ),
                ],
                default_enabled=["github"],
            )
        },
    )
    manager = McpManager(config)
    enabled = manager.enabled_servers(chat_id="12345", enabled_names=None)
    assert enabled == []


def test_mcp_manager_renders_placeholders(tmp_path: Path) -> None:
    config = Config(
        persona={"name": "test-pilot", "profile_root": "/tmp"},
        harness={
            "sessions_root": str(tmp_path),
            "mcp": McpConfig(
                servers=[
                    McpServerConfig(
                        name="example-mcp",
                        command="python",
                        args=[
                            "-m",
                            "diploid_agent.example_mcp",
                            "--chat-id",
                            "{chat_id}",
                            "--sessions-root",
                            "{sessions_root}",
                        ],
                        env=["CHAT_DIR={chat_dir}"],
                    ),
                ],
                default_enabled=["example-mcp"],
            ),
        },
    )
    manager = McpManager(config)
    enabled = manager.enabled_servers(chat_id="chat/1")
    assert len(enabled) == 1
    assert enabled[0]["name"] == "example-mcp"
    assert enabled[0]["args"][3] == "chat/1"
    assert Path(enabled[0]["args"][5]).is_absolute()
    assert f"CHAT_DIR={tmp_path / 'chat_1'}" in enabled[0]["env"]
    assert "HARNESS_URL=http://127.0.0.1:4003" in enabled[0]["env"]


def test_mcp_manager_renders_harness_url(tmp_path: Path) -> None:
    from diploid_agent.config import Config, McpConfig, McpServerConfig

    config = Config(
        persona={"name": "test-pilot", "profile_root": "/tmp"},
        harness={
            "listen_host": "127.0.0.1",
            "listen_port": 4003,
            "sessions_root": str(tmp_path),
            "mcp": McpConfig(
                servers=[
                    McpServerConfig(
                        name="diploid-memory",
                        command="python",
                        args=[
                            "-m",
                            "diploid_agent.memory_mcp",
                            "--chat-id",
                            "{chat_id}",
                            "--harness-url",
                            "{harness_url}",
                        ],
                        env=[],
                    ),
                ],
                default_enabled=["diploid-memory"],
            ),
        },
    )
    manager = McpManager(config)
    enabled = manager.enabled_servers(chat_id="chat-1")
    assert len(enabled) == 1
    assert enabled[0]["args"][5] == "http://127.0.0.1:4003"
