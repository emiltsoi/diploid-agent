"""Tests for AcpClient helpers and construction."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from acp_fleet_harness.acp_client import (
    AcpClient,
    AcpPromptResult,
    _load_windsurf_api_key,
    _normalize_model,
    _Prompt,
    _resolve_devin_bin,
)


def test_load_windsurf_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("WINDSURF_API_KEY", "env-key")
    assert _load_windsurf_api_key() == "env-key"


def test_load_windsurf_api_key_from_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WINDSURF_API_KEY", raising=False)
    creds = tmp_path / ".local" / "share" / "devin" / "credentials.toml"
    creds.parent.mkdir(parents=True, exist_ok=True)
    creds.write_text('windsurf_api_key = "creds-key"')
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert _load_windsurf_api_key() == "creds-key"


def test_resolve_devin_bin_existing_path(tmp_path: Path) -> None:
    devin = tmp_path / "devin"
    devin.write_text("#!/bin/sh\n")
    devin.chmod(0o755)
    assert _resolve_devin_bin(str(devin)) == devin


def test_resolve_devin_bin_prefers_path_arg(monkeypatch, tmp_path: Path) -> None:
    # If the explicit path exists, it should win over PATH lookup.
    devin = tmp_path / "devin"
    devin.write_text("#!/bin/sh\n")
    devin.chmod(0o755)
    assert _resolve_devin_bin(str(devin)) == devin


def test_resolve_devin_bin_falls_back_to_path(monkeypatch, tmp_path: Path) -> None:
    devin = tmp_path / "devin"
    devin.write_text("#!/bin/sh\n")
    devin.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert _resolve_devin_bin("/nonexistent/devin") == devin


def test_resolve_devin_bin_raises_when_not_found(tmp_path: Path) -> None:
    monkeypatch_tmp = pytest.MonkeyPatch()
    monkeypatch_tmp.setenv("PATH", str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="devin binary not found"):
            _resolve_devin_bin("/nonexistent/devin")
    finally:
        monkeypatch_tmp.undo()


def test_normalize_model_converts_dotted_aliases() -> None:
    assert _normalize_model("swe-1.7") == "swe-1-7"
    assert _normalize_model("swe-1.7-medium") == "swe-1-7-medium"
    assert _normalize_model("swe-1-7") == "swe-1-7"


def test_acp_client_uses_api_key_argument() -> None:
    client = AcpClient(devin_bin="/bin/true", api_key="passed-key")
    assert client._api_key == "passed-key"


def test_acp_client_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("WINDSURF_API_KEY", "env-key")
    client = AcpClient(devin_bin="/bin/true")
    assert client._api_key == "env-key"


def test_acp_client_raises_when_no_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WINDSURF_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    devin_bin = tmp_path / "devin"
    devin_bin.write_text("#!/bin/sh\n")
    with pytest.raises(RuntimeError, match="No WINDSURF_API_KEY"):
        AcpClient(devin_bin=str(devin_bin))


def test_route_update_collects_full_and_chunked_messages() -> None:
    """ACP may send agent_message (full) or agent_message_chunk (incremental)."""
    client = AcpClient(devin_bin="/bin/true", api_key="test-key")
    loop = asyncio.new_event_loop()
    client._loop = loop
    client._active_prompts = {}

    chunks: list[str] = []
    updates: list[dict[str, Any]] = []

    prompt = _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=loop.create_future(),
        cancel_done=loop.create_future(),
        on_chunk=chunks.append,
        on_update=updates.append,
    )
    client._active_prompts["s-1"] = prompt

    # Full agent_message with a list of content blocks.
    client._route_update(
        {
            "params": {
                "sessionId": "s-1",
                "update": {
                    "sessionUpdate": "agent_message",
                    "content": [
                        {"type": "text", "text": "Hello, "},
                        {"type": "text", "text": "world!"},
                    ],
                },
            }
        }
    )

    # Incremental chunk.
    client._route_update(
        {
            "params": {
                "sessionId": "s-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": " How are you?"},
                },
            }
        }
    )

    assert chunks == ["Hello, ", "world!", " How are you?"]
    assert prompt.chunks == ["Hello, ", "world!", " How are you?"]
    assert len(updates) == 2


def test_route_update_passes_thought_updates_to_on_update() -> None:
    """agent_thought and agent_thought_chunk are routed via on_update, not on_chunk."""
    client = AcpClient(devin_bin="/bin/true", api_key="test-key")
    loop = asyncio.new_event_loop()
    client._loop = loop
    client._active_prompts = {}

    chunks: list[str] = []
    updates: list[dict[str, Any]] = []

    prompt = _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=loop.create_future(),
        cancel_done=loop.create_future(),
        on_chunk=chunks.append,
        on_update=updates.append,
    )
    client._active_prompts["s-1"] = prompt

    client._route_update(
        {
            "params": {
                "sessionId": "s-1",
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "thinking..."},
                },
            }
        }
    )

    assert not chunks
    assert len(updates) == 1
    assert updates[0]["sessionUpdate"] == "agent_thought_chunk"


def test_create_session_passes_mcp_servers(monkeypatch, tmp_path: Path) -> None:
    client = AcpClient(devin_bin="/bin/true", api_key="test-key")
    client._loop = asyncio.new_event_loop()
    calls: list[tuple[str, Any]] = []

    async def fake_call(method: str, params: Any) -> Any:
        calls.append((method, params))
        if method == "session/new":
            return {"sessionId": "s-1", "configOptions": []}
        return {}

    async def fake_prompt(*args: Any, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok")

    monkeypatch.setattr(client, "_call", fake_call)
    monkeypatch.setattr(client, "_prompt", fake_prompt)

    result = client._loop.run_until_complete(
        client._create_session(
            "hi",
            mcp_servers=[{"name": "github", "command": "npx", "args": ["-y"], "env": []}],
        )
    )

    assert result.reply == "ok"
    new_session_calls = [c for c in calls if c[0] == "session/new"]
    assert len(new_session_calls) == 1
    assert new_session_calls[0][1]["mcpServers"] == [
        {"name": "github", "command": "npx", "args": ["-y"], "env": []}
    ]


class FakeStream:
    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    async def readline(self) -> bytes:
        await asyncio.sleep(100)
        return b""


class FakeProcess:
    stdin = FakeStream()
    stdout = FakeStream()
    returncode: int | None = None

    def terminate(self) -> None:
        pass

    async def wait(self) -> int:
        return 0


def test_prompt_hard_timeout_returns_partial() -> None:
    """A hard timeout produces a partial result with stop_reason='timeout'."""
    client = AcpClient(devin_bin="/bin/true", api_key="test-key")
    client.timeout = 0.05
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]

    result = client._loop.run_until_complete(client._prompt("s-1", "hi"))
    assert result.partial is True
    assert result.stop_reason == "timeout"
    assert result.timed_out is True
    assert result.cancelled is False


class FakeProcessForRestart:
    def __init__(self) -> None:
        self.stdin = FakeStream()
        self.stdout = FakeStream()
        self.returncode: int | None = None

    def terminate(self) -> None:
        pass

    async def wait(self) -> int:
        return 0


def test_restart_transport_starts_new_subprocess(monkeypatch) -> None:
    """restart_transport closes the old process and starts a new one."""
    client = AcpClient(devin_bin="/bin/true", api_key="test-key")

    starts = [0]

    async def fake_start() -> None:
        starts[0] += 1
        client._proc = FakeProcessForRestart()  # type: ignore[assignment]

    async def fake_close() -> None:
        if client._proc is not None:
            client._proc.returncode = -1

    monkeypatch.setattr(client, "_start_transport", fake_start)
    monkeypatch.setattr(client, "_close_transport", fake_close)

    client._ensure_started()
    assert client._initialized
    assert starts[0] == 1

    client.restart_transport()
    assert client._initialized
    assert starts[0] == 2
