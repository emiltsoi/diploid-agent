"""Tests for ACP session resume (session/resume and session/load)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from diploid_agent.acp_client import AcpClient, AcpError, AcpLifecycleLog, AcpPromptResult
from diploid_agent.config import (
    Config,
    DiploidConfig,
    EngineConfig,
    HarnessConfig,
    PersonaConfig,
    Secrets,
)
from diploid_agent.engine import AcpEngine
from diploid_agent.harness import ConversationHarness
from diploid_agent.models import ChatResult


@pytest.fixture
def client(tmp_path: Path) -> AcpClient:
    c = AcpClient(
        agent_bin="/bin/echo",
        api_key="test-key",
    )
    c._loop = asyncio.new_event_loop()
    c._initialized = False
    yield c
    if c._loop is not None and not c._loop.is_closed():
        c._loop.close()


def _make_config(tmp_path: Path, fixture_root: Path, acp_resume_enabled: bool = False) -> Config:
    return Config(
        diploid=DiploidConfig(
            bin="/bin/echo",
            model="swe-1-7",
            acp_resume_enabled=acp_resume_enabled,
        ),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=fixture_root,
            fleet_root=tmp_path / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def test_acp_resume_enabled_default_is_true() -> None:
    """The production default for ACP session resume is enabled."""
    assert EngineConfig.model_fields["acp_resume_enabled"].default is True


def test_resume_session_tries_resume_then_load(client: AcpClient, monkeypatch) -> None:
    """_resume_session falls back from session/resume to session/load on method-not-found."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        calls.append((method, params))
        if method == "session/resume":
            raise AcpError(method, {"code": -32601, "message": "Method not found"})
        return {}

    monkeypatch.setattr(client, "_call", fake_call)
    result = client._loop.run_until_complete(client._resume_session("s-1", cwd=Path("/")))
    assert result == "s-1"
    methods = [c[0] for c in calls]
    assert "session/resume" in methods
    assert "session/load" in methods
    assert "session/set_config_option" in methods


def test_resume_session_succeeds_immediately(client: AcpClient, monkeypatch) -> None:
    """When session/resume works, session/load is not called."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        calls.append((method, params))
        if method == "session/resume":
            return {}
        return {}

    monkeypatch.setattr(client, "_call", fake_call)
    result = client._loop.run_until_complete(client._resume_session("s-2", cwd=Path("/")))
    assert result == "s-2"
    methods = [c[0] for c in calls]
    assert methods == [
        "session/resume",
        "session/set_config_option",
        "session/set_config_option",
    ]


def test_session_load_passes_empty_mcp_servers(client: AcpClient, monkeypatch) -> None:
    """_session_load sends an empty mcpServers list; devin acp loads from mcp_config.json."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        calls.append((method, params))
        return {}

    monkeypatch.setattr(client, "_call", fake_call)
    mcp_servers = [{"name": "test-mcp", "command": "cmd"}]
    result = client._loop.run_until_complete(
        client._session_load("s-3", cwd=Path("/"), mcp_servers=mcp_servers)
    )
    assert result == "s-3"
    load_params = next(c[1] for c in calls if c[0] == "session/load")
    assert load_params["mcpServers"] == []
    assert client._mcp_servers == mcp_servers


@pytest.mark.parametrize(
    "error,expected",
    [
        (AcpError("session/resume", {"code": -32601}), True),
        (
            AcpError(
                "session/resume",
                {"code": -32602, "message": "Method not found"},
            ),
            True,
        ),
        (
            AcpError(
                "session/resume",
                {"code": 123, "message": "Something else"},
            ),
            False,
        ),
    ],
)
def test_is_method_not_found(error: AcpError, expected: bool) -> None:
    """_is_method_not_found detects JSON-RPC method-not-found responses."""
    assert AcpClient._is_method_not_found(error) is expected


def test_acp_engine_resume_delegates_to_client(monkeypatch, tmp_path: Path) -> None:
    """AcpEngine.resume_session passes parameters to the AcpClient."""
    engine = AcpEngine(
        config=EngineConfig(bin="/bin/echo"),
        api_key="test",
        metrics=None,  # type: ignore[arg-type]
    )
    captured: dict[str, Any] = {}

    def fake_resume(*args: Any, **kwargs: Any) -> str:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "resumed-id"

    monkeypatch.setattr(engine._client, "resume_session", fake_resume)
    try:
        result = engine.resume_session(
            "s-4",
            cwd=tmp_path,
            model="swe-1-7",
            mcp_servers=[{"name": "mcp"}],
        )
        assert result == "resumed-id"
        assert captured["kwargs"]["model"] == "swe-1-7"
        assert captured["kwargs"]["cwd"] == tmp_path
    finally:
        engine.close()


def test_resume_command_uses_acp_resume(monkeypatch, tmp_path: Path) -> None:
    """`/resume` uses ACP session resume when enabled."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    create_count: list[int] = []
    resume_calls: list[tuple[str, str | None, dict[str, Any]]] = []
    send_calls: list[tuple[str, str, str | None]] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        create_count.append(1)
        return AcpPromptResult(reply="Ready.", session_id=f"session-{model}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        resume_calls.append((session_id, model, kwargs))
        return session_id

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        send_calls.append((session_id, prompt[:20], model))
        return AcpPromptResult(reply="Resumed reply.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)

    try:
        harness.process("chat-r1", "hello")
        harness.new_session("chat-r1")

        result = harness.resume_session("chat-r1", 1)
        assert isinstance(result, ChatResult)
        assert resume_calls
        assert resume_calls[0][0] == result.session_id
        assert send_calls
        assert send_calls[0][0] == result.session_id
        assert len(create_count) == 2  # process + new_session only
    finally:
        harness.client.close()


def test_branch_command_uses_acp_resume(monkeypatch, tmp_path: Path) -> None:
    """`/branch` resumes the source ACP session and sends a follow-up."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    resume_calls: list[tuple[str, str | None]] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Ready.", session_id=f"session-{model}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        resume_calls.append((session_id, model))
        return session_id

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Branched reply.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)

    try:
        harness.process("chat-r2", "hello")
        harness.new_session("chat-r2")

        result = harness.branch_session("chat-r2", 1)
        assert result.session_number == 3
        assert resume_calls
        assert result.session_id == resume_calls[0][0]
    finally:
        harness.client.close()


def test_process_stale_session_attempts_resume(monkeypatch, tmp_path: Path) -> None:
    """A stale follow-up prompt attempts ACP session resume before rehydration."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    call_order: list[str] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id=f"session-{len(call_order)}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        return session_id

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        if call_order.count("send") == 0:
            call_order.append("send")
            raise RuntimeError("ACP session/prompt failed: Session not found")
        call_order.append("send")
        return AcpPromptResult(reply="Follow-up after resume.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)

    try:
        result1 = harness.process("chat-r3", "hello")
        assert result1.session_id == "session-1"
        assert result1.session_number == 1

        result2 = harness.process("chat-r3", "follow-up")
        assert result2.session_number == 1
        assert result2.session_id == "session-1"
        assert "resume" in call_order
        assert call_order.count("create") == 1
    finally:
        harness.client.close()


def test_resume_session_writes_lifecycle_log(
    client: AcpClient, monkeypatch, tmp_path: Path
) -> None:
    """_resume_session appends attempt and success events to the lifecycle log."""
    log = AcpLifecycleLog(tmp_path / "acp-lifecycle.jsonl")
    client._lifecycle_log = log

    async def fake_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        return {}

    monkeypatch.setattr(client, "_call", fake_call)
    result = client._loop.run_until_complete(client._resume_session("s-log", cwd=Path("/")))
    assert result == "s-log"

    lines = log.path.read_text().strip().split("\n")
    events = [json.loads(line)["event"] for line in lines if line]
    assert "session.resume.attempt" in events
    assert "session.resume.success" in events


def test_process_stale_session_uses_session_alive_when_resume_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    """When ACP resume is disabled, a stale session falls back to session_alive."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=False)
    harness = ConversationHarness(config)

    call_order: list[str] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id=f"session-{len(call_order)}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        return session_id

    def fake_session_alive(session_id: str) -> bool:
        call_order.append("alive")
        return True

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        if call_order.count("send") == 0:
            call_order.append("send")
            raise RuntimeError("ACP session/prompt failed: Session not found")
        call_order.append("send")
        return AcpPromptResult(reply="Follow-up after alive.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)
    monkeypatch.setattr(harness.client, "session_alive", fake_session_alive)

    try:
        result1 = harness.process("chat-r4", "hello")
        assert result1.session_id == "session-1"
        assert result1.session_number == 1

        result2 = harness.process("chat-r4", "follow-up")
        assert result2.session_number == 1
        assert result2.session_id == "session-1"
        assert "alive" in call_order
        assert "resume" not in call_order
        assert call_order.count("create") == 1
    finally:
        harness.client.close()
