"""Tests for ACP session resume (session/resume and session/load)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from diploid_agent.acp_client import (
    AcpClient,
    AcpError,
    AcpLifecycleLog,
    AcpPromptResult,
    AcpSessionStaleError,
    AcpTransportError,
)
from diploid_agent.config import (
    Config,
    DiploidConfig,
    EngineConfig,
    HarnessConfig,
    McpConfig,
    McpServerConfig,
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


def test_resume_session_retries_transient_errors(client: AcpClient, monkeypatch) -> None:
    """_resume_session retries session/resume before giving up on transient errors."""
    client.acp_resume_max_retries = 2
    calls: list[tuple[str, dict[str, Any]]] = []
    attempts = [0]

    async def fake_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        calls.append((method, params))
        if method == "session/resume":
            attempts[0] += 1
            if attempts[0] < 3:
                raise AcpError(method, {"code": -32000, "message": "transient transport error"})
            return {}
        return {}

    monkeypatch.setattr(client, "_call", fake_call)
    monkeypatch.setattr(client, "_resume_jitter", lambda attempt: 0.0)
    result = client._loop.run_until_complete(client._resume_session("s-retry", cwd=Path("/")))
    assert result == "s-retry"
    resume_calls = [c[0] for c in calls if c[0] == "session/resume"]
    assert len(resume_calls) == 3


def test_resume_session_retries_session_load_after_resume_not_found(
    client: AcpClient, monkeypatch
) -> None:
    """When session/resume is not found, session/load is also retried on transient errors."""
    client.acp_resume_max_retries = 1
    calls: list[tuple[str, dict[str, Any]]] = []
    load_attempts = [0]

    async def fake_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        calls.append((method, params))
        if method == "session/resume":
            raise AcpError(method, {"code": -32601, "message": "Method not found"})
        if method == "session/load":
            load_attempts[0] += 1
            if load_attempts[0] == 1:
                raise AcpError(method, {"code": -32000, "message": "transient transport error"})
            return {}
        return {}

    monkeypatch.setattr(client, "_call", fake_call)
    monkeypatch.setattr(client, "_resume_jitter", lambda attempt: 0.0)
    result = client._loop.run_until_complete(client._resume_session("s-load-retry", cwd=Path("/")))
    assert result == "s-load-retry"
    assert client.acp_resume_max_retries + 1 == load_attempts[0]


def test_resume_session_jitter_is_bounded(client: AcpClient) -> None:
    """_resume_jitter returns a non-negative, capped delay."""
    delay0 = client._resume_jitter(0)
    delay1 = client._resume_jitter(1)
    delay10 = client._resume_jitter(10)
    assert 0 <= delay0 <= client.acp_resume_retry_max_seconds
    assert 0 <= delay1 <= client.acp_resume_retry_max_seconds
    assert 0 <= delay10 <= client.acp_resume_retry_max_seconds


def test_set_session_model_issues_config_option(client: AcpClient, monkeypatch) -> None:
    """_set_session_model sends session/set_config_option and tracks the model."""
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        calls.append((method, params))
        return {}

    monkeypatch.setattr(client, "_call", fake_call)
    applied = client._loop.run_until_complete(client._set_session_model("s-1", "glm-5.2"))
    assert applied == "glm-5-2"
    assert calls == [
        (
            "session/set_config_option",
            {"sessionId": "s-1", "configId": "model", "value": "glm-5-2"},
        )
    ]
    assert client._session_models["s-1"] == "glm-5-2"

    # A repeat call with the same model is a no-op.
    applied = client._loop.run_until_complete(client._set_session_model("s-1", "glm-5-2"))
    assert applied == "glm-5-2"
    assert len(calls) == 1


def test_send_message_does_not_reconfigure_after_set_session_model(
    client: AcpClient, monkeypatch
) -> None:
    """A follow-up send_message on the switched model issues no config call."""
    calls: list[str] = []

    async def fake_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        calls.append(method)
        return {}

    async def fake_prompt(session_id: str, prompt_text: str, **kwargs: Any) -> Any:
        return AcpPromptResult(reply="ok", session_id=session_id)

    monkeypatch.setattr(client, "_call", fake_call)
    monkeypatch.setattr(client, "_prompt", fake_prompt)

    applied = client._loop.run_until_complete(client._set_session_model("s-1", "glm-5-2"))
    assert applied == "glm-5-2"
    calls.clear()

    client._loop.run_until_complete(client._send_message("s-1", "hi", model="glm-5.2"))
    assert "session/set_config_option" not in calls

    # A genuinely different model still triggers exactly one config call.
    client._loop.run_until_complete(client._send_message("s-1", "hi", model="swe-1-7"))
    assert calls == ["session/set_config_option"]


def test_set_session_model_failure_drops_cached_model(client: AcpClient, monkeypatch) -> None:
    """On failure the cached model is dropped so send_message re-pins it."""

    async def failing_call(method: str, params: dict[str, Any], **kwargs: Any) -> Any:
        raise AcpTransportError("response lost")

    monkeypatch.setattr(client, "_call", failing_call)
    monkeypatch.setattr(client, "_ensure_started", lambda: None)
    monkeypatch.setattr(
        client,
        "_run",
        lambda coro, timeout=None: client._loop.run_until_complete(coro),
    )
    client._session_models["s-1"] = "swe-1-7"

    with pytest.raises(AcpTransportError):
        client.set_session_model("s-1", "glm-5-2")
    assert "s-1" not in client._session_models


def test_acp_engine_resume_forwards_timeout(monkeypatch) -> None:
    """AcpEngine.resume_session forwards an explicit timeout and applies the default."""
    engine = AcpEngine(
        config=EngineConfig(bin="/bin/echo"),
        api_key="test",
        metrics=None,  # type: ignore[arg-type]
    )
    captured: list[dict[str, Any]] = []

    def fake_resume(*args: Any, **kwargs: Any) -> str:
        captured.append(kwargs)
        return "resumed-id"

    monkeypatch.setattr(engine._client, "resume_session", fake_resume)
    try:
        engine.resume_session("s-9", timeout=7.5)
        assert captured[-1]["timeout"] == 7.5

        engine.resume_session("s-9")
        assert captured[-1]["timeout"] == engine.config.acp_resume_timeout
    finally:
        engine.close()


def test_rehydrate_after_restart_uses_short_resume_budget(monkeypatch, tmp_path: Path) -> None:
    """A restart-first rehydrate resumes with acp_resume_after_restart_timeout."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Ready.", session_id="s-1")

    send_calls = [0]

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        send_calls[0] += 1
        if send_calls[0] == 1:
            raise AcpTransportError("session/prompt", msg="stuck")
        return AcpPromptResult(reply="Resumed reply.", session_id=session_id)

    resume_timeouts: list[float | None] = []

    def fake_resume(session_id: str, *, timeout=None, **kwargs):
        resume_timeouts.append(timeout)
        return session_id

    restarts: list[None] = []

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)
    monkeypatch.setattr(
        harness.client,
        "restart_transport",
        lambda reason=None, chat_id=None: restarts.append(None),
    )

    try:
        result1 = harness.process("chat-restart", "hello")
        assert result1.session_id == "s-1"

        result2 = harness.process("chat-restart", "follow-up")
        assert result2.reply == "Resumed reply."
        assert result2.session_id == "s-1"
        assert restarts  # the transport was restarted before the resume attempt
        assert resume_timeouts == [config.engine.acp_resume_after_restart_timeout]
    finally:
        harness.client.close()


def test_rehydrate_stale_session_uses_full_resume_budget(monkeypatch, tmp_path: Path) -> None:
    """A stale-session rehydrate (no restart) keeps the full acp_resume_timeout."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Ready.", session_id="s-1")

    send_calls = [0]

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        send_calls[0] += 1
        if send_calls[0] == 1:
            raise AcpSessionStaleError(
                "session/prompt",
                {"code": -32002, "message": "Session not found"},
            )
        return AcpPromptResult(reply="Resumed reply.", session_id=session_id)

    resume_timeouts: list[float | None] = []

    def fake_resume(session_id: str, *, timeout=None, **kwargs):
        resume_timeouts.append(timeout)
        return session_id

    restarts: list[None] = []

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)
    monkeypatch.setattr(
        harness.client,
        "restart_transport",
        lambda reason=None, chat_id=None: restarts.append(None),
    )

    try:
        harness.process("chat-stale", "hello")

        result2 = harness.process("chat-stale", "follow-up")
        assert result2.reply == "Resumed reply."
        assert result2.session_id == "s-1"
        assert not restarts  # a stale session does not restart the transport first
        # None = no override; AcpEngine resolves it to the full acp_resume_timeout.
        assert resume_timeouts == [None]
    finally:
        harness.client.close()


def test_process_stale_session_resumes_despite_mcp_drift(monkeypatch, tmp_path: Path) -> None:
    """MCP drift is absorbed by resume (transport restart + session/load)."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    call_order: list[str] = []
    resume_kwargs: list[dict[str, Any]] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id=f"session-{len(call_order)}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        resume_kwargs.append(kwargs)
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
        result1 = harness.process("chat-mcp", "hello")
        assert result1.session_id == "session-1"

        # The default MCP set grows after the record's last stamp.
        monkeypatch.setattr(
            harness.runtime._mcp_skills,
            "_active_mcp_server_names",
            lambda chat_id: ["a-new-default"],
        )

        result2 = harness.process("chat-mcp", "follow-up")
        assert result2.session_number == 1
        assert result2.session_id == "session-1"
        assert "resume" in call_order
        assert call_order.count("create") == 1
        # The active MCP list is passed through so the client can restart
        # the transport and write the new mcp_config.json.
        assert resume_kwargs and "mcp_servers" in resume_kwargs[0]
    finally:
        harness.client.close()


@pytest.mark.parametrize("acp_resume_enabled", [True, False])
def test_rehydrate_skips_alive_probe_on_consistency_failure(
    monkeypatch, tmp_path: Path, acp_resume_enabled: bool
) -> None:
    """A record rejected by _can_resume_record must not be revived by the
    session_alive probe — under either acp_resume_enabled value.

    Drives _rehydrate directly: the normal process path catches skills
    drift upstream via skills_changed, but continue_turn/dispatch wake
    paths reach _rehydrate without that gate.
    """
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=acp_resume_enabled)
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
        call_order.append("send")
        return AcpPromptResult(reply="Fresh reply.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)
    monkeypatch.setattr(harness.client, "session_alive", fake_session_alive)

    try:
        harness.process("chat-skill", "hello")

        # Known skills drift: record tracks an empty set while a turn-matched
        # skill remains active (record.enabled_skills is unioned into the
        # active set, so drift requires a source outside the record).
        record = harness.active_record("chat-skill")
        record.enabled_skills = []
        harness.runtime._mcp_skills._active_chat_skills["chat-skill"] = {"matched-skill"}

        ret = harness.runtime.turn_controller._rehydrate(
            "chat-skill",
            "follow-up",
            record,
            "swe-1-7",
            on_chunk=lambda chunk: None,
            on_update=lambda update: None,
        )
        assert not isinstance(ret, ChatResult)
        assert "alive" not in call_order
        assert "resume" not in call_order
        assert call_order.count("create") == 2
    finally:
        harness.client.close()


def test_process_mcp_drift_resyncs_live_session(monkeypatch, tmp_path: Path) -> None:
    """MCP drift on a live session resyncs via resume instead of session/new."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    call_order: list[str] = []
    resume_kwargs: list[dict[str, Any]] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id=f"session-{len(call_order)}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        resume_kwargs.append(kwargs)
        return session_id

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("send")
        return AcpPromptResult(reply="Follow-up.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)

    try:
        result1 = harness.process("chat-mcp-live", "hello")
        assert result1.session_id == "session-1"

        monkeypatch.setattr(
            harness.runtime._mcp_skills,
            "_active_mcp_server_names",
            lambda chat_id: ["a-new-default"],
        )

        result2 = harness.process("chat-mcp-live", "follow-up")
        assert result2.session_number == 1
        assert result2.session_id == "session-1"
        assert "resume" in call_order
        assert call_order.count("create") == 1
        record = harness.active_record("chat-mcp-live")
        assert record.enabled_mcp_servers == ["a-new-default"]
    finally:
        harness.client.close()


def test_process_mcp_resync_failure_falls_back_to_new_session(monkeypatch, tmp_path: Path) -> None:
    """A failed MCP resync falls through the normal rehydrate path to
    session/new rather than leaving stale MCP on the live session.

    The resync failure marks the session unreachable, so _rehydrate must
    not re-attempt resume or the session_alive probe — that would double
    the stall."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    call_order: list[str] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id=f"session-{len(call_order)}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        raise RuntimeError("ACP session/prompt failed: Session not found")

    def fake_session_alive(session_id: str) -> bool:
        call_order.append("alive")
        return False

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("send")
        return AcpPromptResult(reply="Follow-up.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)
    monkeypatch.setattr(harness.client, "session_alive", fake_session_alive)

    try:
        harness.process("chat-mcp-fail", "hello")

        monkeypatch.setattr(
            harness.runtime._mcp_skills,
            "_active_mcp_server_names",
            lambda chat_id: ["a-new-default"],
        )

        result2 = harness.process("chat-mcp-fail", "follow-up")
        assert result2.session_id != "session-1"
        assert call_order.count("create") == 2
        # Exactly one resume attempt (the resync) — no second attempt or
        # alive probe inside _rehydrate.
        assert call_order.count("resume") == 1
        assert "alive" not in call_order
    finally:
        harness.client.close()


def test_model_boundary_does_not_resurrect_archived_session(monkeypatch, tmp_path: Path) -> None:
    """A transient session/new failure after a deliberate model boundary
    must not resume the archived session — the user asked for a fresh
    context, so _rehydrate runs with allow_resume=False."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    call_order: list[str] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        if call_order.count("create") == 2:
            raise AcpTransportError("session/new", msg="connection lost")
        return AcpPromptResult(reply=f"Ready-{model}", session_id=f"session-{len(call_order)}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        return session_id

    def fake_session_alive(session_id: str) -> bool:
        call_order.append("alive")
        return True

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)
    monkeypatch.setattr(harness.client, "session_alive", fake_session_alive)
    monkeypatch.setattr(harness.client, "restart_transport", lambda reason=None, chat_id=None: None)

    try:
        result1 = harness.process("chat-boundary", "hello")
        assert result1.session_id == "session-1"

        result2 = harness.process("chat-boundary", "hi", model="glm-5-2")
        # The archived session-1 must not be revived: no resume attempt,
        # no alive probe — just a fresh session/new after the restart.
        assert "resume" not in call_order
        assert "alive" not in call_order
        assert call_order.count("create") == 3
        assert result2.session_id == "session-3"
        assert result2.session_id != "session-1"
    finally:
        harness.client.close()


def test_continue_turn_resyncs_mcp_drift(monkeypatch, tmp_path: Path) -> None:
    """Dispatch continuations resync MCP drift like normal turns — without
    it the continuation prompts against stale MCP config and _finalize_turn
    restamps the record, silently masking the drift."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    call_order: list[str] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        return session_id

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("send")
        return AcpPromptResult(reply="Done.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)

    try:
        harness.process("chat-dispatch", "Please dispatch")
        chat_result = harness.dispatch("chat-dispatch")

        monkeypatch.setattr(
            harness.runtime._mcp_skills,
            "_active_mcp_server_names",
            lambda chat_id: ["a-new-default"],
        )

        result = harness.continue_turn(chat_result.dispatch_id, "worker done")
        assert result.session_id == "session-1"
        assert "resume" in call_order
        # The continuation prompt went to the resynced session.
        assert call_order[-1] == "send"
        record = harness.active_record("chat-dispatch")
        assert record.enabled_mcp_servers == ["a-new-default"]
    finally:
        harness.client.close()


def test_resume_session_runs_as_background(client: AcpClient, monkeypatch) -> None:
    """resume_session marks its _run background — an opportunistic resume
    budget expiring must not mark the shared transport unhealthy."""
    captured: dict[str, Any] = {}

    def fake_run(coro: Any, timeout: float | None = None, background: bool = False):
        captured["background"] = background
        coro.close()
        return "s-1"

    monkeypatch.setattr(client, "_ensure_started", lambda mcp_servers=None: None)
    monkeypatch.setattr(client, "_run", fake_run)

    assert client.resume_session("s-1") == "s-1"
    assert captured["background"] is True


def test_mcp_disable_default_does_not_trigger_spurious_resync(monkeypatch, tmp_path: Path) -> None:
    """Disabling a default MCP server must not look like drift — the
    disabled overlay keeps the effective set equal to the record stamp,
    so follow-ups do not waste a resync restart, and /new keeps the
    disable instead of the default union silently re-adding it."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    config.harness.mcp = McpConfig(
        servers=[McpServerConfig(name="github", command="npx", args=["-y"], env=[])],
        default_enabled=["github"],
    )
    harness = ConversationHarness(config)

    call_order: list[str] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        return session_id

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("send")
        return AcpPromptResult(reply="Follow-up.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)

    try:
        harness.process("chat-mcp-off", "hello")
        record = harness.active_record("chat-mcp-off")
        assert record.enabled_mcp_servers == ["github"]

        harness.mcp_disable("chat-mcp-off", "github")
        assert record.enabled_mcp_servers == []
        assert record.disabled_mcp_servers == ["github"]
        assert harness.runtime._mcp_skills._active_mcp_server_names("chat-mcp-off") == []

        result = harness.process("chat-mcp-off", "follow-up")
        assert "resume" not in call_order  # effective set unchanged — no resync
        assert result.session_id == "session-1"

        # The disable is per-chat config and survives the session boundary.
        harness.new_session("chat-mcp-off")
        record2 = harness.active_record("chat-mcp-off")
        assert record2.disabled_mcp_servers == ["github"]
        assert harness.runtime._mcp_skills._active_mcp_server_names("chat-mcp-off") == []
    finally:
        harness.client.close()


def test_process_mcp_resync_transport_failure_is_stale_class(monkeypatch, tmp_path: Path) -> None:
    """A transport-class resync failure is reclassified as stale, not
    transport-unhealthy: no restart_first, no second resume attempt, no
    alive probe — straight to session/new on the existing transport."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    harness = ConversationHarness(config)

    call_order: list[str] = []
    restarts: list[None] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id=f"session-{len(call_order)}")

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        call_order.append("resume")
        raise AcpTransportError("session/resume", msg="resume budget expired")

    def fake_session_alive(session_id: str) -> bool:
        call_order.append("alive")
        return True

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("send")
        return AcpPromptResult(reply="Follow-up.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)
    monkeypatch.setattr(harness.client, "session_alive", fake_session_alive)
    monkeypatch.setattr(
        harness.client,
        "restart_transport",
        lambda reason=None, chat_id=None: restarts.append(None),
    )

    try:
        harness.process("chat-mcp-transport", "hello")

        monkeypatch.setattr(
            harness.runtime._mcp_skills,
            "_active_mcp_server_names",
            lambda chat_id: ["a-new-default"],
        )

        result2 = harness.process("chat-mcp-transport", "follow-up")
        assert result2.session_id != "session-1"
        assert call_order.count("create") == 2
        assert call_order.count("resume") == 1
        assert "alive" not in call_order
        # Stale-class, not transport-class: the rehydrate path does not
        # restart the transport first.
        assert not restarts
    finally:
        harness.client.close()


def test_implicit_boundary_preserves_mcp_disable(monkeypatch, tmp_path: Path) -> None:
    """A stale-rehydrate session/new must carry disabled_mcp_servers onto
    the new record — otherwise the next turn's default union re-adds the
    disabled server and fakes MCP drift (a pointless resync restart)."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=True)
    config.harness.mcp = McpConfig(
        servers=[McpServerConfig(name="github", command="npx", args=["-y"], env=[])],
        default_enabled=["github"],
    )
    harness = ConversationHarness(config)

    send_calls = [0]

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Ready.", session_id=f"session-{len(send_calls) + 1}")

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        send_calls[0] += 1
        if send_calls[0] == 1:
            raise AcpSessionStaleError(
                "session/prompt", {"code": -32002, "message": "Session not found"}
            )
        return AcpPromptResult(reply="Follow-up.", session_id=session_id)

    def fake_resume(session_id: str, *, cwd=None, model=None, **kwargs):
        raise AcpSessionStaleError(
            "session/resume", {"code": -32002, "message": "Session not found"}
        )

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "resume_session", fake_resume)
    monkeypatch.setattr(harness.client, "session_alive", lambda session_id: False)

    try:
        harness.process("chat-disable-boundary", "hello")
        harness.mcp_disable("chat-disable-boundary", "github")

        # The stale send forces rehydrate → session/new → new record.
        result = harness.process("chat-disable-boundary", "follow-up")
        record = harness.active_record("chat-disable-boundary")
        assert result.session_id != "session-1"
        assert record.disabled_mcp_servers == ["github"]
        assert record.enabled_mcp_servers == []
        # And the effective set stays empty — no phantom drift next turn.
        assert not harness.runtime._mcp_skills._mcp_record_drifted("chat-disable-boundary", record)
    finally:
        harness.client.close()


def test_resume_command_probe_is_consistency_gated(monkeypatch, tmp_path: Path) -> None:
    """`/resume` keeps its explicit resume attempt, but a timeout-flagged
    source must not be silently revived by the session_alive probe."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root, acp_resume_enabled=False)
    harness = ConversationHarness(config)

    call_order: list[str] = []

    def fake_create_session(prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("create")
        return AcpPromptResult(reply="Ready.", session_id=f"session-{len(call_order)}")

    def fake_session_alive(session_id: str) -> bool:
        call_order.append("alive")
        return True

    def fake_send_message(session_id: str, prompt: str, *, cwd=None, model=None, **kwargs):
        call_order.append("send")
        return AcpPromptResult(reply="Follow-up.", session_id=session_id)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)
    monkeypatch.setattr(harness.client, "session_alive", fake_session_alive)

    try:
        harness.process("chat-resume-cmd", "hello")
        harness.new_session("chat-resume-cmd")

        # The archived session-1 record carries a timeout stop reason —
        # consistency rules say it must not be revived.
        state = harness.runtime.chat_state("chat-resume-cmd")
        source = state.sessions[1]
        source.last_stop_reason = "timeout"

        harness.resume_session("chat-resume-cmd", 1)
        assert "alive" not in call_order
        # Rehydration fell through to a fresh session under session 1.
        assert source.session_id != "session-1"
    finally:
        harness.client.close()


def test_can_resume_record_expected_skills_baseline(monkeypatch, tmp_path: Path) -> None:
    """_can_resume_record(expected_skills=...) compares the record against
    the given baseline — /branch must judge the source session by the
    source's own skill set, not the active record's."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    monkeypatch.setattr(
        harness.client,
        "create_session",
        lambda prompt, *, cwd=None, model=None, **kwargs: AcpPromptResult(
            reply="Ready.", session_id="session-1"
        ),
    )

    try:
        harness.process("chat-br", "hello")
        record = harness.active_record("chat-br")
        record.enabled_skills = ["skill-a"]
        # The active set drifts from the source's own skills.
        harness.runtime._mcp_skills._active_chat_skills["chat-br"] = {"skill-b"}

        session = harness.runtime.turn_controller.session
        assert session._can_resume_record("chat-br", record) is False
        assert session._can_resume_record("chat-br", record, expected_skills={"skill-a"}) is True
    finally:
        harness.client.close()
