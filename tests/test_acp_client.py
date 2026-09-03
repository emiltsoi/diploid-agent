"""Tests for AcpClient helpers and construction."""

import asyncio
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from diploid_agent.acp_client import (
    AcpClient,
    AcpError,
    AcpMcpError,
    AcpModelError,
    AcpPromptResult,
    AcpRestartHistory,
    AcpSessionStaleError,
    AcpTransportError,
    _acp_error_from_response,
    _load_windsurf_api_key,
    _normalize_model,
    _Prompt,
    _resolve_agent_bin,
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


def test_resolve_agent_bin_existing_path(tmp_path: Path) -> None:
    agent_bin = tmp_path / "devin"
    agent_bin.write_text("#!/bin/sh\n")
    agent_bin.chmod(0o755)
    assert _resolve_agent_bin(str(agent_bin)) == agent_bin


def test_resolve_agent_bin_prefers_path_arg(monkeypatch, tmp_path: Path) -> None:
    # If the explicit path exists, it should win over PATH lookup.
    agent_bin = tmp_path / "devin"
    agent_bin.write_text("#!/bin/sh\n")
    agent_bin.chmod(0o755)
    assert _resolve_agent_bin(str(agent_bin)) == agent_bin


def test_resolve_agent_bin_falls_back_to_path(monkeypatch, tmp_path: Path) -> None:
    agent_bin = tmp_path / "devin"
    agent_bin.write_text("#!/bin/sh\n")
    agent_bin.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert _resolve_agent_bin("/nonexistent/devin") == agent_bin


def test_resolve_agent_bin_raises_when_not_found(tmp_path: Path) -> None:
    monkeypatch_tmp = pytest.MonkeyPatch()
    monkeypatch_tmp.setenv("PATH", str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="agent binary not found"):
            _resolve_agent_bin("/nonexistent/devin")
    finally:
        monkeypatch_tmp.undo()


def test_normalize_model_converts_dotted_aliases() -> None:
    assert _normalize_model("swe-1.7") == "swe-1-7"
    assert _normalize_model("swe-1.7-medium") == "swe-1-7-medium"
    assert _normalize_model("swe-1-7") == "swe-1-7"


def test_acp_client_uses_api_key_argument() -> None:
    client = AcpClient(agent_bin="/bin/true", api_key="passed-key")
    assert client._api_key == "passed-key"


def test_acp_client_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("WINDSURF_API_KEY", "env-key")
    client = AcpClient(agent_bin="/bin/true")
    assert client._api_key == "env-key"


def test_acp_client_raises_when_no_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WINDSURF_API_KEY", raising=False)
    monkeypatch.delenv("ACP_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    agent_bin = tmp_path / "devin"
    agent_bin.write_text("#!/bin/sh\n")
    with pytest.raises(RuntimeError, match="No api_key provided"):
        AcpClient(agent_bin=str(agent_bin))


def test_route_update_collects_full_and_chunked_messages() -> None:
    """ACP may send agent_message (full) or agent_message_chunk (incremental)."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
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
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
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


def test_create_session_uses_empty_mcp_servers(monkeypatch, tmp_path: Path) -> None:
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
    client._loop = asyncio.new_event_loop()
    calls: list[tuple[str, Any]] = []

    async def fake_call(method: str, params: Any, timeout: float | None = None) -> Any:
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
    # `session/new` must not receive inline server definitions; the active list
    # is communicated to `devin acp` via `mcp_config.json` instead.
    assert new_session_calls[0][1]["mcpServers"] == []


def test_acp_error_classifier_detects_model_error() -> None:
    """A genuine 'Model not found' with a non-empty available list is AcpModelError."""
    error = {
        "code": -32002,
        "message": "Resource not found",
        "data": {"uri": "Model not found: unknown. Available models: 'swe-1-7, swe-1-6'"},
    }
    exc = _acp_error_from_response("session/set_config_option", error)
    assert isinstance(exc, AcpModelError)


def test_acp_error_classifier_treats_model_not_found_with_empty_list_as_stale() -> None:
    """`devin acp 3000.6.7` misreports a stale session as Model not found with an empty list."""
    error = {
        "code": -32002,
        "message": "Resource not found",
        "data": {"uri": "Model not found: swe-1-7. Available models: "},
    }
    exc = _acp_error_from_response("session/set_config_option", error)
    assert isinstance(exc, AcpSessionStaleError)
    assert not isinstance(exc, AcpModelError)


def test_acp_error_classifier_detects_stale_session() -> None:
    """A 'Session not found' -32002 error is classified as AcpSessionStaleError."""
    error = {
        "code": -32002,
        "message": "Resource not found",
        "data": {"uri": "Session not found: foo-bar"},
    }
    exc = _acp_error_from_response("session/set_config_option", error)
    assert isinstance(exc, AcpSessionStaleError)
    assert not isinstance(exc, AcpModelError)


def test_acp_error_classifier_detects_mcp_error() -> None:
    """An invalid mcpServers -32602 error is classified as AcpMcpError."""
    error = {
        "code": -32602,
        "message": "Invalid params",
        "data": {
            "error": "data did not match any variant of untagged enum McpServer",
            "json": {"mcpServers": []},
            "phase": "deserialization",
        },
    }
    exc = _acp_error_from_response("session/new", error)
    assert isinstance(exc, AcpMcpError)


def test_is_stale_session_error_with_typed_exceptions() -> None:
    """The stale-session guard recognises typed exceptions and rejects model/MCP."""
    assert AcpClient._is_stale_session_error(
        AcpSessionStaleError("session/set_config_option", {"code": -32002, "message": "x"})
    )
    assert not AcpClient._is_stale_session_error(
        AcpModelError("session/set_config_option", {"code": -32002, "message": "x"})
    )
    assert not AcpClient._is_stale_session_error(
        AcpMcpError("session/new", {"code": -32602, "message": "x"})
    )
    assert not AcpClient._is_stale_session_error(
        AcpTransportError("acp.send", msg="ACP process not running")
    )


def test_is_method_not_found_precedence() -> None:
    """Only true method-not-found or session/resume-not-found are reported as missing methods."""
    assert AcpClient._is_method_not_found(
        AcpError("foo/unknown", {"code": -32601, "message": "Method not found"})
    )
    assert AcpClient._is_method_not_found(
        AcpError("session/resume", {"code": -32602, "message": "not found"})
    )
    assert not AcpClient._is_method_not_found(
        AcpError("session/new", {"code": -32602, "message": "not found"})
    )
    assert not AcpClient._is_method_not_found(
        AcpError("session/new", {"code": -32602, "message": "session/resume not found"})
    )


def test_restart_transport_unblocks_inflight_future(monkeypatch) -> None:
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
    client._transport_healthy = True
    client._initialized = True
    client._proc = None
    client._loop = asyncio.new_event_loop()
    client._thread = threading.Thread(target=client._loop.run_forever, daemon=True)
    client._thread.start()

    fut = concurrent.futures.Future()
    client._inflight_future = fut

    close_called: list[bool] = []
    ensure_started: list[bool] = []

    def fake_close() -> None:
        close_called.append(True)

    def fake_ensure_started(*a: Any, **k: Any) -> None:
        ensure_started.append(True)

    monkeypatch.setattr(client, "close", fake_close)
    monkeypatch.setattr(client, "_ensure_started", fake_ensure_started)
    monkeypatch.setattr(client, "_check_restart_backoff", lambda _reason=None: None)
    monkeypatch.setattr(client, "_record_restart_attempt", lambda _reason=None: None)

    client.restart_transport()

    assert fut.done()
    with pytest.raises(TimeoutError, match="ACP transport restarted"):
        fut.result()
    assert close_called
    assert ensure_started


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
    pid = 12345

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return 0


def test_prompt_hard_timeout_returns_partial() -> None:
    """A hard timeout produces a partial result with stop_reason='timeout'."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
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
        self.pid = 12345

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return 0


def test_restart_transport_starts_new_subprocess(monkeypatch) -> None:
    """restart_transport closes the old process and starts a new one."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")

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


def test_watchdog_kills_stuck_process() -> None:
    """If a prompt produces no stdout and exceeds the deadline, the watchdog kills devin."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", watchdog_timeout=0.01)
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._watchdog_running = True

    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = time.monotonic() - 0.1

    loop = asyncio.new_event_loop()
    prompt = _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=loop.create_future(),
        cancel_done=loop.create_future(),
    )
    client._active_prompts["s-1"] = prompt

    client._check_watchdog()

    assert inflight.done()
    assert isinstance(inflight.exception(), TimeoutError)
    assert client._proc.returncode == -9
    assert client._transport_healthy is False
    assert client._initialized is False


def test_sandbox_systemctl_wrapper_routes_to_harness(monkeypatch, tmp_path: Path) -> None:
    """The fake systemctl wrapper should send restart requests to the harness socket."""
    monkeypatch.setenv("WINDSURF_API_KEY", "test-key")
    monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
    monkeypatch.delenv("DBUS_SYSTEM_BUS_ADDRESS", raising=False)

    called: dict[str, Any] = {}
    event = threading.Event()

    def _on_restart(service: str, reason: str) -> None:
        called["service"] = service
        called["reason"] = reason
        event.set()

    client = AcpClient(agent_bin="/bin/true", api_key="test-key", on_service_restart=_on_restart)
    try:
        client._sandbox.prepare()
        assert client._sandbox.devin_home is not None

        wrapper = client._sandbox.devin_home / ".local" / "bin" / "systemctl"
        assert wrapper.exists() and os.access(wrapper, os.X_OK)

        env = os.environ.copy()
        env["DIPLOID_CONTROL_SOCKET"] = str(client._control_socket_path)
        env["PATH"] = (
            f"{client._sandbox.devin_home / '.local' / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        )
        env.pop("DBUS_SESSION_BUS_ADDRESS", None)
        env.pop("DBUS_SYSTEM_BUS_ADDRESS", None)

        result = subprocess.run(
            [str(wrapper), "--user", "restart", "vesper.service"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "scheduled via harness" in result.stdout

        assert event.wait(timeout=5), "on_service_restart callback was not called"
        assert called["service"] == "vesper.service"
    finally:
        client.close()


def test_watchdog_kills_stuck_control_call() -> None:
    """If a control call produces no stdout for the watchdog timeout, the watchdog kills devin."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", watchdog_timeout=0.01)
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._watchdog_running = True

    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = time.monotonic() + 60.0
    client._last_request_at = time.monotonic() - 0.1
    client._pending[1] = client._loop.create_future()

    client._check_watchdog()

    assert inflight.done()
    assert isinstance(inflight.exception(), TimeoutError)
    assert client._proc.returncode == -9


def test_normalize_mcp_servers_drops_lean_ctx(tmp_path: Path) -> None:
    """lean-ctx entries are stripped because the shared daemon is not used."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
    client._sandbox.devin_home = tmp_path

    out = client._sandbox.normalize_mcp_servers(
        [
            {
                "name": "lean-ctx",
                "command": "lean-ctx",
                "args": [],
                "env": ["OTHER=value", "LEAN_CTX_SOCKET=/tmp/noop.sock"],
            },
            {"name": "diploid-memory", "command": "python", "args": ["-m", "memory"], "env": []},
        ]
    )

    assert len(out) == 1
    assert out[0]["name"] == "diploid-memory"
    assert out[0]["command"] == "python"


def test_watchdog_waits_for_prompt_soft_timeout() -> None:
    """The prompt inactivity watchdog does not fire before soft_timeout."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", watchdog_timeout=0.01)
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._watchdog_running = True

    now = time.monotonic()
    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = now + 60.0
    client._last_progress_at = now
    client._last_stdout_at = now

    loop = asyncio.new_event_loop()
    prompt = _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=loop.create_future(),
        cancel_done=loop.create_future(),
        soft_timeout=10.0,
        started_at=now,
    )
    client._active_prompts["s-1"] = prompt

    client._check_watchdog()

    assert not inflight.done()


def test_watchdog_does_not_kill_prompt_after_soft_timeout() -> None:
    """The prompt watchdog does not kill a prompt just because soft_timeout passed.

    A soft timeout triggers a client-side cancel and a partial reply; the subprocess
    is only killed if the process exits or the explicit hard `timeout` deadline
    is exceeded.
    """
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", watchdog_timeout=0.01)
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._watchdog_running = True

    now = time.monotonic()
    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = now + 60.0
    client._last_progress_at = now - 45.0

    loop = asyncio.new_event_loop()
    prompt = _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=loop.create_future(),
        cancel_done=loop.create_future(),
        soft_timeout=10.0,
        started_at=now - 45.0,
    )
    client._active_prompts["s-1"] = prompt

    client._check_watchdog()

    assert not inflight.done()
    assert client._proc.returncode is None


def test_watchdog_does_not_kill_prompt_that_keeps_printing_past_soft_timeout() -> None:
    """A prompt that is still producing output past soft_timeout is not killed."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", watchdog_timeout=10.0)
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._watchdog_running = True

    now = time.monotonic()
    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = now + 60.0
    client._last_progress_at = now - 0.1  # output is still recent

    loop = asyncio.new_event_loop()
    prompt = _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=loop.create_future(),
        cancel_done=loop.create_future(),
        soft_timeout=10.0,
        started_at=now - 45.0,
    )
    client._active_prompts["s-1"] = prompt

    client._check_watchdog()

    assert not inflight.done()
    assert client._proc.returncode is None


def test_watchdog_respects_per_call_control_timeout() -> None:
    """The control-call watchdog waits for the call's own deadline, not the global default."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", control_timeout=0.01)
    client._watchdog_running = True
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]

    now = time.monotonic()
    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = now + 60.0
    client._last_request_at = now - 25.0
    client._last_control_call_deadline = now + 60.0
    client._pending[1] = client._loop.create_future()

    client._check_watchdog()

    assert not inflight.done()


def test_watchdog_kills_per_call_control_timeout() -> None:
    """The control-call watchdog fires when the per-call deadline is exceeded."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", control_timeout=120.0)
    client._watchdog_running = True
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]

    now = time.monotonic()
    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = now + 60.0
    client._last_request_at = now - 25.0
    client._last_control_call_deadline = now - 1.0
    client._pending[1] = client._loop.create_future()

    client._check_watchdog()

    assert inflight.done()
    assert isinstance(inflight.exception(), TimeoutError)
    assert client._proc.returncode == -9


def test_health_reports_busy_despite_failed_last_call() -> None:
    """A prompt or control call in progress should look healthy to the harness watchdog.

    After a transient failure such as a stale session, ``_transport_healthy`` may
    be False while the next call is in flight. The health probe must not return
    False for a busy service, or the external harness watchdog will restart a
    legitimately running turn.
    """
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._initialized = True
    client._transport_healthy = False

    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = time.monotonic() + 60.0

    assert client.health() is True

    # Once the call finishes and the transport is still marked unhealthy,
    # health should reflect the actual transport state.
    client._inflight_future = None
    client._inflight_deadline = 0.0
    assert client.health() is False


def test_health_returns_false_when_deadline_exceeded() -> None:
    """A long call that has exceeded its hard deadline is not healthy."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._initialized = True
    client._transport_healthy = False

    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = time.monotonic() - 1.0

    assert client.health() is False


class ExitedFakeProcess(FakeProcess):
    """Fake process that has already exited."""

    returncode = 1


def test_watchdog_fires_when_child_process_exits() -> None:
    """The watchdog triggers recovery immediately when the subprocess has exited."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key")
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = ExitedFakeProcess()  # type: ignore[assignment]
    client._watchdog_running = True

    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = time.monotonic() + 60.0

    client._check_watchdog()

    assert inflight.done()
    assert isinstance(inflight.exception(), TimeoutError)


def test_watchdog_does_not_fire_for_long_silent_prompt() -> None:
    """A prompt that has not exceeded soft_timeout + grace is not killed even
    if it produces no progress updates."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", watchdog_timeout=0.01)
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._watchdog_running = True

    now = time.monotonic()
    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = now + 60.0
    client._last_progress_at = now - 115.0
    client._last_stdout_at = now

    loop = asyncio.new_event_loop()
    prompt = _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=loop.create_future(),
        cancel_done=loop.create_future(),
        soft_timeout=120.0,
        started_at=now - 115.0,
    )
    client._active_prompts["s-1"] = prompt

    client._check_watchdog()

    # 115s < 120s + 30s grace, so the watchdog should not fire.
    assert not inflight.done()


def test_watchdog_does_not_kill_silent_prompt_before_soft_timeout() -> None:
    """A prompt that is silent but alive is only killed by its own soft/hard deadline."""
    client = AcpClient(agent_bin="/bin/true", api_key="test-key", watchdog_timeout=0.01)
    client._control_timeout = 0.05
    client._loop = asyncio.new_event_loop()
    client._proc = FakeProcess()  # type: ignore[assignment]
    client._watchdog_running = True

    now = time.monotonic()
    inflight: concurrent.futures.Future[Any] = concurrent.futures.Future()
    client._inflight_future = inflight
    client._inflight_deadline = now + 600.0
    client._last_stdout_at = now - 40.0

    loop = asyncio.new_event_loop()
    prompt = _Prompt(
        session_id="s-1",
        prompt_id=1,
        text="hi",
        future=loop.create_future(),
        cancel_done=loop.create_future(),
        soft_timeout=600.0,
        started_at=now - 40.0,
    )
    client._active_prompts["s-1"] = prompt

    client._check_watchdog()

    # 40s of silence is well short of soft_timeout + grace (600 + 30).
    assert not inflight.done()


def test_restart_history_persists_to_disk(tmp_path: Path) -> None:
    """AcpRestartHistory loads previous restart attempts from disk."""
    path = tmp_path / "acp_restart_history.jsonl"
    now = time.time()
    with path.open("w") as f:
        f.write(json.dumps({"timestamp": now}) + "\n")
        f.write(json.dumps({"timestamp": now - 1.0}) + "\n")

    store = AcpRestartHistory(path, window=60.0)
    assert store.count() == 2

    # A stale entry outside the window is ignored after the next prune.
    with path.open("w") as f:
        f.write(json.dumps({"timestamp": now - 120.0}) + "\n")
    store = AcpRestartHistory(path, window=60.0)
    assert store.count() == 0


def test_restart_transport_backoff() -> None:
    """restart_transport refuses to cycle more than max_restarts in the backoff window."""
    client = AcpClient(
        agent_bin="/bin/true",
        api_key="test-key",
        max_restarts=2,
        restart_backoff_window=60.0,
    )

    # Seed the restart history with two recent restarts.
    client._restart_history_store._history = [
        {"timestamp": time.time(), "reason": "transport_error"},
        {"timestamp": time.time() - 1.0, "reason": "transport_error"},
    ]

    with pytest.raises(AcpTransportError):
        client.restart_transport()


def test_restart_transport_mcp_backoff_is_separate() -> None:
    """MCP-change restarts use their own, higher backoff limit."""
    client = AcpClient(
        agent_bin="/bin/true",
        api_key="test-key",
        max_restarts=2,
        max_mcp_restarts=4,
        restart_backoff_window=60.0,
    )

    # Seed two transport-error restarts, which exhausts the transport budget.
    client._restart_history_store._history = [
        {"timestamp": time.time(), "reason": "transport_error"},
        {"timestamp": time.time() - 1.0, "reason": "transport_error"},
    ]

    # A plain restart should still be rejected.
    with pytest.raises(AcpTransportError):
        client.restart_transport()

    # An MCP-change restart is bucketed separately and should still be allowed.
    client._check_restart_backoff("mcp_change")
