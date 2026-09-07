"""Tests for AcpClient hard-timeout -> transport-unhealthy behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from diploid_agent.acp_client import AcpClient, AcpPromptResult


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> AcpClient:
    """Return a minimally initialized AcpClient with a fake devin binary."""
    monkeypatch.setenv("WINDSURF_API_KEY", "test-key")
    fake_devin = tmp_path / "devin"
    fake_devin.write_text("#!/bin/sh\n")
    fake_devin.chmod(0o755)
    c = AcpClient(
        model="swe-1-7",
        agent_bin=str(fake_devin),
        api_key="test-key",
        timeout=18120.0,
    )
    # Start un-initialized so _ensure_started runs _start_transport.
    c._transport_healthy = False
    c._transport._initialized = False
    return c


def _make_result(stop_reason: str | None, timed_out: bool = False) -> AcpPromptResult:
    return AcpPromptResult(
        reply="partial",
        session_id="s1",
        stop_reason=stop_reason,
        partial=stop_reason is not None or timed_out,
        timed_out=timed_out,
    )


def test_create_session_marks_transport_unhealthy_on_timeout(client, monkeypatch) -> None:
    """A hard timeout (stop_reason='timeout') makes the transport unhealthy."""
    monkeypatch.setattr(client, "_ensure_started", lambda *a, **k: None)
    monkeypatch.setattr(
        client, "_run", lambda coro, timeout=None, **kw: _make_result("timeout", True)
    )

    client.create_session("test prompt")

    assert client._transport_healthy is False
    assert client._transport._is_transport_healthy() is False


def test_send_message_marks_transport_unhealthy_on_timeout(client, monkeypatch) -> None:
    """Same for follow-up messages."""
    monkeypatch.setattr(client, "_ensure_started", lambda *a, **k: None)
    monkeypatch.setattr(
        client, "_run", lambda coro, timeout=None, **kw: _make_result("timeout", True)
    )

    client.send_message("s1", "continue")

    assert client._transport_healthy is False


def test_create_session_keeps_transport_healthy_on_cancelled(client, monkeypatch) -> None:
    """A soft cancel (stop_reason='cancelled') should not kill the transport."""
    monkeypatch.setattr(client, "_ensure_started", lambda *a, **k: None)
    client._transport_healthy = True
    # timed_out is True because _soft_timeout_canceller sets prompt.timed_out,
    # but the stop_reason is cancelled.
    monkeypatch.setattr(
        client, "_run", lambda coro, timeout=None, **kw: _make_result("cancelled", True)
    )

    client.create_session("test prompt")

    assert client._transport_healthy is True


def test_create_session_keeps_transport_healthy_on_completed(client, monkeypatch) -> None:
    """A completed result should not kill the transport."""
    monkeypatch.setattr(client, "_ensure_started", lambda *a, **k: None)
    client._transport_healthy = True
    monkeypatch.setattr(
        client, "_run", lambda coro, timeout=None, **kw: _make_result(None, False)
    )

    client.create_session("test prompt")

    assert client._transport_healthy is True


def test_is_transport_healthy_respects_flag(client) -> None:
    """_is_transport_healthy must return False when _transport_healthy is False."""
    client._transport_healthy = False
    assert client._transport._is_transport_healthy() is False


def test_create_session_restarts_transport_after_timeout(client, monkeypatch) -> None:
    """After a hard timeout, the next create_session must restart the child."""
    starts = []

    async def fake_start_transport(mcp_servers: Any = None) -> None:
        starts.append(1)
        client._transport._initialized = True
        client._transport._transport_healthy = True
        client._transport._loop = mock.Mock(is_closed=lambda: False, is_running=lambda: True)
        client._transport._proc = mock.Mock(returncode=None)

    monkeypatch.setattr(client._transport, "_start_transport", fake_start_transport)
    monkeypatch.setattr(client._transport, "_kill_process_group", lambda proc: None)

    results = iter(
        [
            _make_result("timeout", True),  # first turn hard times out
            _make_result(None, False),       # second turn completes
        ]
    )

    def fake_run(coro, timeout=None, **kw):
        name = getattr(coro, "__name__", None)
        if name == "_start_transport":
            starts.append(1)
            client._transport._initialized = True
            client._transport._transport_healthy = True
            client._transport._loop = mock.Mock(is_closed=lambda: False, is_running=lambda: True)
            client._transport._proc = mock.Mock(returncode=None)
            return None
        if name in ("_create_session", "_send_message"):
            return next(results)
        raise NotImplementedError(name)

    monkeypatch.setattr(client, "_run", fake_run)

    # First call: start transport, run, timeout -> mark unhealthy.
    client.create_session("first prompt", cwd=Path("/tmp"))
    assert client._transport_healthy is False

    # Second call: _ensure_started sees unhealthy, restarts, runs, completes.
    client.create_session("second prompt", cwd=Path("/tmp"))
    assert client._transport_healthy is True
    assert len(starts) == 2
