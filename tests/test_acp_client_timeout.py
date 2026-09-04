"""Tests for AcpClient timeout handling."""

from pathlib import Path

import pytest

from diploid_agent.acp_client import AcpClient, AcpPromptResult


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> AcpClient:
    """Return a minimally initialized AcpClient with a fake devin binary."""
    monkeypatch.setenv("WINDSURF_API_KEY", "test-key")
    fake_devin = tmp_path / "devin"
    fake_devin.write_text("#!/bin/sh\necho '{\"ok\": true}'\n")
    fake_devin.chmod(0o755)
    c = AcpClient(
        model="swe-1-7",
        agent_bin=str(fake_devin),
        api_key="test-key",
        timeout=18000.0,
    )
    # Pretend the transport is up so we can test the timeout path.
    c._transport_healthy = True
    return c


def test_create_session_marks_transport_unhealthy_on_timeout(
    client, tmp_path: Path, monkeypatch
) -> None:
    """When _prompt returns a hard timeout, _transport_healthy must become False."""
    timed_out_result = AcpPromptResult(
        reply="",
        session_id="test-session",
        stop_reason="timeout",
        partial=True,
        timed_out=True,
    )
    monkeypatch.setattr(client, "_run", lambda coro, timeout=None: timed_out_result)

    client.create_session("test prompt", cwd=tmp_path)

    assert client._transport_healthy is False


def test_send_message_marks_transport_unhealthy_on_timeout(client, monkeypatch) -> None:
    """Same for follow-up messages."""
    timed_out_result = AcpPromptResult(
        reply="partial",
        session_id="test-session",
        stop_reason="timeout",
        partial=True,
        timed_out=True,
    )
    monkeypatch.setattr(client, "_run", lambda coro, timeout=None: timed_out_result)

    client.send_message("test-session", "continue")

    assert client._transport_healthy is False


def test_create_session_keeps_transport_healthy_on_normal_result(
    client, tmp_path: Path, monkeypatch
) -> None:
    """A completed result must not mark the transport unhealthy."""
    completed_result = AcpPromptResult(
        reply="done",
        session_id="test-session",
        stop_reason=None,
        partial=False,
        timed_out=False,
    )
    monkeypatch.setattr(client, "_run", lambda coro, timeout=None: completed_result)

    client.create_session("test prompt", cwd=tmp_path)

    assert client._transport_healthy is True
