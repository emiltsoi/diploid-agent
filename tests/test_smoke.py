"""Real chat-level smoke test against a live `devin` ACP child.

This test is skipped automatically when `devin` is not in `$PATH` or when
`devin` cannot authenticate, so it can be committed and run locally or in CI
only when those prerequisites are satisfied.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from diploid_agent.acp_client import AcpError, AcpTransportError
from diploid_agent.acp_client.utils import _load_windsurf_api_key
from diploid_agent.config import (
    Config,
    DiploidConfig,
    EngineConfig,
    HarnessConfig,
    PersonaConfig,
    Secrets,
)
from diploid_agent.harness import ConversationHarness


def _smoke_config(tmp_path: Path, fixture_root: Path) -> Config:
    return Config(
        diploid=DiploidConfig(
            bin=shutil.which("devin") or "devin",
            model="swe-1-7",
        ),
        engine=EngineConfig(
            provider="diploid",
            model="swe-1-7",
            timeout=60.0,
            soft_timeout=15.0,
            acp_startup_timeout=30.0,
            acp_control_timeout=30.0,
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
        secrets=Secrets(WINDSURF_API_KEY=_load_windsurf_api_key() or "test-key"),
    )


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("devin") is None,
    reason="devin is not in $PATH",
)
def test_real_chat_turn(tmp_path: Path) -> None:
    """A single user message produces a real reply and a session id."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _smoke_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    try:
        result = harness.process("smoke-chat", "hello")
    except (AcpError, AcpTransportError) as exc:
        message = str(exc).lower()
        if "authentication" in message or "api key" in message or "invalid api" in message:
            pytest.skip(f"devin is not authenticated: {exc}")
        raise

    assert result.reply
    assert result.session_id is not None
    assert result.session_number == 1
