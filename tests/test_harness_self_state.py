"""Integration tests for the self_state plugin through ConversationHarness."""

from __future__ import annotations

from pathlib import Path

from diploid_agent.acp_client import AcpPromptResult
from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PluginConfig,
    Secrets,
)
from diploid_agent.harness import ConversationHarness


def _make_config(tmp_path: Path, fixture_root: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=fixture_root,
            fleet_root=tmp_path / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
            session_prune_enabled=False,
            plugins=[
                PluginConfig(
                    name="self_state",
                    enabled=True,
                    module="diploid_plugins.self_state",
                    prompt_slot="self_state",
                    prompt_order=90,
                    state_file="chat_self_state.md",
                    max_prompt_chars=512,
                ),
            ],
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def test_self_state_survives_stale_session_rehydration(monkeypatch, tmp_path: Path) -> None:
    """A <self_state> block is stripped, saved, and reinjected after rehydration."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    create_count = [0]
    create_prompts: list[str] = []
    send_count = [0]

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        create_count[0] += 1
        create_prompts.append(prompt)
        if create_count[0] == 1:
            return AcpPromptResult(
                reply="I was explaining restarts. <self_state>I was explaining restarts.</self_state>",
                session_id="session-1",
            )
        return AcpPromptResult(reply="Ready.", session_id=f"session-{create_count[0]}")

    def fake_send_message(session_id, prompt, *, cwd=None, model=None, **kwargs):
        send_count[0] += 1
        raise RuntimeError("ACP session/prompt failed: Session not found")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    result1 = harness.process("chat-ss", "hello")
    assert result1.session_number == 1
    assert result1.session_id == "session-1"
    assert "<self_state>" not in result1.reply
    assert "I was explaining restarts." in result1.reply

    result2 = harness.process("chat-ss", "follow-up")
    assert result2.session_number == 1
    assert result2.session_id == "session-2"
    assert send_count[0] == 1
    assert create_count[0] == 2
    assert "I was explaining restarts." in create_prompts[1]
