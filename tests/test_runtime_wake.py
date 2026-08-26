"""Tests for AgentRuntime wake queue and wake() integration."""

from pathlib import Path

from diploid_agent.config import Config, DiploidConfig, HarnessConfig, PersonaConfig, Secrets
from diploid_agent.runtime.agent_runtime import AgentRuntime


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=_fixture_root(),
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def test_agent_runtime_has_wake_queue_and_instance_manager(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    assert runtime.wake_queue is not None
    assert runtime.instance_manager is not None


def test_wake_silent_does_not_notify(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    sent = []
    monkeypatch.setattr(runtime.notifier, "send", lambda *a, **k: sent.append(a))

    from diploid_agent.models import WakeEvent

    e = runtime.wake_queue.enqueue(
        WakeEvent(
            id="w1",
            chat_id="chat-1",
            reason="timer",
            priority=1,
            scheduled_at=0.0,
            created_at=0.0,
            silent=True,
            ready=True,
        )
    )

    class FakeEngine:
        def prompt(self, *a, **k):
            from diploid_agent.engine import TurnResult

            return TurnResult(reply="ok", session_id="s1")

        def list_models(self):
            return ["m1"]

        def restart(self):
            pass

        def is_stale_session_error(self, exc):
            return False

        def close(self):
            pass

    runtime.engine = FakeEngine()
    result = runtime.wake("chat-1", event_id=e.id)
    assert result.reply
    assert not sent
