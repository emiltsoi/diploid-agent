"""Tests for WakeQueue integration in dispatch and continue_turn."""

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


class FakeEngine:
    def prompt(self, *a, **k):
        from diploid_agent.engine import TurnResult

        return TurnResult(reply="done", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def test_dispatch_creates_pending_wake_event(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    runtime.new_session("chat-1")
    result = runtime.dispatch("chat-1", context="do work")
    dispatch_id = result.dispatch_id
    assert dispatch_id
    event = next(iter(runtime.wake_queue.pending(chat_id="chat-1")))
    assert event.payload["dispatch_id"] == dispatch_id
    assert event.ready is False

    # Complete the dispatch
    runtime.continue_turn(dispatch_id, "work done")
    assert runtime.wake_queue.get(event.id) is None
