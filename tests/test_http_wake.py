"""Tests for the POST /wake HTTP endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from devin_fleet_harness.config import Config, DevinConfig, HarnessConfig, PersonaConfig, Secrets
from devin_fleet_harness.runtime.agent_runtime import AgentRuntime
from devin_fleet_harness.transport.http import create_app


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path) -> Config:
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
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
        from devin_fleet_harness.engine import TurnResult

        return TurnResult(reply="woken", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def test_post_wake(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    app = create_app(_make_config(tmp_path), runtime)
    client = TestClient(app)

    from devin_fleet_harness.models import WakeEvent

    e = runtime.wake_queue.enqueue(
        WakeEvent(
            id="w1",
            chat_id="chat-1",
            reason="timer",
            priority=1,
            scheduled_at=0.0,
            created_at=0.0,
            silent=False,
            ready=True,
        )
    )

    resp = client.post("/wake", json={"chat_id": "chat-1", "reason": "timer", "event_id": e.id})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "woken"
    assert runtime.wake_queue.get(e.id) is None


def test_post_wake_silent_does_not_notify(tmp_path: Path, monkeypatch) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    sent = []
    monkeypatch.setattr(runtime.notifier, "send", lambda *a, **k: sent.append(a))
    app = create_app(_make_config(tmp_path), runtime)
    client = TestClient(app)

    from devin_fleet_harness.models import WakeEvent

    e = runtime.wake_queue.enqueue(
        WakeEvent(
            id="w2",
            chat_id="chat-1",
            reason="timer",
            priority=1,
            scheduled_at=0.0,
            created_at=0.0,
            silent=True,
            ready=True,
        )
    )

    resp = client.post(
        "/wake", json={"chat_id": "chat-1", "reason": "timer", "event_id": e.id, "silent": True}
    )
    assert resp.status_code == 200
    assert not sent
    assert runtime.wake_queue.get(e.id) is None
