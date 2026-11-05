"""Tests for the webui endpoints: /chats, /events, and /chat notify mirroring."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    NotificationsConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
    TimerConfig,
)
from diploid_agent.models import ChatResult
from diploid_agent.runtime.agent_runtime import AgentRuntime
from diploid_agent.transport.http import create_app


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
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
            timer=TimerConfig(enabled=True, interval_seconds=0.1),
            notifications=NotificationsConfig(enabled=True, outbox_delivery=True),
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
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


@pytest.fixture
def client(tmp_path: Path):
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    with TestClient(create_app(_make_config(tmp_path), runtime)) as client:
        yield client


def test_chats_lists_sessions_with_kind(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "42", "message": "hi"})
    client.post("/chat", json={"chat_id": "webui:desk", "message": "hi"})

    resp = client.get("/chats")
    assert resp.status_code == 200
    by_id = {c["chat_id"]: c for c in resp.json()["chats"]}
    assert by_id["42"]["kind"] == "telegram"
    assert by_id["webui:desk"]["kind"] == "webui"
    assert by_id["42"]["sessions"] == 1
    assert by_id["42"]["next_seq"] == 3  # turn_started + result


def test_chat_turn_writes_turn_started_and_result_events(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "webui:desk", "message": "hello love"})

    resp = client.get("/events/webui:desk")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert [e["kind"] for e in events] == ["turn_started", "result"]
    assert events[0]["payload"]["user_message"] == "hello love"
    assert events[1]["payload"]["reply"] == "ok"


def test_events_since_filters_and_reports_next_seq(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "webui:desk", "message": "one"})
    first = client.get("/events/webui:desk").json()
    last_seq = first["events"][-1]["seq"]

    client.post("/chat", json={"chat_id": "webui:desk", "message": "two"})
    resp = client.get("/events/webui:desk", params={"since": last_seq})
    events = resp.json()["events"]
    assert [e["kind"] for e in events] == ["turn_started", "result"]
    assert resp.json()["next_seq"] == events[-1]["seq"] + 1


def test_events_long_poll_wakes_on_new_turn(client: TestClient) -> None:
    def sender() -> None:
        time.sleep(0.05)
        client.post("/chat", json={"chat_id": "webui:desk", "message": "hi"})

    threading.Thread(target=sender, daemon=True).start()
    start = time.perf_counter()
    resp = client.get("/events/webui:desk", params={"since": 0, "wait": 5.0})
    assert time.perf_counter() - start < 4.0
    assert resp.json()["events"][0]["kind"] == "turn_started"


def test_chat_notify_true_enqueues_result_for_telegram_chat(client: TestClient) -> None:
    """notify=true mirrors the reply into the outbox queue for delivery."""
    resp = client.post("/chat", json={"chat_id": "42", "message": "hi", "notify": True})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "ok"

    outbox = client.get("/outbox/42", params={"wait": 0.5})
    assert outbox.json()["result"]["reply"] == "ok"


def test_chat_default_does_not_enqueue(client: TestClient) -> None:
    """notify=false (default) keeps the reply in-band only."""
    client.post("/chat", json={"chat_id": "42", "message": "hi"})
    outbox = client.get("/outbox/42", params={"wait": 0.05})
    assert outbox.json()["result"] is None


def test_service_started_event_appended_for_recent_chats(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "webui:desk", "message": "hi"})
    runtime = client.app.state.runtime
    runtime._outbox._send_restart_notices()

    events = client.get("/events/webui:desk").json()["events"]
    assert events[-1]["kind"] == "service_started"


def test_ui_page_serves(client: TestClient) -> None:
    resp = client.get("/ui")
    assert resp.status_code == 200
    assert "Common Room" in resp.text


def test_transient_enqueue_lands_in_event_log(client: TestClient) -> None:
    runtime = client.app.state.runtime
    runtime._enqueue_outbox("42", ChatResult(reply="System: [mesh] hi", transient=True))
    events = client.get("/events/42").json()["events"]
    assert events[0]["kind"] == "system"
    assert events[0]["payload"]["text"] == "System: [mesh] hi"


def test_events_and_turn_endpoints_cover_webui_chat(client: TestClient) -> None:
    """A webui-only chat streams through /turn and lands in /events end to end."""
    client.post("/chat", json={"chat_id": "webui:desk", "message": "hi"})
    status = client.get("/turn/webui:desk").json()
    assert status == {"chat_id": "webui:desk", "status": "idle"}
    kinds = [e["kind"] for e in client.get("/events/webui:desk").json()["events"]]
    assert kinds == ["turn_started", "result"]
