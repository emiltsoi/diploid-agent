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
from diploid_agent.models import ChatResult, WakeEvent
from diploid_agent.runtime.agent_runtime import AgentRuntime
from diploid_agent.runtime.event_bus import Event
from diploid_agent.transport.http import create_app


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path, *, harness_api_key: str | None = None) -> Config:
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
        secrets=Secrets(WINDSURF_API_KEY="test-key", HARNESS_API_KEY=harness_api_key),
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
    # Bare TestClient, not `with` — entering the context runs the anyio-portal
    # lifespan (runtime.start/shutdown) whose teardown can wedge on Linux
    # (CI freeze: parked in portal thread.join inside __exit__). None of these
    # tests need the lifespan running.
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    app = create_app(_make_config(tmp_path), runtime)
    return TestClient(app)


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


def test_ui_page_open_while_data_endpoints_stay_keyed(tmp_path: Path) -> None:
    """A browser navigation can't set X-API-Key — /ui must serve openly so the
    page's own setup form can collect the key for the data calls."""
    config = _make_config(tmp_path, harness_api_key="harness-secret")
    runtime = AgentRuntime(config)
    runtime.engine = FakeEngine()
    keyed = TestClient(create_app(config, runtime))
    assert keyed.get("/ui").status_code == 200
    assert keyed.get("/chats").status_code == 403
    assert keyed.get("/events/webui:desk").status_code == 403
    assert keyed.get("/chats", headers={"X-API-Key": "harness-secret"}).status_code == 200


def test_session_op_ack_lands_in_event_log(client: TestClient) -> None:
    """A queued session_op's ack is delivered outside the turn pipeline — it
    must still append a result event or the replay silently misses it."""
    runtime = client.app.state.runtime
    event = runtime.wake_queue.enqueue(
        WakeEvent(
            id="",
            chat_id="webui:desk",
            reason="session_op",
            priority=1,
            scheduled_at=time.time() - 1,
            payload={"op": "resume", "session_number": 9},
            silent=True,
            created_at=time.time(),
            ready=True,
        )
    )
    result = runtime.wake("webui:desk", event_id=event.id)
    assert result.reply == ""

    events = runtime.events_since("webui:desk")["events"]
    assert [e["kind"] for e in events] == ["result"]
    assert events[0]["payload"]["reply"] == "Session 9 not found for this chat."


def test_wake_delivery_without_turn_lands_in_event_log(client: TestClient) -> None:
    """A non-turn wake result (dispatch diagnostic) delivered by the lifecycle
    handler must append a result event too — same silent-miss class."""
    runtime = client.app.state.runtime
    dispatch = runtime.dispatch_store.add("webui:desk", "sess-x", context="ctx")
    event = runtime.wake_queue.enqueue(
        WakeEvent(
            id="",
            chat_id="webui:desk",
            reason="dispatch",
            priority=1,
            scheduled_at=time.time() - 1,
            payload={"dispatch_id": dispatch.id},
            silent=True,
            created_at=time.time(),
            ready=True,
        )
    )
    runtime._lifecycle._handle_timer_fired(
        Event(
            type="timer.fired",
            payload={
                "event_id": event.id,
                "chat_id": "webui:desk",
                "reason": "dispatch",
                "silent": True,
            },
        )
    )

    events = runtime.events_since("webui:desk")["events"]
    assert [e["kind"] for e in events] == ["result"]
    assert events[0]["payload"]["reply"] == "No active session for this chat."
