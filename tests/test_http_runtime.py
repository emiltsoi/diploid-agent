"""Tests for the runtime daemon HTTP endpoints."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

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
    TaskConfig,
    TimerConfig,
    WakerConfig,
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


def test_runtime_status_returns_daemon_state(client: TestClient) -> None:
    resp = client.get("/runtime/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["event_bus_running"] is True
    assert body["timer_running"] is True
    assert body["active_chat_count"] == 0


def test_timer_endpoint_creates_scheduled_wake(client: TestClient, monkeypatch) -> None:
    scheduled_at = time.time() + 3600
    resp = client.post(
        "/timer",
        json={
            "chat_id": "chat-1",
            "reason": "test",
            "scheduled_at": scheduled_at,
            "payload": {"foo": "bar"},
            "silent": True,
            "priority": 2,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "event_id" in data

    # WakeQueue is on the runtime behind the ConversationHarness wrapper.
    runtime = client.app.state.runtime
    event = runtime.wake_queue.get(data["event_id"])
    assert event is not None
    assert event.chat_id == "chat-1"
    assert event.reason == "test"
    assert event.scheduled_at == scheduled_at
    assert event.payload == {"foo": "bar"}
    assert event.silent is True
    assert event.priority == 2


def test_runtime_stop_and_start_are_idempotent(client: TestClient) -> None:
    assert client.post("/runtime/stop").json() == {"ok": True}
    status = client.get("/runtime/status").json()
    assert status["timer_running"] is False

    assert client.post("/runtime/start").json() == {"ok": True}
    status = client.get("/runtime/status").json()
    assert status["event_bus_running"] is True
    assert status["timer_running"] is True


def test_task_config_get_and_update(client: TestClient) -> None:
    # The default config should be returned.
    resp = client.get("/task/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["workers"] == 4
    assert body["shell_timeout"] == 60.0
    assert body["enabled_types"] == ["shell", "noop", "acp", "subagent"]

    # Update workers and enabled types.
    resp = client.post(
        "/task/config",
        json={
            "workers": 2,
            "shell_timeout": 30.0,
            "enabled_types": ["shell", "noop"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["workers"] == 2
    assert body["shell_timeout"] == 30.0
    assert body["enabled_types"] == ["shell", "noop"]

    # The runtime's TaskEngine should see the same object.
    runtime = client.app.state.runtime
    assert runtime.get_task_config().workers == 2
    assert "acp" not in runtime.get_task_config().enabled_types


def test_task_config_persists_and_reloads(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    runtime = AgentRuntime(config)
    runtime.engine = FakeEngine()
    runtime.update_task_config(
        TaskConfig(workers=2, shell_timeout=30.0, enabled_types=["shell", "noop"])
    )

    override_path = tmp_path / "sessions.jsonl"  # session_store_path parent
    # session_store_path is tmp_path/sessions.jsonl, so runtime-overrides.yaml is in tmp_path
    overrides_file = override_path.parent / "runtime-overrides.yaml"
    assert overrides_file.exists()

    # A fresh runtime with the same config should load the persisted task config.
    fresh = AgentRuntime(config)
    fresh.engine = FakeEngine()
    assert fresh.get_task_config().workers == 2
    assert fresh.get_task_config().shell_timeout == 30.0
    assert fresh.get_task_config().enabled_types == ["shell", "noop"]


def test_task_config_persistence_is_optional(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    fresh = AgentRuntime(config)
    fresh.engine = FakeEngine()
    assert fresh.get_task_config().workers == 4
    assert fresh.get_task_config().enabled_types == ["shell", "noop", "acp", "subagent"]


@pytest.mark.parametrize(
    "content",
    [
        "not yaml : [",  # invalid YAML
        "task: []",  # task is not a dict
        "task:\n  workers: not_a_number",  # invalid field value
    ],
)
def test_task_config_persistence_skips_malformed_overrides(tmp_path: Path, content: str) -> None:
    config = _make_config(tmp_path)
    overrides_file = tmp_path / "runtime-overrides.yaml"
    overrides_file.write_text(content)
    fresh = AgentRuntime(config)
    fresh.engine = FakeEngine()
    assert fresh.get_task_config().workers == 4
    assert fresh.get_task_config().enabled_types == ["shell", "noop", "acp", "subagent"]


def test_task_config_update_rejects_invalid_workers(client: TestClient) -> None:
    resp = client.post(
        "/task/config",
        json={
            "workers": 0,
            "shell_timeout": 30.0,
            "enabled_types": ["shell"],
        },
    )
    assert resp.status_code == 422


def test_task_config_update_rejects_invalid_timeout(client: TestClient) -> None:
    resp = client.post(
        "/task/config",
        json={
            "workers": 2,
            "shell_timeout": -1.0,
            "enabled_types": ["shell"],
        },
    )
    assert resp.status_code == 422


def test_task_config_update_rejects_invalid_types(client: TestClient) -> None:
    resp = client.post(
        "/task/config",
        json={
            "workers": 2,
            "shell_timeout": 30.0,
            "enabled_types": ["shell", "bad"],
        },
    )
    assert resp.status_code == 422


def test_agent_runtime_update_task_config_validates_in_place(client: TestClient) -> None:
    runtime = client.app.state.runtime
    cfg = runtime.get_task_config()
    with pytest.raises(ValueError):
        cfg.workers = 0


def test_task_config_validates_enabled_types_at_assignment(client: TestClient) -> None:
    cfg = client.app.state.runtime.get_task_config()
    with pytest.raises(ValueError):
        cfg.enabled_types = ["shell", "bad"]


def test_task_config_partial_update_preserves_other_fields(client: TestClient) -> None:
    client.post(
        "/task/config",
        json={
            "workers": 2,
            "shell_timeout": 30.0,
            "enabled_types": ["shell", "noop"],
            "acp_model": "custom-model",
            "acp_timeout": 120.0,
        },
    )

    resp = client.post("/task/config", json={"workers": 6})
    assert resp.status_code == 200
    body = resp.json()
    assert body["workers"] == 6
    assert body["shell_timeout"] == 30.0
    assert body["enabled_types"] == ["shell", "noop"]
    assert body["acp_model"] == "custom-model"
    assert body["acp_timeout"] == 120.0


def test_task_config_update_rejects_empty_acp_model(client: TestClient) -> None:
    resp = client.post(
        "/task/config",
        json={
            "workers": 2,
            "shell_timeout": 30.0,
            "enabled_types": ["shell"],
            "acp_model": "   ",
        },
    )
    assert resp.status_code == 422


def test_waker_config_get_and_update(client: TestClient) -> None:
    resp = client.get("/waker/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["retry_after"] == 30.0

    resp = client.post(
        "/waker/config",
        json={"enabled": True, "retry_after": 60.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["retry_after"] == 60.0

    # Partial update should preserve other waker fields.
    resp = client.post("/waker/config", json={"retry_after": 90.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["retry_after"] == 90.0


def test_restart_endpoint_calls_runtime_restart(client: TestClient, monkeypatch) -> None:
    called: list[str] = []

    def fake_restart(chat_id: str) -> ChatResult:
        called.append(chat_id)
        return ChatResult(reply="Restarted")

    monkeypatch.setattr(client.app.state.runtime, "restart", fake_restart)

    resp = client.post("/restart", json={"chat_id": "12345"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Restarted"
    assert called == ["12345"]


def test_graceful_restart_endpoint_calls_runtime_graceful_restart(
    client: TestClient, monkeypatch
) -> None:
    called: list[tuple[str, str, str]] = []

    def fake_graceful_restart(chat_id: str, service: str, reason: str = "") -> ChatResult:
        called.append((chat_id, service, reason))
        return ChatResult(reply=f"Restarting {service}")

    monkeypatch.setattr(
        client.app.state.runtime, "graceful_service_restart", fake_graceful_restart
    )

    resp = client.post(
        "/graceful-restart",
        json={"chat_id": "12345", "service": "vesper.service", "reason": "telegram"},
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Restarting vesper.service"
    assert called == [("12345", "vesper.service", "telegram")]


def test_waker_config_persists_and_reloads(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    runtime = AgentRuntime(config)
    runtime.engine = FakeEngine()
    runtime.update_waker_config(WakerConfig(enabled=True, retry_after=45.0))

    fresh = AgentRuntime(config)
    fresh.engine = FakeEngine()
    assert fresh.get_waker_config().enabled is True
    assert fresh.get_waker_config().retry_after == 45.0


def test_timer_config_get_and_update(client: TestClient) -> None:
    resp = client.get("/timer/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["interval_seconds"] == 0.1

    resp = client.post(
        "/timer/config",
        json={"enabled": False, "interval_seconds": 10.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["interval_seconds"] == 10.0

    # Partial update should preserve other timer fields.
    resp = client.post("/timer/config", json={"interval_seconds": 15.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["interval_seconds"] == 15.0


def test_timer_config_persists_and_reloads(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    runtime = AgentRuntime(config)
    runtime.engine = FakeEngine()
    runtime.update_timer_config(TimerConfig(enabled=False, interval_seconds=7.0))

    fresh = AgentRuntime(config)
    fresh.engine = FakeEngine()
    assert fresh.get_timer_config().enabled is False
    assert fresh.get_timer_config().interval_seconds == 7.0


def test_notifications_config_get_and_update(client: TestClient) -> None:
    resp = client.get("/notifications/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["webhook_url"] is None

    resp = client.post(
        "/notifications/config",
        json={"enabled": False, "webhook_url": "https://example.com/hook"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["webhook_url"] == "https://example.com/hook"

    # Partial update should preserve other notifications fields.
    resp = client.post("/notifications/config", json={"enabled": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["webhook_url"] == "https://example.com/hook"


def test_notifications_config_persists_and_reloads(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    runtime = AgentRuntime(config)
    runtime.engine = FakeEngine()
    old_notifier = runtime.notifier
    runtime.update_notifications_config(
        NotificationsConfig(enabled=False, webhook_url="https://example.com/hook")
    )

    assert runtime.notifier is not old_notifier
    fresh = AgentRuntime(config)
    fresh.engine = FakeEngine()
    assert fresh.get_notifications_config().enabled is False
    assert fresh.get_notifications_config().webhook_url == "https://example.com/hook"


def test_notifications_webhook_url_normalizes_empty_string(client: TestClient) -> None:
    resp = client.post(
        "/notifications/config",
        json={"webhook_url": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["webhook_url"] is None


def test_task_config_update_returns_503_on_persistence_failure(
    client: TestClient, monkeypatch
) -> None:
    def _bad_save() -> bool:
        return False

    client.app.state.runtime._save_runtime_overrides = _bad_save

    resp = client.post(
        "/task/config",
        json={"workers": 2},
    )
    assert resp.status_code == 503
    assert "persistence failed" in resp.json()["detail"]


def test_task_config_update_rejects_zero_acp_timeout(client: TestClient) -> None:
    resp = client.post(
        "/task/config",
        json={
            "workers": 2,
            "shell_timeout": 30.0,
            "enabled_types": ["shell"],
            "acp_timeout": 0,
        },
    )
    assert resp.status_code == 422


def test_subagents_endpoint_returns_status(client: TestClient, monkeypatch) -> None:
    def fake_subagent_status(chat_id: str) -> dict[str, Any]:
        return {
            "chat_id": chat_id,
            "subagents": [
                {
                    "dispatch_id": "dispatch-1",
                    "status": "running",
                    "summary": "Working on it",
                    "started_at": 0.0,
                    "finished_at": None,
                }
            ],
        }

    monkeypatch.setattr(client.app.state.runtime, "subagent_status", fake_subagent_status)

    resp = client.get("/subagents/chat-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chat_id"] == "chat-1"
    assert len(body["subagents"]) == 1
    assert body["subagents"][0]["dispatch_id"] == "dispatch-1"
    assert body["subagents"][0]["status"] == "running"
