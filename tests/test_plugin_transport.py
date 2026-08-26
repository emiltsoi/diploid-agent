"""Tests for plugin control over HTTP."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    PluginConfig,
    Secrets,
    TimerConfig,
)
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
            plugins=[PluginConfig(name="json", enabled=True)],
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


def test_plugin_list_after_chat(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "chat-1", "message": "hello"})
    resp = client.get("/plugins/chat-1")
    assert resp.status_code == 200
    plugins = resp.json()["plugins"]
    assert any(p["name"] == "json" and p["enabled"] is True for p in plugins)


def test_plugin_enable_and_disable(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "chat-1", "message": "hello"})

    resp = client.post(
        "/plugin/enable", json={"chat_id": "chat-1", "name": "json", "enabled": False}
    )
    assert resp.status_code == 200
    assert "disabled" in resp.json()["reply"]

    resp = client.get("/plugins/chat-1")
    plugins = {p["name"]: p for p in resp.json()["plugins"]}
    assert plugins["json"]["enabled"] is False

    resp = client.post(
        "/plugin/enable", json={"chat_id": "chat-1", "name": "json", "enabled": True}
    )
    assert resp.status_code == 200
    assert "enabled" in resp.json()["reply"]


def test_plugin_reload(client: TestClient) -> None:
    client.post("/chat", json={"chat_id": "chat-1", "message": "hello"})
    resp = client.post("/plugin/reload", json={"chat_id": "chat-1", "name": "json"})
    assert resp.status_code == 200
    assert "reloaded" in resp.json()["reply"]
