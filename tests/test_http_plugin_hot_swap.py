"""HTTP tests for hot-swappable plugin endpoints."""

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
    Secrets,
    TimerConfig,
)
from diploid_agent.runtime.agent_runtime import AgentRuntime
from diploid_agent.transport.http import create_app


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


def _make_config(tmp_path: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(name="test", profile_root=tmp_path / "persona"),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            session_prune_enabled=False,
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
            timer=TimerConfig(enabled=False),
            plugins=[
                {
                    "name": "continuity",
                    "enabled": True,
                    "module": "diploid_agent.plugins.continuity",
                }
            ],
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


@pytest.fixture
def client(tmp_path: Path):
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    with TestClient(create_app(_make_config(tmp_path), runtime)) as client:
        yield client


def test_plugin_add_and_remove(client: TestClient) -> None:
    payload = {
        "plugin": {
            "name": "self_state",
            "enabled": True,
            "module": "diploid_agent.plugins.self_state",
            "prompt_slot": "persona_state",
            "state_file": "chat_self_state.md",
        }
    }
    resp = client.post("/plugins", json=payload)
    assert resp.status_code == 200
    assert "added" in resp.json()["reply"]

    resp = client.delete("/plugins/self_state")
    assert resp.status_code == 200
    assert "removed" in resp.json()["reply"]


def test_plugin_add_duplicate_rejected(client: TestClient) -> None:
    payload = {
        "plugin": {
            "name": "json_state",
            "enabled": True,
        }
    }
    resp = client.post("/plugins", json=payload)
    assert resp.status_code == 200

    resp = client.post("/plugins", json=payload)
    assert resp.status_code == 400
    assert "already exists" in resp.json()["detail"]


def test_plugin_dry_run_validates_module(client: TestClient) -> None:
    # Valid module.
    resp = client.post(
        "/plugins",
        json={
            "plugin": {
                "name": "self_state",
                "enabled": True,
                "module": "diploid_agent.plugins.self_state",
                "prompt_slot": "persona_state",
            },
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    assert "Dry run OK" in resp.json()["reply"]

    # Invalid/built-in module.
    resp = client.post(
        "/plugins",
        json={
            "plugin": {
                "name": "os",
                "enabled": True,
                "module": "os",
            },
            "dry_run": True,
        },
    )
    assert resp.status_code == 400


def test_plugin_update_validates_module(client: TestClient) -> None:
    resp = client.patch(
        "/plugins/continuity",
        json={
            "name": "continuity",
            "plugin": {"module": "builtins"},
            "dry_run": True,
        },
    )
    assert resp.status_code == 400

    resp = client.patch(
        "/plugins/continuity",
        json={
            "name": "continuity",
            "plugin": {"module": "diploid_agent.plugins.continuity"},
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    assert "Dry run OK" in resp.json()["reply"]


def test_plugin_toggle_and_rollback(client: TestClient) -> None:
    resp = client.post(
        "/plugins",
        json={"plugin": {"name": "json_state", "enabled": True}},
    )
    assert resp.status_code == 200

    resp = client.post(
        "/plugins/continuity/toggle",
        json={"name": "continuity", "enabled": False},
    )
    assert resp.status_code == 200
    assert "disabled" in resp.json()["reply"]

    resp = client.post("/config/rollback", json={"steps": 1})
    assert resp.status_code == 200
    assert "Rolled back" in resp.json()["reply"]

    resp = client.post("/config/rollback", json={"steps": 100})
    assert resp.status_code == 400
    assert "No earlier configuration" in resp.json()["detail"]


def test_unknown_plugin_operations(client: TestClient) -> None:
    assert client.delete("/plugins/unknown").status_code == 400
    assert (
        client.post(
            "/plugins/unknown/toggle", json={"name": "unknown", "enabled": False}
        ).status_code
        == 400
    )
