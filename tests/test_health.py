"""Tests for the enriched /health endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PluginConfig,
    Secrets,
)
from diploid_agent.transport.http import create_app


def _test_config(tmp_path: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=Path(__file__).parent / "fixtures" / "test-pilot",
            fleet_root=Path(__file__).parent / "fixtures" / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "test-sessions",
            session_store_path=tmp_path / "test-sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
            skills={"shared_root": str(tmp_path / "shared")},  # type: ignore[arg-type]
            plugins=[
                PluginConfig(
                    name="curriculum",
                    module="diploid_plugins.curriculum",
                    prompt_slot="self_state",
                    state_file="chat_curriculum.json",
                    max_prompt_chars=1024,
                ),
            ],  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


class _FakeEngine:
    def health(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["swe-1-7"]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(_test_config(tmp_path))
    app.state.harness.client = _FakeEngine()  # type: ignore[assignment]
    return TestClient(app)


def test_health_returns_components(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["uptime_seconds"] >= 0
    assert "components" in data
    for name in ("acp", "hindsight", "telegram"):
        assert data["components"][name]["healthy"] is True
    assert data["pending_restart"] is None


class _IdleEngine(_FakeEngine):
    """A lazily-started transport that has not been asked to work yet."""

    def health(self) -> bool:
        return False

    def transport_started(self) -> bool:
        return False


class _FailedEngine(_FakeEngine):
    """A transport that started and then went unhealthy."""

    def health(self) -> bool:
        return False

    def transport_started(self) -> bool:
        return True


def test_health_acp_not_started_reports_idle(tmp_path: Path) -> None:
    """A fresh instance's unstarted transport is idle, not an error."""
    app = create_app(_test_config(tmp_path))
    app.state.harness.client = _IdleEngine()  # type: ignore[assignment]
    data = TestClient(app).get("/health").json()
    assert data["components"]["acp"]["status"] == "idle"
    assert data["components"]["acp"]["healthy"] is True
    assert data["status"] == "ok"


def test_health_acp_started_failure_still_errors(tmp_path: Path) -> None:
    """A started-then-unhealthy transport still reports degraded."""
    app = create_app(_test_config(tmp_path))
    app.state.harness.client = _FailedEngine()  # type: ignore[assignment]
    data = TestClient(app).get("/health").json()
    assert data["components"]["acp"]["status"] == "error"
    assert data["components"]["acp"]["healthy"] is False
    assert data["status"] == "degraded"


def test_health_reports_pending_restart(tmp_path: Path) -> None:
    """A draining restart surfaces service/reason metadata via /health."""
    app = create_app(_test_config(tmp_path))
    app.state.harness.client = _FakeEngine()  # type: ignore[assignment]
    runtime = app.state.harness
    runtime._state.pending_restart = {
        "service": "test-pilot.service",
        "reason": "maintenance",
        "chat_id": "chat-1",
        "draining_since": 123.0,
    }
    runtime._state.restart_draining.set()
    data = TestClient(app).get("/health").json()
    pending = data["pending_restart"]
    assert pending["service"] == "test-pilot.service"
    assert pending["reason"] == "maintenance"
    assert pending["chat_id"] == "chat-1"
    assert pending["active_turns"] == 0
    assert pending["session_ops_pending"] is False
