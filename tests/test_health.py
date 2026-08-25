"""Tests for the enriched /health endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from acp_fleet_harness.config import (
    Config,
    DevinConfig,
    HarnessConfig,
    PersonaConfig,
    PluginConfig,
    Secrets,
)
from acp_fleet_harness.transport.http import create_app


def _test_config(tmp_path: Path) -> Config:
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
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
                    module="acp_fleet_harness.plugins.curriculum",
                    prompt_slot="persona_state",
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
