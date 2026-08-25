"""Tests for the live /config endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from devin_fleet_harness.config import (
    Config,
    DevinConfig,
    HarnessConfig,
    PersonaConfig,
    PluginConfig,
    Secrets,
    TelegramConfig,
)
from devin_fleet_harness.transport.http import create_app


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
            memory={"backend": "file", "hindsight": {"api_key": "hindsight-secret"}},  # type: ignore[arg-type]
            skills={"shared_root": str(tmp_path / "shared")},  # type: ignore[arg-type]
            plugins=[
                PluginConfig(
                    name="curriculum",
                    module="devin_fleet_harness.plugins.curriculum",
                    prompt_slot="persona_state",
                    state_file="chat_curriculum.json",
                    max_prompt_chars=1024,
                ),
            ],  # type: ignore[arg-type]
            telegram=TelegramConfig(enabled=False, token="dummy-token"),
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key", HARNESS_API_KEY="harness-secret"),
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


def test_config_get_excludes_secrets(client: TestClient) -> None:
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "secrets" not in data
    assert data["harness"]["telegram"]["enabled"] is False
    assert data["harness"]["telegram"]["token"] == "***"
    assert data["harness"]["memory"]["hindsight"]["api_key"] == "***"
    assert data["harness"]["plugins"][0]["name"] == "curriculum"


def test_config_patch_telegram(client: TestClient) -> None:
    response = client.patch(
        "/config",
        json={
            "telegram": {
                "enabled": True,
                "stream_thoughts": True,
                "stream_chunk_interval": 0.5,
            }
        },
        headers={"X-API-Key": "harness-secret"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["harness"]["telegram"]["enabled"] is True
    assert data["harness"]["telegram"]["stream_thoughts"] is True
    assert data["harness"]["telegram"]["stream_chunk_interval"] == 0.5

    # Change should persist in memory.
    response = client.get("/config")
    assert response.json()["harness"]["telegram"]["enabled"] is True


def test_config_patch_plugins(client: TestClient) -> None:
    response = client.patch(
        "/config",
        json={
            "plugins": [
                {
                    "name": "curriculum",
                    "enabled": False,
                    "config": {"foo": "bar"},
                }
            ]
        },
        headers={"X-API-Key": "harness-secret"},
    )
    assert response.status_code == 200
    data = response.json()
    plugin = data["harness"]["plugins"][0]
    assert plugin["enabled"] is False
    assert plugin["config"] == {"foo": "bar"}
