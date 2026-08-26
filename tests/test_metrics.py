"""Tests for the Prometheus-style metrics collector and endpoints."""

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
from diploid_agent.metrics import MetricsCollector
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
                    module="diploid_agent.plugins.curriculum",
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


def test_metrics_collector_counts_and_renders() -> None:
    metrics = MetricsCollector(prefix="harness")
    metrics.inc("turns_total")
    metrics.inc("turns_total", value=2)
    metrics.set("active_turns", value=1)
    text = metrics.render()
    assert "harness_turns_total 3.0" in text
    assert "harness_active_turns 1" in text
    assert "# TYPE harness_turns_total counter" in text
    assert "# TYPE harness_active_turns gauge" in text


def test_metrics_collector_labels() -> None:
    metrics = MetricsCollector(prefix="harness")
    metrics.inc("tokens_total", value=10, kind="input")
    metrics.inc("tokens_total", value=20, kind="output")
    text = metrics.render()
    assert 'harness_tokens_total{kind="input"} 10.0' in text
    assert 'harness_tokens_total{kind="output"} 20.0' in text


def test_prometheus_endpoint_returns_text(client: TestClient) -> None:
    response = client.get("/prometheus")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
