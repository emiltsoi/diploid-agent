"""Tests for the pluggable HTTP ingress extension point."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import Request, Response
from fastapi.testclient import TestClient

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    MeshConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
    TimerConfig,
)
from diploid_agent.runtime.agent_runtime import AgentRuntime
from diploid_agent.transport.http import create_app
from diploid_agent.transport.ingress import IngressHandler


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


class DummyMeshHandler(IngressHandler):
    async def handle(self, request: Request) -> Response:
        body = await request.body()
        return Response(f"mesh-ok:{body.decode()}", media_type="text/plain", status_code=202)


def _make_config(tmp_path: Path, *, mesh_enabled: bool = True) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=Path(__file__).parent / "fixtures" / "test-pilot",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
            timer=TimerConfig(enabled=True, interval_seconds=0.1),
            mesh=MeshConfig(enabled=mesh_enabled, ingress_module="tests.ingress_stub"),
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


@pytest.fixture
def client(tmp_path: Path):
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    with TestClient(create_app(_make_config(tmp_path), runtime)) as client:
        yield client


def test_ingress_generic_route(client: TestClient) -> None:
    resp = client.post("/ingress/mesh", data="hello")
    assert resp.status_code == 202
    assert resp.text == "mesh-ok:hello"


def test_mesh_receive_alias(client: TestClient) -> None:
    resp = client.post("/mesh/receive", data="hello")
    assert resp.status_code == 202
    assert resp.text == "mesh-ok:hello"


def test_openclaw_mesh_webhook_alias(client: TestClient) -> None:
    resp = client.post("/plugins/openclaw-mesh/webhook", data="hello")
    assert resp.status_code == 202
    assert resp.text == "mesh-ok:hello"


def test_ingress_unknown_protocol(client: TestClient) -> None:
    resp = client.post("/ingress/unknown", data="hello")
    assert resp.status_code == 404
