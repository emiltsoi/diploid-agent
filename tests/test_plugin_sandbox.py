import json

from fastapi.testclient import TestClient

from acp_fleet_harness.config import Config, DevinConfig, HarnessConfig, PersonaConfig, PluginConfig
from acp_fleet_harness.runtime.agent_runtime import AgentRuntime
from acp_fleet_harness.transport.http import create_app


class FakeEngine:
    def prompt(self, *a, **k):
        from acp_fleet_harness.engine import TurnResult
        return TurnResult(reply="ok", session_id="s1")

    def list_models(self):
        return ["m1"]

    def restart(self):
        pass

    def is_stale_session_error(self, exc):
        return False

    def close(self):
        pass


def _make_config(tmp_path):
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(name="test", profile_root=tmp_path / "persona"),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            session_prune_enabled=False,
            plugins=[
                PluginConfig(name="continuity", enabled=True, module="acp_fleet_harness.plugins.continuity")
            ],
        ),
    )


def test_plugin_sandbox_valid_module(tmp_path):
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    client = TestClient(create_app(_make_config(tmp_path), runtime))
    resp = client.post("/plugin/sandbox", json={"module": "acp_fleet_harness.plugins.continuity"})
    assert resp.status_code == 200
    data = json.loads(resp.json()["reply"])
    assert data["ok"] is True


def test_plugin_sandbox_invalid_module(tmp_path):
    runtime = AgentRuntime(_make_config(tmp_path))
    runtime.engine = FakeEngine()
    client = TestClient(create_app(_make_config(tmp_path), runtime))
    resp = client.post("/plugin/sandbox", json={"module": "os"})
    assert resp.status_code == 200
    data = json.loads(resp.json()["reply"])
    assert data["ok"] is False
