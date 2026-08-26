from pathlib import Path

from diploid_agent.config import Config, DiploidConfig, HarnessConfig, PersonaConfig, PluginConfig
from diploid_agent.runtime.agent_runtime import AgentRuntime


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


def _make_config(tmp_path: Path, plugins: list[PluginConfig]) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(name="test", profile_root=tmp_path / "persona"),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            session_prune_enabled=False,
            plugins=plugins,
        ),
    )


def test_validate_contract_rejects_builtin_module(tmp_path: Path) -> None:
    from diploid_agent.plugins.manager import PluginManager

    pm = PluginManager(
        plugins=[],
        sessions_root=tmp_path,
        instance_id="i1",
        instance_started_at=0.0,
    )
    errors = pm.validate_contract("os")
    assert any("must expose a 'Plugin' class" in e for e in errors)


def test_runtime_health_reports_plugin_health(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        [PluginConfig(name="continuity", enabled=True, module="diploid_agent.plugins.continuity")],
    )
    runtime = AgentRuntime(config)
    runtime.engine = FakeEngine()
    health = runtime.health()
    assert "plugins" in health["components"]
    assert health["components"]["plugins"]["healthy"] is True
    details = health["components"]["plugins"]["details"]
    assert any(d["name"] == "continuity" for d in details)
