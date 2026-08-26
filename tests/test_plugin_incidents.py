from pathlib import Path

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.plugin_incidents import PluginIncidentStore
from acp_fleet_harness.plugins.manager import PluginManager


def test_incident_store_records_and_retrieves(tmp_path: Path) -> None:
    store = PluginIncidentStore(tmp_path / "incidents.jsonl")
    store.record(plugin="p1", phase="start", error="boom")
    assert len(store.recent()) == 1
    assert store.recent()[0]["plugin"] == "p1"


def test_manager_records_incident_on_failed_start(tmp_path: Path) -> None:
    store = PluginIncidentStore(tmp_path / "incidents.jsonl")
    bad = PluginConfig(name="bad", enabled=True, module="definitely.not.a.module")
    pm = PluginManager(
        plugins=[bad],
        sessions_root=tmp_path,
        instance_id="i1",
        instance_started_at=0.0,
        incident_store=store,
    )
    pm._get_or_create("c1", bad)
    assert any(i["plugin"] == "bad" for i in store.recent())
