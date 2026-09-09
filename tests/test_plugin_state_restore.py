"""Tests for guarded plugin-state snapshot restore.

Regression coverage for the 2026-09-08 rollback bug: ``_restore_plugin_states``
copied every ``.snapshots/*.snapshot`` over the live file on every wake, so an
arbitrarily stale snapshot silently reverted newer agent writes
(``chat_working_memory.json`` and ``chat_self_state.md`` were rolled back in
production). Restore must now skip snapshots that are older than the live file
and files outside the durable set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PlanConfig,
    Secrets,
)
from diploid_agent.runtime import AgentRuntime


def _make_config(tmp_path: Path) -> Config:
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=fixture_root,
            fleet_root=tmp_path / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            plan=PlanConfig(root=tmp_path / "plans"),
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def _write(path: Path, content: str, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    os.utime(path, (mtime, mtime))


def _snapshot_dir(runtime: AgentRuntime, chat_id: str) -> Path:
    return runtime._chat_dir(chat_id) / ".snapshots"


def _set_durable(runtime: AgentRuntime, monkeypatch: pytest.MonkeyPatch, files: list[str]) -> None:
    monkeypatch.setattr(runtime._plugins, "durable_files", lambda: files)


T1 = 1_700_000_000.0
T2 = 1_700_000_100.0
T3 = 1_700_000_200.0


def test_restore_preserves_newer_live_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    chat_id = "chat1"
    _set_durable(runtime, monkeypatch, ["chat_working_memory.json"])

    live = runtime._chat_dir(chat_id) / "chat_working_memory.json"
    snapshot = _snapshot_dir(runtime, chat_id) / "chat_working_memory.json.snapshot"
    _write(snapshot, "stale snapshot content", T1)
    _write(live, "newer live content", T2)

    runtime._restore_plugin_states(chat_id)

    assert live.read_text() == "newer live content"


def test_restore_recovers_missing_live_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    chat_id = "chat1"
    _set_durable(runtime, monkeypatch, ["chat_working_memory.json"])

    snapshot = _snapshot_dir(runtime, chat_id) / "chat_working_memory.json.snapshot"
    _write(snapshot, "snapshot content", T1)

    runtime._restore_plugin_states(chat_id)

    live = runtime._chat_dir(chat_id) / "chat_working_memory.json"
    assert live.read_text() == "snapshot content"


def test_restore_applies_newer_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    chat_id = "chat1"
    _set_durable(runtime, monkeypatch, ["chat_working_memory.json"])

    live = runtime._chat_dir(chat_id) / "chat_working_memory.json"
    snapshot = _snapshot_dir(runtime, chat_id) / "chat_working_memory.json.snapshot"
    _write(live, "older live content", T1)
    _write(snapshot, "newer snapshot content", T3)

    runtime._restore_plugin_states(chat_id)

    assert live.read_text() == "newer snapshot content"


def test_restore_skips_snapshot_outside_durable_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    chat_id = "chat1"
    _set_durable(runtime, monkeypatch, ["chat_other.json"])

    live = runtime._chat_dir(chat_id) / "chat_retired_plugin.json"
    snapshot = _snapshot_dir(runtime, chat_id) / "chat_retired_plugin.json.snapshot"
    _write(live, "live content", T2)
    _write(snapshot, "frozen snapshot of a retired plugin", T1)

    runtime._restore_plugin_states(chat_id)

    assert live.read_text() == "live content"


def test_restore_always_allows_body_state_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    chat_id = "chat1"
    _set_durable(runtime, monkeypatch, [])

    live = runtime._chat_dir(chat_id) / "chat_body_state.json"
    snapshot = _snapshot_dir(runtime, chat_id) / "chat_body_state.json.snapshot"
    _write(snapshot, "snapshot body", T3)
    _write(live, "older live body", T1)

    runtime._restore_plugin_states(chat_id)

    assert live.read_text() == "snapshot body"


def test_restore_noop_without_snapshot_dir(tmp_path: Path) -> None:
    runtime = AgentRuntime(_make_config(tmp_path))
    chat_id = "chat1"
    live = runtime._chat_dir(chat_id) / "chat_working_memory.json"
    _write(live, "live content", T2)

    runtime._restore_plugin_states(chat_id)

    assert live.read_text() == "live content"


def test_snapshot_then_restore_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A snapshot taken before restart restores cleanly when the live file vanished."""
    runtime = AgentRuntime(_make_config(tmp_path))
    chat_id = "chat1"
    _set_durable(runtime, monkeypatch, ["chat_working_memory.json"])

    live = runtime._chat_dir(chat_id) / "chat_working_memory.json"
    _write(live, "pre-restart content", T2)

    runtime._snapshot_plugin_states(chat_id)
    live.unlink()

    runtime._restore_plugin_states(chat_id)

    assert live.read_text() == "pre-restart content"
