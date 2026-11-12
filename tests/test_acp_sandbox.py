"""Tests for the isolated ACP HOME sandbox."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from diploid_agent._compat import IS_WINDOWS
from diploid_agent.acp_client.sandbox import AcpSandbox


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows AppData shadowing is Windows-only")
def test_prepare_windows_home_junctions_shared_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Windows devin resolves config via %APPDATA%: the isolated home must
    shadow it so the harness-owned mcp_config.json is the one read, while
    shared state (sessions.db, locks, acp-messages) stays junctioned to the
    real dirs — resumes and custody locks must survive child restarts."""
    real_roaming = tmp_path / "Roaming"
    real_local = tmp_path / "Local"
    real_devin = real_roaming / "devin"
    for sub in ("cli", "User", "mcp"):
        (real_devin / sub).mkdir(parents=True)
    (real_devin / "cli" / "sessions.db").write_bytes(b"sqlite")
    (real_devin / "config.json").write_text('{"version": 1}')
    (real_local / "devin").mkdir(parents=True)
    real_pkgs = real_roaming / "Python" / "Python314" / "site-packages"
    real_pkgs.mkdir(parents=True)
    (real_pkgs / "diploid_mesh").mkdir()
    monkeypatch.setenv("APPDATA", str(real_roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(real_local))

    sandbox = AcpSandbox()
    sandbox.prepare(
        mcp_servers=[
            {
                "name": "diploid-mesh",
                "command": "python",
                "args": ["-m", "diploid_mesh.mcp"],
                "env": ["MESH_AGENT_NAME=iris"],
            }
        ]
    )
    try:
        home = sandbox.devin_home
        assert home is not None
        iso_devin = home / "AppData" / "Roaming" / "devin"

        # Isolated mcp_config.json at the path Windows devin actually reads.
        config = json.loads((iso_devin / "mcp_config.json").read_text())
        assert config["mcpServers"]["diploid-mesh"]["env"] == {
            "MESH_AGENT_NAME": "iris",
            "PYTHONIOENCODING": "utf-8",
        }
        # Real user config.json copied into the shadow AppData.
        assert json.loads((iso_devin / "config.json").read_text()) == {"version": 1}

        # Shared state resolves through the junctions into the real dirs.
        assert (iso_devin / "cli" / "sessions.db").read_bytes() == b"sqlite"
        assert (iso_devin / "User").is_dir()
        assert (iso_devin / "mcp").is_dir()
        # User site-packages stays reachable — children resolve imports.
        assert (
            iso_devin.parent / "Python" / "Python314" / "site-packages" / "diploid_mesh"
        ).is_dir()
        if hasattr(os.path, "isjunction"):
            assert os.path.isjunction(iso_devin / "cli")
    finally:
        sandbox.cleanup()

    # Cleanup removes the junctions and the temp home, never the real store.
    assert (real_devin / "cli" / "sessions.db").read_bytes() == b"sqlite"
    assert not home.exists()


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only")
def test_windows_cleanup_unlinks_junctions_before_rmtree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """os.path.islink() does not detect junctions, so shutil.rmtree could
    descend into them and empty the real target. cleanup() must rmdir the
    junctions first — this test is the regression guard for that order."""
    real_roaming = tmp_path / "Roaming"
    real_devin = real_roaming / "devin"
    (real_devin / "cli").mkdir(parents=True)
    (real_devin / "cli" / "sessions.db").write_bytes(b"precious")
    monkeypatch.setenv("APPDATA", str(real_roaming))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))

    sandbox = AcpSandbox()
    sandbox.prepare(mcp_servers=[])
    sandbox.cleanup()

    assert (real_devin / "cli" / "sessions.db").read_bytes() == b"precious"


def test_prepare_creates_home_under_durable_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The isolated HOME must not live under $TMPDIR: macOS sweeps stale tmp
    files while the devin child still runs — an aged-out mcp_config.json left
    sessions resolving an empty MCP map (2026-11-07). Homes go under
    DIPLOID_ACP_HOME_ROOT (a cache dir in production) instead."""
    base = tmp_path / "acp-homes"
    monkeypatch.setenv("DIPLOID_ACP_HOME_ROOT", str(base))

    sandbox = AcpSandbox()
    sandbox.prepare(mcp_servers=[{"name": "s", "command": "cmd", "args": []}])
    try:
        home = sandbox.devin_home
        assert home is not None
        assert home.parent == base
        assert home.name.startswith("acp-home-")
        config = json.loads((home / ".config" / "devin" / "mcp_config.json").read_text())
        assert "s" in config["mcpServers"]
    finally:
        sandbox.cleanup()

    assert not home.exists()


def _make_orphan(base: Path, name: str, age_seconds: float) -> Path:
    """Create a fake leftover acp-home-* backdated to look orphaned."""
    orphan = base / name
    config_dir = orphan / ".config" / "devin"
    config_dir.mkdir(parents=True)
    config = config_dir / "mcp_config.json"
    config.write_text('{"mcpServers": {}}')
    old = time.time() - age_seconds
    os.utime(config, (old, old))
    os.utime(config_dir, (old, old))
    os.utime(orphan, (old, old))
    return orphan


def test_prepare_sweeps_stale_orphan_homes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Killed or crashed children leak acp-home-* dirs forever under the
    durable base — prepare() must sweep ones past the orphan age while
    leaving fresh homes and non-matching dirs alone."""
    base = tmp_path / "acp-homes"
    base.mkdir()
    monkeypatch.setenv("DIPLOID_ACP_HOME_ROOT", str(base))

    stale = _make_orphan(base, "acp-home-stale00", 31 * 24 * 60 * 60)
    fresh = _make_orphan(base, "acp-home-fresh00", 60)
    (base / "not-ours").mkdir()

    sandbox = AcpSandbox()
    sandbox.prepare(mcp_servers=[])
    try:
        assert not stale.exists()
        assert fresh.exists()
        assert (base / "not-ours").is_dir()
        assert sandbox.devin_home is not None and sandbox.devin_home.exists()
    finally:
        sandbox.cleanup()


def test_prepare_never_sweeps_orphan_with_live_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live child's home is never reapable however stale it looks — the
    base is shared across seats and a quiet seat can exceed the orphan age
    between session ops on a still-running child."""
    base = tmp_path / "acp-homes"
    base.mkdir()
    monkeypatch.setenv("DIPLOID_ACP_HOME_ROOT", str(base))

    orphan = _make_orphan(base, "acp-home-livepid", 31 * 24 * 60 * 60)
    (orphan / ".acp-child-pid").write_text(str(os.getpid()))
    old = time.time() - 31 * 24 * 60 * 60
    os.utime(orphan, (old, old))

    sandbox = AcpSandbox()
    sandbox.prepare(mcp_servers=[])
    try:
        assert orphan.exists()
    finally:
        sandbox.cleanup()


def test_prepare_sweeps_orphan_with_dead_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A dead pid marker is no protection — the home reaps by age."""
    import subprocess
    import sys

    base = tmp_path / "acp-homes"
    base.mkdir()
    monkeypatch.setenv("DIPLOID_ACP_HOME_ROOT", str(base))

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    orphan = _make_orphan(base, "acp-home-deadpid", 31 * 24 * 60 * 60)
    (orphan / ".acp-child-pid").write_text(str(proc.pid))
    old = time.time() - 31 * 24 * 60 * 60
    os.utime(orphan, (old, old))

    sandbox = AcpSandbox()
    sandbox.prepare(mcp_servers=[])
    try:
        assert not orphan.exists()
    finally:
        sandbox.cleanup()


def test_write_mcp_configs_resurrects_missing_parents(tmp_path: Path) -> None:
    """write_mcp_configs must mkdir the config dirs it writes into — a reaped
    or reset home has no .config tree, and the session-op rewrite is the
    self-heal: it must not fail silently on a missing parent."""
    sandbox = AcpSandbox()
    sandbox.devin_home = tmp_path  # bare home: no config dirs exist

    sandbox.write_mcp_configs([{"name": "s", "command": "cmd"}])

    config = json.loads((tmp_path / ".config" / "devin" / "mcp_config.json").read_text())
    assert "s" in config["mcpServers"]


def test_prepare_keeps_orphan_with_fresh_config_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Liveness is the config heartbeat, not the dir mtime: session ops
    rewrite mcp_config.json on every establish, so a long-lived home whose
    top dir has gone stale must NOT be swept while it still serves ops."""
    base = tmp_path / "acp-homes"
    base.mkdir()
    monkeypatch.setenv("DIPLOID_ACP_HOME_ROOT", str(base))

    orphan = _make_orphan(base, "acp-home-quiet00", 31 * 24 * 60 * 60)
    # Rewrite the config fresh WITHOUT touching the dir mtime — exactly what
    # a session op does on a home that has been quiet since spawn.
    config = orphan / ".config" / "devin" / "mcp_config.json"
    config.write_text('{"mcpServers": {}}')
    old = time.time() - 31 * 24 * 60 * 60
    os.utime(orphan, (old, old))

    sandbox = AcpSandbox()
    sandbox.prepare(mcp_servers=[])
    try:
        assert orphan.exists()
    finally:
        sandbox.cleanup()


def test_write_mcp_configs_none_preserves_existing_file(tmp_path: Path) -> None:
    """mcp_servers=None means 'leave the file alone', not 'write empty' —
    session ops rewrite on every establish and must not wipe a config that
    was written with an explicit list earlier."""
    sandbox = AcpSandbox()
    sandbox.devin_home = tmp_path
    (tmp_path / ".config" / "devin").mkdir(parents=True)
    (tmp_path / ".codeium" / "windsurf").mkdir(parents=True)
    if IS_WINDOWS:
        (tmp_path / "AppData" / "Roaming" / "devin").mkdir(parents=True)

    sandbox.write_mcp_configs([{"name": "s", "command": "cmd"}])
    before = (tmp_path / ".config" / "devin" / "mcp_config.json").read_text()
    sandbox.write_mcp_configs(None)

    assert (tmp_path / ".config" / "devin" / "mcp_config.json").read_text() == before


def test_write_mcp_configs_empty_list_writes_empty_map(tmp_path: Path) -> None:
    """An explicit empty list is a real signal — probes run with no servers."""
    sandbox = AcpSandbox()
    sandbox.devin_home = tmp_path
    (tmp_path / ".config" / "devin").mkdir(parents=True)
    (tmp_path / ".codeium" / "windsurf").mkdir(parents=True)

    sandbox.write_mcp_configs([])

    config = json.loads((tmp_path / ".config" / "devin" / "mcp_config.json").read_text())
    assert config == {"mcpServers": {}}


def test_write_mcp_configs_normalizes_env_list(tmp_path: Path) -> None:
    sandbox = AcpSandbox()
    sandbox.devin_home = tmp_path
    (tmp_path / ".config" / "devin").mkdir(parents=True)
    (tmp_path / ".codeium" / "windsurf").mkdir(parents=True)
    if IS_WINDOWS:
        (tmp_path / "AppData" / "Roaming" / "devin").mkdir(parents=True)
    sandbox.write_mcp_configs([{"name": "s", "command": "cmd", "args": [], "env": ["A=1", "B=2"]}])
    config = json.loads((tmp_path / ".config" / "devin" / "mcp_config.json").read_text())
    expected = {"A": "1", "B": "2"}
    if IS_WINDOWS:
        expected["PYTHONIOENCODING"] = "utf-8"
    assert config["mcpServers"]["s"]["env"] == expected


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-only")
def test_write_mcp_configs_respects_explicit_stdio_encoding(tmp_path: Path) -> None:
    """A server that sets PYTHONIOENCODING itself must not be overridden."""
    sandbox = AcpSandbox()
    sandbox.devin_home = tmp_path
    for sub in (".config/devin", ".codeium/windsurf", "AppData/Roaming/devin"):
        (tmp_path / sub).mkdir(parents=True)
    sandbox.write_mcp_configs(
        [{"name": "s", "command": "cmd", "args": [], "env": ["PYTHONIOENCODING=latin-1"]}]
    )
    config = json.loads((tmp_path / "AppData/Roaming/devin/mcp_config.json").read_text())
    assert config["mcpServers"]["s"]["env"]["PYTHONIOENCODING"] == "latin-1"
