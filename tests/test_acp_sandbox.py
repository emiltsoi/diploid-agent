"""Tests for the isolated ACP HOME sandbox."""

from __future__ import annotations

import json
import os
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
