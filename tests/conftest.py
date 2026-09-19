"""Shared test fixtures."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from diploid_agent.config import EngineConfig

_orig_engine_init = EngineConfig.__init__


def _engine_init(self, **data):
    """Default ACP resume to off in unit tests unless the test sets it explicitly."""
    if "acp_resume_enabled" not in data:
        data["acp_resume_enabled"] = False
    _orig_engine_init(self, **data)


@pytest.fixture(autouse=True)
def _acp_resume_off_for_tests(monkeypatch) -> None:
    monkeypatch.setattr(EngineConfig, "__init__", _engine_init)


@pytest.fixture(autouse=True)
def _isolated_control_dir(monkeypatch):
    """Scope the ACP control-socket namespace to this test.

    ``ControlListener`` binds ``$DIPLOID_CONTROL_DIR/diploid-ctl-<service>/
    control.sock`` and soft-fails when a live foreign listener holds the path.
    Tests share fixed service names (``unknown.service``, persona names), so a
    per-test dir keeps parallel pytest-xdist workers — and same-named
    listeners across tests — from probing or stealing each other's sockets.

    The dir is a shallow ``/tmp/dpctl-t-*`` rather than ``tmp_path``: AF_UNIX
    ``sun_path`` is ~107 chars and a worker tmpdir would push the socket path
    past the limit.
    """
    base = Path(tempfile.mkdtemp(prefix="dpctl-t-"))
    monkeypatch.setenv("DIPLOID_CONTROL_DIR", str(base))
    yield
    shutil.rmtree(base, ignore_errors=True)
