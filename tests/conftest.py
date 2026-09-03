"""Shared test fixtures."""

from __future__ import annotations

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
