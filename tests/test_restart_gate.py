"""Tests for the agent-initiated restart gate in RuntimeRestart.

Agent doors (the ACP control socket, the ``harness_restart`` MCP tool)
converge on ``RuntimeRestart._on_service_restart``; the authorship
``restart_enabled`` toggle, a required reason, and
``harness.restart_allowed_units`` are enforced there. Operator doors
(``graceful_service_restart``) bypass the gate entirely.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    MeshConfig,
    PersonaConfig,
    PluginConfig,
    Secrets,
)
from diploid_agent.runtime.restart import RuntimeRestart
from diploid_agent.runtime.state import RuntimeState


class _Incidents:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, **kwargs) -> None:
        self.records.append(kwargs)


def _make_config(
    tmp_path: Path,
    *,
    restart_enabled: bool = True,
    restart_allowed_units: list[str] | None = None,
    fallback_chat_id: str = "chat-1",
) -> Config:
    persona_root = tmp_path / "persona"
    persona_root.mkdir(parents=True, exist_ok=True)
    for name in ("SOUL.md", "AGENTS.md", "MEMORY.md"):
        (persona_root / name).write_text(f"# {name}\n")
    harness = HarnessConfig(
        sessions_root=tmp_path / "sessions",
        session_store_path=tmp_path / "sessions.jsonl",
        plugins=[
            PluginConfig(
                name="authorship",
                enabled=True,
                module="diploid_agent.plugins.authorship",
                config={"restart_enabled": restart_enabled},
            )
        ],
        mesh=MeshConfig(fallback_chat_id=fallback_chat_id),
    )
    if restart_allowed_units is not None:
        harness.restart_allowed_units = restart_allowed_units
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(name="test-pilot", profile_root=persona_root),
        harness=harness,
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def _make_restart(
    config: Config,
    incidents: _Incidents,
    notices: list[tuple[str, str]],
) -> RuntimeRestart:
    restart = RuntimeRestart(
        config=config,
        state=RuntimeState(),
        lock=threading.RLock(),
        wake_queue=None,
        incidents=incidents,
        plugins=None,  # type: ignore[arg-type]  # drain store is empty; never called
        chat_store=None,  # type: ignore[arg-type]
        active_turns={},
        session_ops=set(),
        store={},
        instance_id="test",
        instance_started_at=time.time(),
        suppress_auto_continue_fn=lambda *a, **k: None,
        unit_exists_fn=lambda s: s.endswith(".service"),
        memory_manager=None,  # type: ignore[arg-type]
        notify_fn=lambda chat_id, text: notices.append((chat_id, text)),
    )
    # The real implementation spawns `systemd-run`; keep tests hermetic.
    restart._schedule_systemd_restart = lambda *a, **k: None  # type: ignore[method-assign]
    restart._arm_restart_watchdog = lambda *a, **k: None  # type: ignore[method-assign]
    return restart


def test_toggle_disabled_rejects(tmp_path: Path) -> None:
    incidents = _Incidents()
    notices: list[tuple[str, str]] = []
    restart = _make_restart(
        _make_config(tmp_path, restart_enabled=False), incidents, notices
    )
    status = restart._on_service_restart("test-pilot.service", "maintenance")
    assert status.startswith("rejected")
    assert "restart_enabled" in status
    assert incidents.records[-1]["phase"] == "agent_restart_gate"
    assert not notices
    assert not restart._state.restart_draining.is_set()


def test_missing_reason_rejects(tmp_path: Path) -> None:
    incidents = _Incidents()
    restart = _make_restart(_make_config(tmp_path), incidents, [])
    status = restart._on_service_restart("test-pilot.service", "   ")
    assert status.startswith("rejected")
    assert "reason" in status
    assert incidents.records[-1]["action"] == "rejected"


def test_unit_outside_allowlist_rejects(tmp_path: Path) -> None:
    incidents = _Incidents()
    restart = _make_restart(_make_config(tmp_path), incidents, [])
    status = restart._on_service_restart("cups.service", "try something")
    assert status.startswith("rejected")
    assert "restart_allowed_units" in status


def test_own_unit_allowed_by_default(tmp_path: Path) -> None:
    incidents = _Incidents()
    notices: list[tuple[str, str]] = []
    restart = _make_restart(_make_config(tmp_path), incidents, notices)
    status = restart._on_service_restart("test-pilot.service", "reload config")
    assert status == "scheduled"
    assert restart._state.restart_draining.is_set()
    # Operator notice went to the fallback chat with service + reason.
    assert notices and notices[0][0] == "chat-1"
    assert "test-pilot.service" in notices[0][1]
    assert "reload config" in notices[0][1]


def test_explicit_allowlist_permits_peer_unit(tmp_path: Path) -> None:
    restart = _make_restart(
        _make_config(
            tmp_path,
            restart_allowed_units=["test-pilot.service", "sister.service"],
        ),
        _Incidents(),
        [],
    )
    assert restart._on_service_restart("sister.service", "she asked") == "scheduled"


def test_cooldown_returns_status(tmp_path: Path) -> None:
    restart = _make_restart(_make_config(tmp_path), _Incidents(), [])
    assert restart._on_service_restart("test-pilot.service", "first") == "scheduled"
    assert restart._on_service_restart("test-pilot.service", "again") == "cooldown"


def test_missing_unit_rejected_after_gate(tmp_path: Path) -> None:
    restart = _make_restart(_make_config(tmp_path), _Incidents(), [])
    restart._unit_exists = lambda s: False  # type: ignore[method-assign]
    status = restart._on_service_restart("test-pilot.service", "maintenance")
    assert status == "rejected: no such unit"


def test_rejected_request_does_not_burn_cooldown(tmp_path: Path) -> None:
    """A 'no such unit' rejection must not consume the restart window."""
    restart = _make_restart(_make_config(tmp_path), _Incidents(), [])
    restart._unit_exists = lambda s: s != "test-pilot.service"  # type: ignore[method-assign]
    assert (
        restart._on_service_restart("test-pilot.service", "typo")
        == "rejected: no such unit"
    )
    restart._unit_exists = lambda s: True  # type: ignore[method-assign]
    assert restart._on_service_restart("test-pilot.service", "for real") == "scheduled"
