"""Mutable scalar state shared by runtime components."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    """Scalars and flags that multiple runtime components read and mutate.

    Dict-shaped shared state (``active_turns``, ``memory_managers``, …) is
    passed to components by reference instead of living here — only values
    that need a mutable box belong in this object. ``AgentRuntime`` exposes
    property aliases (``runtime._started`` etc.) over these fields so tests
    and turn code that write them keep working.
    """

    started: bool = False
    # Set once a graceful restart begins draining: no new turns may start so a
    # stream of queued messages cannot keep the process alive until the cap.
    restart_draining: threading.Event = field(default_factory=threading.Event)
    # Legible detail of the in-progress graceful restart (service/reason/
    # start time): set alongside restart_draining, cleared on failure, and
    # surfaced via /health so a deferred restart is observable rather than
    # only inferable from logs and uptime.
    pending_restart: dict[str, Any] | None = None
    # Rate-limit for ACP-subprocess-initiated service restarts.
    last_service_restart_at: float = 0.0
    service_restart_cooldown_seconds: float = 60.0
