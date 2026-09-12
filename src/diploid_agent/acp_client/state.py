"""Shared mutable state for the ACP client component graph.

``AcpClientState`` is the single home for state that ``AcpClient``,
``AcpTransport``, ``AcpSessionOps``, and ``PromptWatchdog`` previously shared
by reaching backwards through ``client._*`` private attributes. The client
owns the object; components receive it at construction and access state via
``self._state._x``.

Field names keep their leading underscore on purpose: tests monkeypatch and
fake ``client._x`` names, and both ``AcpClient`` and ``AcpTransport`` expose
the same names via ``_StateAttr`` descriptors that resolve onto this object.
A test fake without a ``_state`` attribute is itself used as the state
namespace, so ``self._state._x`` still lands on the fake's ``_x`` attribute.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diploid_agent.acp_client.types import _Prompt


class _StateAttr:
    """Alias a ``_x`` attribute name onto ``instance._state``.

    Applied at class level on ``AcpClient`` and ``AcpTransport`` so existing
    ``client._x`` / ``transport._x`` seams keep working while the storage
    lives on the shared ``AcpClientState``.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:
            raise AttributeError(self.name)
        return getattr(instance._state, self.name)

    def __set__(self, instance: Any, value: Any) -> None:
        setattr(instance._state, self.name, value)


@dataclass
class AcpClientState:
    """Mutable state shared across ACP client components."""

    # Locks. ``_lifecycle_lock`` serializes transport start/stop/restart and is
    # always acquired before ``_lock`` (order: ``_lifecycle_lock`` -> ``_lock``).
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _lifecycle_lock: threading.RLock = field(default_factory=threading.RLock)

    # Session/prompt state (written by AcpClient + AcpSessionOps).
    _next_id: int = 0
    _active_prompts: dict[str, _Prompt] = field(default_factory=dict)
    _session_models: dict[str, str] = field(default_factory=dict)
    _pending_cancels: set[str] = field(default_factory=set)
    _model_options: list[str] | None = None
    _mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    # Generation of the transport for which transport.stop was last logged.
    _logged_stop_gen: int = -1

    # Transport state (written by AcpTransport, read by client + watchdog).
    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None
    _proc: asyncio.subprocess.Process | None = None
    _reader_task: asyncio.Task[None] | None = None
    _stderr_task: asyncio.Task[None] | None = None
    _last_stdout_at: float = 0.0
    _last_progress_at: float = 0.0
    _last_request_at: float = 0.0
    _last_control_call_deadline: float = 0.0
    _inflight_future: concurrent.futures.Future[Any] | None = None
    _inflight_deadline: float = 0.0
    _pending: dict[int, asyncio.Future[dict[str, Any]]] = field(default_factory=dict)
    _initialized: bool = False
    _transport_healthy: bool = False
    # Set once this generation can no longer deliver responses: killed
    # child, dead stdout reader, or close in progress.  call()/_send()
    # fail fast on a terminated transport -- a request registered after
    # the unblock sweep would otherwise sit in ``_pending`` forever
    # (observed in production: session/resume issued 2 ms after a
    # watchdog kill hung for the full call timeout).
    _terminated: bool = False
    # Legacy name kept for the client._restart_history alias; the live store
    # is ``AcpClient._restart_history_store``.
    _restart_history: list[Any] = field(default_factory=list)
    # Monotonic generation counter, bumped on every transport start so
    # lifecycle events can be attributed to a specific child process.
    generation: int = 0
