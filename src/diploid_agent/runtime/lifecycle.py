"""Runtime lifecycle: startup, shutdown, restart notices."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from diploid_agent.plugins.contexts import ShutdownContext
from diploid_agent.runtime.state import RuntimeState

logger = logging.getLogger(__name__)


class RuntimeLifecycle:
    """Start and stop the runtime's background services."""

    def __init__(
        self,
        *,
        state: RuntimeState,
        config: Any,
        lock: Any,
        event_bus: Any,
        wake_queue: Any,
        instance_manager: Any,
        task_engine: Any,
        timer_service: Any,
        typing: Any,
        restart: Any,
        outbox: Any,
        plugins: Any,
        chat_store: Any,
        memory_managers: dict[str, Any],
        store: dict[str, Any],
        ingress_handlers: dict[str, Any],
        instance_id: str,
        instance_started_at: float,
        on_event_fn: Callable[[Any], None],
        runtime_api: Any,
    ) -> None:
        self._state = state
        self._config = config
        self._lock = lock
        self._event_bus = event_bus
        self._wake_queue = wake_queue
        self._instance_manager = instance_manager
        self._task_engine = task_engine
        self._timer_service = timer_service
        self._typing = typing
        self._restart = restart
        self._outbox = outbox
        self._plugins = plugins
        self._chat_store = chat_store
        self._memory_managers = memory_managers
        self._store = store
        self._ingress_handlers = ingress_handlers
        self._instance_id = instance_id
        self._instance_started_at = instance_started_at
        self._on_event_fn = on_event_fn
        # Handed to the mesh ingress factory; not a general back-reference.
        self._runtime_api = runtime_api

    def _load_mesh_ingress(self) -> None:
        """Load the configured mesh ingress handler if mesh is enabled."""
        from diploid_agent.transport.ingress import load_ingress_handler

        mesh = self._config.harness.mesh
        if not mesh.enabled:
            return
        try:
            handler = load_ingress_handler(mesh.ingress_module, runtime=self._runtime_api)
            self._ingress_handlers["mesh"] = handler
        except Exception:
            logger.exception("Failed to load mesh ingress handler: %s", mesh.ingress_module)

    def start(self) -> None:
        """Start background services. Idempotent."""
        if self._state.started:
            return
        self._state.started = True
        if not self._event_bus.running:
            self._event_bus.start()
        self._event_bus.subscribe(self._on_event_fn)

        # Drop auto-continue wakes that were created by a previous process.
        # Queued user messages and other system wakes are kept, and the
        # conversation/session state used for resume is not touched.
        if self._wake_queue is not None:
            try:
                count = self._wake_queue.cancel_older_than(
                    self._instance_started_at,
                    reason="auto_continue",
                )
                if count:
                    logger.info(
                        "Cancelled %d stale auto-continue wake(s) on startup",
                        count,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to cancel stale auto-continue wakes",
                    exc_info=exc,
                )

        if self._config.harness.timer.enabled:
            self._timer_service.start()
        self._instance_manager.start_heartbeat()
        self._load_mesh_ingress()
        self._outbox._send_restart_notices()

    def send_restart_notices(self) -> None:
        """Notify recently active chats that the service has restarted."""
        self._outbox._send_restart_notices()

    def create_direct_notifier(self) -> Any:
        """Create a notifier that bypasses the outbox if possible."""
        return self._outbox._create_direct_notifier()

    def shutdown(self, drain_timeout: float = 120.0) -> None:
        """Drain active turns, notify plugins, and stop background workers."""
        self._state.started = False
        self._state.restart_draining.set()
        try:
            if not self._restart._wait_for_active_turns(drain_timeout):
                logger.warning(
                    "Shutdown drain cap (%.0fs) expired with turn(s) still active",
                    drain_timeout,
                )
        except Exception:
            logger.exception("Failed to drain active turns during shutdown")
        self._timer_service.stop()
        self._typing.stop()
        try:
            self._event_bus.unsubscribe(self._on_event_fn)
        except ValueError:
            pass
        self._instance_manager.stop_heartbeat()
        self._task_engine.shutdown(wait=False)
        self._event_bus.stop()
        now = time.time()
        for chat_id in list(self._store.keys()):
            record = self._chat_store._active_record(chat_id)
            self._plugins.on_shutdown(
                chat_id,
                ShutdownContext(
                    chat_id=chat_id,
                    record=record,
                    reason="shutdown",
                    now=now,
                    instance_id=self._instance_id,
                    instance_started_at=self._instance_started_at,
                ),
            )
        self._plugins.stop_all()

        with self._lock:
            managers = list(self._memory_managers.values())
            self._memory_managers.clear()
        for manager in managers:
            manager.close()
