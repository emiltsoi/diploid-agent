"""Runtime lifecycle: startup, shutdown, restart notices."""

from __future__ import annotations

import logging
import time
from typing import Any

from diploid_agent.plugins.contexts import ShutdownContext
from diploid_agent.runtime.component import RuntimeComponent

logger = logging.getLogger(__name__)


class RuntimeLifecycle(RuntimeComponent):
    """Start and stop the runtime's background services."""

    def _load_mesh_ingress(self) -> None:
        """Load the configured mesh ingress handler if mesh is enabled."""
        from diploid_agent.transport.ingress import load_ingress_handler

        mesh = self._runtime.config.harness.mesh
        if not mesh.enabled:
            return
        try:
            handler = load_ingress_handler(mesh.ingress_module, runtime=self._runtime)
            self._runtime.register_ingress_handler("mesh", handler)
        except Exception:
            logger.exception("Failed to load mesh ingress handler: %s", mesh.ingress_module)

    def start(self) -> None:
        """Start background services. Idempotent."""
        if self._runtime._started:
            return
        self._runtime._started = True
        if not self._runtime.event_bus.running:
            self._runtime.event_bus.start()
        self._runtime.event_bus.subscribe(self._runtime._on_event)

        # Drop auto-continue wakes that were created by a previous process.
        # Queued user messages and other system wakes are kept, and the
        # conversation/session state used for resume is not touched.
        if self._runtime.wake_queue is not None:
            try:
                count = self._runtime.wake_queue.cancel_older_than(
                    self._runtime.instance_started_at,
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

        if self._runtime.config.harness.timer.enabled:
            self._runtime.timer_service.start()
        self._runtime.instance_manager.start_heartbeat()
        self._load_mesh_ingress()
        self._runtime._outbox._send_restart_notices()

    def send_restart_notices(self) -> None:
        """Notify recently active chats that the service has restarted."""
        self._runtime._outbox._send_restart_notices()

    def create_direct_notifier(self) -> Any:
        """Create a notifier that bypasses the outbox if possible."""
        return self._runtime._outbox._create_direct_notifier()

    def shutdown(self, drain_timeout: float = 120.0) -> None:
        """Drain active turns, notify plugins, and stop background workers."""
        self._runtime._started = False
        self._runtime._restart_draining.set()
        try:
            if not self._runtime._restart._wait_for_active_turns(drain_timeout):
                logger.warning(
                    "Shutdown drain cap (%.0fs) expired with turn(s) still active",
                    drain_timeout,
                )
        except Exception:
            logger.exception("Failed to drain active turns during shutdown")
        if hasattr(self._runtime, "timer_service"):
            self._runtime.timer_service.stop()
        if hasattr(self._runtime, "_typing"):
            self._runtime._typing.stop()
        try:
            self._runtime.event_bus.unsubscribe(self._runtime._on_event)
        except ValueError:
            pass
        if hasattr(self._runtime, "instance_manager"):
            self._runtime.instance_manager.stop_heartbeat()
        if hasattr(self._runtime, "task_engine"):
            self._runtime.task_engine.shutdown(wait=False)
        if hasattr(self._runtime, "event_bus"):
            self._runtime.event_bus.stop()
        now = time.time()
        for chat_id in list(self._runtime._store.keys()):
            record = self._runtime._active_record(chat_id)
            self._runtime._plugins.on_shutdown(
                chat_id,
                ShutdownContext(
                    chat_id=chat_id,
                    record=record,
                    reason="shutdown",
                    now=now,
                    instance_id=self._runtime.instance_id,
                    instance_started_at=self._runtime.instance_started_at,
                ),
            )
        self._runtime._plugins.stop_all()

        with self._runtime._lock:
            managers = list(self._runtime._memory_managers.values())
            self._runtime._memory_managers.clear()
        for manager in managers:
            manager.close()
