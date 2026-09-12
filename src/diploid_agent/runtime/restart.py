"""ACP-subprocess-initiated restart scheduling and drain coordination."""

from __future__ import annotations

import logging
import subprocess
import threading
import time

from diploid_agent.plugins.contexts import ShutdownContext
from diploid_agent.runtime.component import RuntimeComponent

logger = logging.getLogger(__name__)


class RuntimeRestart(RuntimeComponent):
    """Schedule and supervise graceful service restarts requested by the ACP child.

    The runtime handles two related but distinct lifecycle events: a *shutdown*
    (agent process is terminating) and a *restart* (systemd should restart the
    service unit).  This class owns the restart path so ``AgentRuntime`` does not
    have to carry the scheduling, watchdog, and drain logic directly.
    """

    def _on_service_restart(self, service: str, reason: str) -> None:
        """Handle a service restart request from the ACP subprocess.

        Instead of letting the subprocess kill the harness directly, we schedule a
        short-delayed ``systemd-run`` that restarts the service after the current
        turn has a chance to finish and the final reply is delivered.
        """
        with self._runtime._lock:
            now = time.time()
            if (
                now - self._runtime._last_service_restart_at
                < self._runtime._service_restart_cooldown_seconds
            ):
                logger.warning(
                    "Ignoring repeat restart request for %s (cooldown active)",
                    service,
                )
                return
            self._runtime._last_service_restart_at = now

        logger.warning(
            "ACP subprocess requested restart of %s (reason: %s); scheduling graceful restart",
            service,
            reason,
        )

        # Cancel any pending auto-continue wakes so the restart does not loop.
        if self._runtime.wake_queue is not None:
            try:
                self._runtime.wake_queue.cancel(reason="auto_continue")
            except Exception as exc:
                logger.warning(
                    "Failed to cancel auto-continue wakes before restart",
                    exc_info=exc,
                )

        self._runtime.suppress_auto_continue()

        # Record the incident for observability.
        if self._runtime._incidents is not None:
            try:
                self._runtime._incidents.record(
                    plugin="self_management",
                    phase="graceful_restart",
                    error=f"ACP subprocess requested restart of {service}: {reason}",
                    action="scheduled",
                )
            except Exception as exc:
                logger.warning(
                    "Failed to record restart incident",
                    exc_info=exc,
                )

        # Drain in-flight turns, flush plugin state, then schedule the restart.
        self._schedule_draining_restart(service, chat_id=None, reason=reason)

    def _schedule_draining_restart(
        self,
        service: str,
        chat_id: str | None,
        reason: str,
        drain_cap: float = 120.0,
    ) -> bool:
        """Begin the restart drain: block new turns, wait for in-flight turns,
        flush plugin state, then schedule the real systemd restart.

        Returns False (without setting the drain flag) when the named unit
        definitively does not exist. Otherwise returns immediately; the drain
        runs on a background thread so callers (the control-socket listener,
        HTTP/Telegram actions) never block.
        """
        if not self._runtime._unit_exists(service):
            logger.error(
                "Refusing graceful restart: systemd user unit %s is not installed",
                service,
            )
            return False
        self._runtime._restart_draining.set()

        def _drain_then_restart() -> None:
            try:
                if not self._wait_for_active_turns(drain_cap):
                    logger.warning(
                        "Restart drain cap (%.0fs) expired with turn(s) still active; "
                        "restarting anyway",
                        drain_cap,
                    )
                self._flush_plugins_for_restart()
            finally:
                # Short residual delay so the final reply/outbox can deliver.
                self._schedule_systemd_restart(service, delay=5.0, chat_id=chat_id, reason=reason)
                self._arm_restart_watchdog(service, due_in=5.0, chat_id=chat_id)

        threading.Thread(target=_drain_then_restart, daemon=True, name="restart-drain").start()
        return True

    def _arm_restart_watchdog(
        self,
        service: str,
        due_in: float,
        chat_id: str | None,
        margin: float = 60.0,
    ) -> None:
        """Self-heal when a scheduled restart never fires (missing unit,
        systemd-run failure): clear the drain flag so the service does not
        wedge refusing new turns forever.
        """

        def _reaper() -> None:
            time.sleep(due_in + margin)
            if not self._runtime._started or not self._runtime._restart_draining.is_set():
                # A real restart/shutdown is already underway.
                return
            self._runtime._restart_draining.clear()
            self._runtime._last_service_restart_at = 0.0
            logger.error(
                "Scheduled restart of %s never fired; cleared drain state so turns resume",
                service,
            )
            if self._runtime._incidents is not None:
                try:
                    self._runtime._incidents.record(
                        plugin="self_management",
                        phase="graceful_restart",
                        error=f"Scheduled restart of {service} did not fire",
                        action="drain_cleared",
                        chat_id=chat_id,
                    )
                except Exception:
                    logger.exception("Failed to record failed-restart incident")

        threading.Thread(target=_reaper, daemon=True, name="restart-watchdog").start()

    def _wait_for_active_turns(self, timeout: float) -> bool:
        """Block until every ActiveTurn finishes or ``timeout`` expires.

        Never holds ``self._runtime._lock`` while waiting: the turn's ``finally`` needs
        the same RLock to pop ``_active_turns`` and notify ``_condition``, so
        waiting under the lock would deadlock the drain.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._runtime._lock:
                turns = list(self._runtime._active_turns.values())
            if not turns:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            with turns[0]._condition:
                turns[0]._condition.wait(timeout=min(remaining, 0.5))

    def _flush_plugins_for_restart(self) -> None:
        """Run shutdown/sleeping hooks on every chat so plugin state persists."""
        now = time.time()
        for chat_id in list(self._runtime._store.keys()):
            try:
                with self._runtime._lock:
                    record = self._runtime._active_record(chat_id)
                self._runtime._plugins.on_shutdown(
                    chat_id,
                    ShutdownContext(
                        chat_id=chat_id,
                        record=record,
                        reason="restart",
                        now=now,
                        instance_id=self._runtime.instance_id,
                        instance_started_at=self._runtime.instance_started_at,
                    ),
                )
            except Exception:
                logger.exception("Failed to flush plugins for %s before restart", chat_id)

    def _schedule_systemd_restart(
        self,
        service: str,
        delay: float,
        chat_id: str | None,
        reason: str,
    ) -> None:
        """Run a short-delayed systemd-run that restarts the named service."""

        def _do_restart() -> None:
            try:
                subprocess.Popen(
                    [
                        "systemd-run",
                        "--user",
                        f"--on-active={delay}s",
                        "--timer-property=AccuracySec=1s",
                        "/usr/bin/systemctl",
                        "--user",
                        "restart",
                        service,
                    ],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if chat_id is not None:
                    logger.info(
                        "Scheduled graceful restart of %s in %ss (from chat %s)",
                        service,
                        delay,
                        chat_id,
                    )
                else:
                    logger.info("Scheduled graceful restart of %s in %ss", service, delay)
            except Exception:
                logger.exception("Failed to schedule graceful restart of %s", service)

        # Run the scheduler in its own thread so the ACP control listener is not
        # blocked.
        threading.Thread(target=_do_restart, daemon=True).start()
