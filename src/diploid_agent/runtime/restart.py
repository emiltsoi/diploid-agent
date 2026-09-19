"""ACP-subprocess-initiated restart scheduling and drain coordination."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from diploid_agent.config import AuthorshipConfig
from diploid_agent.plugins.contexts import ShutdownContext
from diploid_agent.runtime.state import RuntimeState

if TYPE_CHECKING:
    from diploid_agent.config import Config
    from diploid_agent.memory import MemoryManager
    from diploid_agent.models import ActiveTurn, ChatState
    from diploid_agent.plugin_incidents import PluginIncidentStore
    from diploid_agent.plugins import PluginManager
    from diploid_agent.runtime.store import ChatSessionStore
    from diploid_agent.runtime.wake_queue import WakeQueue


logger = logging.getLogger(__name__)


class RuntimeRestart:
    """Schedule and supervise graceful service restarts requested by the ACP child.

    The runtime handles two related but distinct lifecycle events: a *shutdown*
    (agent process is terminating) and a *restart* (systemd should restart the
    service unit).  This class owns the restart path so ``AgentRuntime`` does not
    have to carry the scheduling, watchdog, and drain logic directly.
    """

    def __init__(
        self,
        *,
        config: Config,
        state: RuntimeState,
        lock: threading.RLock,
        wake_queue: WakeQueue | None,
        incidents: PluginIncidentStore | None,
        plugins: PluginManager,
        chat_store: ChatSessionStore,
        active_turns: dict[str, ActiveTurn],
        session_ops: set[str],
        store: dict[str, ChatState],
        instance_id: str,
        instance_started_at: float,
        suppress_auto_continue_fn: Callable[..., None],
        unit_exists_fn: Callable[[str], bool],
        memory_manager: Callable[[str], MemoryManager],
        notify_fn: Callable[[str, str], None] | None = None,
    ) -> None:
        self._config = config
        self._notify_fn = notify_fn
        self._state = state
        self._lock = lock
        self._wake_queue = wake_queue
        self._incidents = incidents
        self._plugins = plugins
        self._chat_store = chat_store
        self._active_turns = active_turns
        self._session_ops = session_ops
        self._store = store
        self._instance_id = instance_id
        self._instance_started_at = instance_started_at
        self._suppress_auto_continue = suppress_auto_continue_fn
        self._unit_exists = unit_exists_fn
        self._memory_manager = memory_manager
        self._last_restart_memory_written: dict[str, float] = {}

    def _systemd_unit_exists(self, service: str) -> bool:
        """Best-effort check that a user unit exists before draining for it.

        Returns True when the check cannot be made (no systemctl, no user bus)
        so non-systemd environments are not blocked; only a definitive
        "no such unit" answer refuses the restart.
        """
        try:
            proc = subprocess.run(
                ["systemctl", "--user", "cat", service],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return True
        if proc.returncode == 0:
            return True
        output = f"{proc.stdout}\n{proc.stderr}"
        return "No files found" not in output and "not found" not in output.lower()

    def _record_restart_memory(self, chat_id: str, reason: str | None = None) -> None:
        """Record a brief ACP restart observation for memory_recall.

        Deduplicates rapid restarts within a 60-second window per chat so a
        tight restart loop only produces one memory item.
        """
        now = time.time()
        with self._lock:
            last = self._last_restart_memory_written.get(chat_id, 0)
            if now - last < 60:
                return
            self._last_restart_memory_written[chat_id] = now

        ts = datetime.fromtimestamp(now, tz=UTC).isoformat()
        text = f"ACP transport restarted at {ts}."
        if reason:
            text += f" Reason: {reason}."
        try:
            self._memory_manager(chat_id).retain(
                text,
                tags=["system", "acp", "restart"],
            )
        except Exception as exc:
            logger.warning(
                "Failed to record restart memory for %s",
                chat_id,
                exc_info=exc,
            )

    def _agent_restart_verdict(self, service: str, reason: str) -> str | None:
        """Policy check for agent-initiated restarts. ``None`` means allowed.

        Every agent door (the ACP control socket, the ``harness_restart`` MCP
        tool, anything added later) converges on ``_on_service_restart``, so the
        gate lives here — not in the sandbox shim or the tool schema. Operator
        doors (``graceful_service_restart`` via HTTP/Telegram) never pass
        through this function.
        """
        auth: AuthorshipConfig | None = None
        for plugin in self._config.harness.plugins:
            if plugin.name != "authorship":
                continue
            if plugin.enabled:
                try:
                    auth = AuthorshipConfig.model_validate(plugin.config or {})
                except Exception:
                    logger.exception("Invalid authorship plugin config")
                    auth = None
            break
        if auth is None or not auth.restart_enabled:
            return "restart is not enabled for this persona (authorship restart_enabled)"
        if not reason or not reason.strip():
            return "a non-empty reason is required"
        allowed = self._config.harness.restart_allowed_units
        if not allowed:
            own = self._config.persona.name if self._config.persona else ""
            allowed = [f"{own}.service"] if own else []
        if service not in allowed:
            return f"service {service!r} is not in restart_allowed_units"
        return None

    def _notify_agent_restart(self, service: str, reason: str) -> None:
        """Post an operator notice that an agent-initiated restart is scheduled."""
        if self._notify_fn is None:
            return
        chat_id = self._config.harness.mesh.fallback_chat_id
        if not chat_id:
            return
        persona = self._config.persona.name if self._config.persona else "agent"
        text = f"[harness] {persona} scheduled a graceful restart of {service}"
        if reason.strip():
            text += f" — {reason.strip()}"
        try:
            self._notify_fn(chat_id, text)
        except Exception as exc:
            logger.warning("Failed to enqueue agent-restart notice", exc_info=exc)

    def _on_service_restart(self, service: str, reason: str) -> str:
        """Handle a service restart request from the ACP subprocess.

        Instead of letting the subprocess kill the harness directly, we schedule a
        short-delayed ``systemd-run`` that restarts the service after the current
        turn has a chance to finish and the final reply is delivered.

        Returns an ack status string for the control-socket reply:
        ``scheduled``, ``cooldown``, or ``rejected: <why>``.
        """
        verdict = self._agent_restart_verdict(service, reason)
        if verdict is not None:
            logger.warning("Agent restart request refused: %s", verdict)
            if self._incidents is not None:
                try:
                    self._incidents.record(
                        plugin="self_management",
                        phase="agent_restart_gate",
                        error=f"{service}: {verdict}",
                        action="rejected",
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to record restart-gate incident",
                        exc_info=exc,
                    )
            return f"rejected: {verdict}"
        with self._lock:
            now = time.time()
            if (
                now - self._state.last_service_restart_at
                < self._state.service_restart_cooldown_seconds
            ):
                logger.warning(
                    "Ignoring repeat restart request for %s (cooldown active)",
                    service,
                )
                return "cooldown"

        logger.warning(
            "ACP subprocess requested restart of %s (reason: %s); scheduling graceful restart",
            service,
            reason,
        )

        # Cancel any pending auto-continue wakes so the restart does not loop.
        if self._wake_queue is not None:
            try:
                self._wake_queue.cancel(reason="auto_continue")
            except Exception as exc:
                logger.warning(
                    "Failed to cancel auto-continue wakes before restart",
                    exc_info=exc,
                )

        self._suppress_auto_continue()

        # Record the incident for observability.
        if self._incidents is not None:
            try:
                self._incidents.record(
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
        if not self.schedule_draining_restart(service, chat_id=None, reason=reason):
            return "rejected: no such unit"
        # Stamp the cooldown only after the unit is known to exist, so a
        # rejected request does not burn the window for a legitimate one.
        with self._lock:
            self._state.last_service_restart_at = now
        self._notify_agent_restart(service, reason)
        return "scheduled"

    def schedule_draining_restart(
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
        if not self._unit_exists(service):
            logger.error(
                "Refusing graceful restart: systemd user unit %s is not installed",
                service,
            )
            return False
        self._state.restart_draining.set()
        with self._lock:
            self._state.pending_restart = {
                "service": service,
                "reason": reason,
                "chat_id": chat_id,
                "draining_since": time.time(),
            }

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

    def pending_restart(self) -> dict[str, Any] | None:
        """Return legible detail of an in-progress graceful restart, if any.

        A drain (restart or shutdown) without recorded detail reports
        ``{"draining": True}`` — still observable, just without a named unit.
        """
        if not self._state.restart_draining.is_set():
            return None
        with self._lock:
            info = dict(self._state.pending_restart or {})
            info["active_turns"] = len(self._active_turns)
            info["session_ops_pending"] = bool(self._session_ops)
        if not info.get("service"):
            info["draining"] = True
        return info

    def _notify_restart_failed(
        self,
        service: str,
        chat_id: str | None,
        detail: str,
    ) -> None:
        """Post an operator notice that a scheduled restart will not fire."""
        if self._notify_fn is None:
            return
        target = chat_id or self._config.harness.mesh.fallback_chat_id
        if not target:
            return
        persona = self._config.persona.name if self._config.persona else "agent"
        text = f"[harness] {persona} restart of {service} did not fire — {detail}"
        try:
            self._notify_fn(target, text)
        except Exception as exc:
            logger.warning("Failed to enqueue restart-failure notice", exc_info=exc)

    def _restart_failed(
        self,
        service: str,
        chat_id: str | None,
        cause: str,
    ) -> None:
        """A scheduled restart will not fire: reopen the gate, record, notify.

        Called when ``systemd-run`` fails synchronously and by the watchdog
        when the timer never fires — clears the drain so turns resume instead
        of the service refusing work while still alive, and resets the
        cooldown so a retry is not blocked.
        """
        self._state.restart_draining.clear()
        self._state.pending_restart = None
        self._state.last_service_restart_at = 0.0
        if self._incidents is not None:
            try:
                self._incidents.record(
                    plugin="self_management",
                    phase="graceful_restart",
                    error=f"Scheduled restart of {service} did not fire: {cause}",
                    action="drain_cleared",
                    chat_id=chat_id,
                )
            except Exception:
                logger.exception("Failed to record failed-restart incident")
        self._notify_restart_failed(service, chat_id, cause)

    def _arm_restart_watchdog(
        self,
        service: str,
        due_in: float,
        chat_id: str | None,
        margin: float = 60.0,
    ) -> None:
        """Self-heal when a scheduled restart never fires (missing unit,
        timer dropped): clear the drain flag so the service does not wedge
        refusing new turns forever, and tell the requester it is still alive.
        """

        def _reaper() -> None:
            time.sleep(due_in + margin)
            if not self._state.started or not self._state.restart_draining.is_set():
                # A real restart/shutdown is already underway, or the schedule
                # itself failed synchronously and was already reported.
                return
            logger.error(
                "Scheduled restart of %s never fired; cleared drain state so turns resume",
                service,
            )
            self._restart_failed(service, chat_id, "scheduled restart never fired")

        threading.Thread(target=_reaper, daemon=True, name="restart-watchdog").start()

    def _wait_for_active_turns(self, timeout: float) -> bool:
        """Block until every ActiveTurn finishes or ``timeout`` expires.

        Never holds ``self._lock`` while waiting: the turn's ``finally`` needs
        the same RLock to pop ``_active_turns`` and notify ``_condition``, so
        waiting under the lock would deadlock the drain.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                turns = list(self._active_turns.values())
                ops_pending = bool(self._session_ops)
            if not turns and not ops_pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if turns:
                with turns[0]._condition:
                    turns[0]._condition.wait(timeout=min(remaining, 0.5))
            else:
                # Session ops clear themselves via `finally` — no condition to
                # wait on; they are sub-second, so poll briefly.
                time.sleep(min(remaining, 0.1))

    def _flush_plugins_for_restart(self) -> None:
        """Run shutdown/sleeping hooks on every chat so plugin state persists."""
        now = time.time()
        for chat_id in list(self._store.keys()):
            try:
                with self._lock:
                    record = self._chat_store.active_record(chat_id)
                self._plugins.on_shutdown(
                    chat_id,
                    ShutdownContext(
                        chat_id=chat_id,
                        record=record,
                        reason="restart",
                        now=now,
                        instance_id=self._instance_id,
                        instance_started_at=self._instance_started_at,
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
                proc = subprocess.run(
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
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            except Exception as exc:
                logger.exception("Failed to schedule graceful restart of %s", service)
                self._restart_failed(service, chat_id, f"systemd-run error: {exc}")
                return
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
                logger.error(
                    "systemd-run scheduling restart of %s exited %s: %s",
                    service,
                    proc.returncode,
                    detail,
                )
                self._restart_failed(service, chat_id, detail)
                return
            if chat_id is not None:
                logger.info(
                    "Scheduled graceful restart of %s in %ss (from chat %s)",
                    service,
                    delay,
                    chat_id,
                )
            else:
                logger.info("Scheduled graceful restart of %s in %ss", service, delay)

        # Run the scheduler in its own thread so the ACP control listener is not
        # blocked.
        threading.Thread(target=_do_restart, daemon=True).start()
