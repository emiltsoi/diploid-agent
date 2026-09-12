"""Watchdog for detecting stuck ACP transport and prompting recovery."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class PromptWatchdog:
    """Monitor ACP transport I/O and trigger recovery on stalls."""

    def __init__(self, client: Any) -> None:
        self._client = client
        # Shared mutable state; a test fake without ``_state`` is used as the
        # state namespace directly (its ``_x`` attrs stand in for the fields).
        self._state = getattr(client, "_state", None) or client
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_silence_warn = 0.0

    def start(self) -> None:
        """Start the watchdog thread if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._watchdog,
            daemon=True,
            name="acp-watchdog",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the watchdog thread to stop and wait briefly."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _watchdog(self) -> None:
        """Watchdog loop."""
        while True:
            if not self._running:
                break
            time.sleep(self._client._watchdog_interval)
            try:
                self.check()
            except Exception:
                logger.exception("ACP watchdog check failed")

    def check(self) -> None:
        """Detect unresponsive ACP transport and trigger recovery."""
        client = self._client
        with self._state._lock:
            if not self._running:
                return
            if self._state._inflight_future is None or self._state._inflight_future.done():
                return

            now = time.monotonic()
            deadline = self._state._inflight_deadline
            last_request = self._state._last_request_at
            last_stdout = getattr(client, "_last_stdout_at", 0.0)
            call_deadline = self._state._last_control_call_deadline
            has_prompt = bool(self._state._active_prompts)
            has_pending = bool(self._state._pending)
            proc = self._state._proc
            proc_dead = proc is not None and proc.returncode is not None
            silence_after = getattr(client, "_silence_warn_after", 600.0)
            # Snapshot the transport identity being judged stalled.  Recovery
            # waits on ``_lifecycle_lock``, and a concurrent _ensure_started can
            # swap in a brand-new child while we wait -- without this token the
            # kill would land on the replacement (observed in production: a
            # watchdog_stall kill 1 ms after ``transport.start``).
            transport = getattr(client, "_transport", None)
            gen = getattr(transport if transport is not None else client, "generation", None)

        # All _stall_recovery calls must happen outside ``self._state._lock``:
        # recovery acquires ``_lifecycle_lock`` and the required lock order is
        # ``_lifecycle_lock`` -> ``_lock``.
        if proc_dead:
            # If the subprocess has already exited, the transport is dead and
            # recovery should start immediately.
            logger.warning(
                "ACP process %s exited with code %s; watchdog recovering",
                proc.pid,
                proc.returncode,
            )
            self._stall_recovery("proc_dead", proc, gen)
            return

        if now > deadline:
            logger.warning("ACP call exceeded its deadline; watchdog recovering")
            self._stall_recovery("inflight_deadline", proc, gen)
            return

        # The `_pending` map also holds the future for an in-flight prompt, so
        # only use the request timing for non-prompt control calls. Prompts are
        # not killed by the watchdog for time; they are governed by their own
        # `soft_timeout` (client-side cancel that returns a partial reply) and by
        # the overall `_run` deadline. The transport-death check above already
        # handles an exited subprocess.
        if has_pending and not has_prompt:
            if call_deadline and now > call_deadline:
                logger.warning(
                    "ACP control call produced no response for %.1fs; watchdog recovering",
                    now - last_request,
                )
                self._stall_recovery("control_deadline", proc, gen)
                return
            if not call_deadline and now - last_request > client._watchdog_timeout:
                logger.warning(
                    "ACP control call produced no response for %ss; watchdog recovering",
                    client._watchdog_timeout,
                )
                self._stall_recovery("control_deadline", proc, gen)
                return

        # An in-flight prompt with a live child and zero stdout traffic is the
        # one wedge mode the reader/drain fixes cannot see: the server itself
        # may be stalled mid-turn.  Prompts are not killed on time (long tool
        # runs are legitimately quiet), so surface the silence as telemetry --
        # a log line, a lifecycle event, and a metric -- at most once per
        # ``silence_after`` interval while it persists.
        if has_prompt and silence_after > 0 and last_stdout > 0:
            silence = now - last_stdout
            if silence >= silence_after and now - self._last_silence_warn >= silence_after:
                self._last_silence_warn = now
                session_id = next(iter(self._state._active_prompts), None)
                logger.warning(
                    "ACP prompt for session %s has produced no stdout for %.0fs "
                    "(child alive; not auto-killing)",
                    session_id,
                    silence,
                )
                lifecycle_log = getattr(client, "_lifecycle_log", None)
                if lifecycle_log is not None:
                    lifecycle_log.write(
                        "prompt.silence",
                        session_id=session_id,
                        detail={"silence_s": round(silence, 1)},
                    )
                if client.metrics is not None:
                    client.metrics.inc("acp_prompt_silence_total")

    def _stall_recovery(
        self,
        trigger: str = "unknown",
        observed_proc: Any = None,
        observed_gen: Any = None,
    ) -> None:
        """Kill the ACP subprocess and unblock the in-flight caller (watchdog path)."""
        # Serialize against _ensure_started/close/restart_transport so the
        # watchdog never kills a child mid-startup or stops a loop another
        # thread has just swapped in.  Fall back to ``_lock`` for test fakes
        # that do not model the lifecycle lock.
        lifecycle_lock = getattr(self._client, "_lifecycle_lock", None)
        if lifecycle_lock is None:
            lifecycle_lock = self._state._lock
        with lifecycle_lock:
            self._stall_recovery_inner(trigger, observed_proc, observed_gen)

    def _still_stalled(self, client: Any, trigger: str, observed_proc: Any) -> bool:
        """Re-verify the stall condition under ``self._state._lock``.

        The in-flight call may have completed while recovery waited on
        ``_lifecycle_lock``; restarting a healthy transport then is
        gratuitous.  Returns True when in doubt so direct callers keep the
        historical always-recover behaviour.
        """
        now = time.monotonic()
        if trigger == "proc_dead":
            return observed_proc is not None and observed_proc.returncode is not None
        if trigger == "inflight_deadline":
            inflight = self._state._inflight_future
            return inflight is not None and not inflight.done() and now > self._state._inflight_deadline
        if trigger == "control_deadline":
            if not self._state._pending or self._state._active_prompts:
                return False
            call_deadline = self._state._last_control_call_deadline
            if call_deadline:
                return now > call_deadline
            return now - self._state._last_request_at > client._watchdog_timeout
        return True

    def _stall_recovery_inner(
        self,
        trigger: str = "unknown",
        observed_proc: Any = None,
        observed_gen: Any = None,
    ) -> None:
        client = self._client
        with self._state._lock:
            transport = getattr(client, "_transport", None)
            current_gen = getattr(
                transport if transport is not None else client, "generation", None
            )
            current_proc = self._state._proc
            if (observed_gen is not None or observed_proc is not None) and (
                current_gen != observed_gen or current_proc is not observed_proc
            ):
                # The transport judged stalled was already replaced while we
                # waited on ``_lifecycle_lock``.  Recovering now would kill a
                # healthy new child and mark the fresh transport unhealthy.
                logger.warning(
                    "Watchdog stall observation is stale (gen %s pid %s -> "
                    "gen %s pid %s); skipping recovery",
                    observed_gen,
                    getattr(observed_proc, "pid", None),
                    current_gen,
                    getattr(current_proc, "pid", None),
                )
                lifecycle_log = getattr(client, "_lifecycle_log", None)
                if lifecycle_log is not None:
                    lifecycle_log.write(
                        "watchdog.recovery.skipped",
                        reason="stale_generation",
                        detail={
                            "trigger": trigger,
                            "observed_gen": observed_gen,
                            "observed_pid": getattr(observed_proc, "pid", None),
                            "current_gen": current_gen,
                            "current_pid": getattr(current_proc, "pid", None),
                        },
                    )
                return
            if not self._still_stalled(client, trigger, observed_proc):
                logger.info(
                    "Watchdog stall cleared before recovery (trigger=%s); skipping",
                    trigger,
                )
                lifecycle_log = getattr(client, "_lifecycle_log", None)
                if lifecycle_log is not None:
                    lifecycle_log.write(
                        "watchdog.recovery.skipped",
                        reason="cleared",
                        detail={"trigger": trigger},
                    )
                return
            if client.metrics is not None:
                client.metrics.inc("acp_watchdog_fired_total")
            lifecycle_log = getattr(client, "_lifecycle_log", None)
            if lifecycle_log is not None:
                lifecycle_log.write(
                    "transport.restart",
                    reason="watchdog_stall",
                    detail={
                        "killed": True,
                        "trigger": trigger,
                        "stalled_gen": observed_gen,
                        "stalled_pid": getattr(observed_proc, "pid", None),
                    },
                )
            record_restart = getattr(client, "_record_restart_attempt", None)
            if record_restart is not None:
                record_restart()

        client._unblock_inflight("ACP transport watchdog detected a stall")

        with self._state._lock:
            # Kill the process and stop the loop.
            self._state._transport_healthy = False
            self._state._initialized = False
            if self._state._proc is not None and self._state._proc.returncode is None:
                try:
                    logger.warning("Killing unresponsive ACP process %s", self._state._proc.pid)
                    client._kill_process_group(self._state._proc)
                    if client.metrics is not None:
                        client.metrics.inc("acp_transport_killed_total")
                except Exception:
                    logger.exception("Failed to kill ACP process during watchdog recovery")
            if self._state._loop is not None and self._state._loop.is_running():
                try:
                    self._state._loop.call_soon_threadsafe(self._state._loop.stop)
                except Exception:
                    logger.exception("Failed to stop ACP event loop during watchdog recovery")
