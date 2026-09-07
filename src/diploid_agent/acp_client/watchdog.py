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
        with client._lock:
            if not self._running:
                return
            if client._inflight_future is None or client._inflight_future.done():
                return

            now = time.monotonic()
            deadline = client._inflight_deadline
            last_request = client._last_request_at
            last_stdout = getattr(client, "_last_stdout_at", 0.0)
            call_deadline = client._last_control_call_deadline
            has_prompt = bool(client._active_prompts)
            has_pending = bool(client._pending)
            proc = client._proc
            proc_dead = proc is not None and proc.returncode is not None
            silence_after = getattr(client, "_silence_warn_after", 600.0)

        # All _stall_recovery calls must happen outside ``client._lock``:
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
            self._stall_recovery()
            return

        if now > deadline:
            logger.warning("ACP call exceeded its deadline; watchdog recovering")
            self._stall_recovery()
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
                self._stall_recovery()
                return
            if not call_deadline and now - last_request > client._watchdog_timeout:
                logger.warning(
                    "ACP control call produced no response for %ss; watchdog recovering",
                    client._watchdog_timeout,
                )
                self._stall_recovery()
                return

        # An in-flight prompt with a live child and zero stdout traffic is the
        # one wedge mode the reader/drain fixes cannot see: the server itself
        # may be stalled mid-turn.  Prompts are not killed on time (long tool
        # runs are legitimately quiet), so surface the silence as telemetry --
        # a log line, a lifecycle event, and a metric -- at most once per
        # ``silence_after`` interval while it persists.
        if has_prompt and silence_after > 0 and last_stdout > 0:
            silence = now - last_stdout
            if (
                silence >= silence_after
                and now - self._last_silence_warn >= silence_after
            ):
                self._last_silence_warn = now
                session_id = next(iter(client._active_prompts), None)
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

    def _stall_recovery(self) -> None:
        """Kill the ACP subprocess and unblock the in-flight caller (watchdog path)."""
        # Serialize against _ensure_started/close/restart_transport so the
        # watchdog never kills a child mid-startup or stops a loop another
        # thread has just swapped in.  Fall back to ``_lock`` for test fakes
        # that do not model the lifecycle lock.
        lifecycle_lock = getattr(self._client, "_lifecycle_lock", None)
        if lifecycle_lock is None:
            lifecycle_lock = self._client._lock
        with lifecycle_lock:
            self._stall_recovery_inner()

    def _stall_recovery_inner(self) -> None:
        client = self._client
        with client._lock:
            if client.metrics is not None:
                client.metrics.inc("acp_watchdog_fired_total")
            lifecycle_log = getattr(client, "_lifecycle_log", None)
            if lifecycle_log is not None:
                lifecycle_log.write(
                    "transport.restart",
                    reason="watchdog_stall",
                    detail={"killed": True},
                )
            record_restart = getattr(client, "_record_restart_attempt", None)
            if record_restart is not None:
                record_restart()

        client._unblock_inflight("ACP transport watchdog detected a stall")

        with client._lock:
            # Kill the process and stop the loop.
            client._transport_healthy = False
            client._initialized = False
            if client._proc is not None and client._proc.returncode is None:
                try:
                    logger.warning("Killing unresponsive ACP process %s", client._proc.pid)
                    client._kill_process_group(client._proc)
                    if client.metrics is not None:
                        client.metrics.inc("acp_transport_killed_total")
                except Exception:
                    logger.exception("Failed to kill ACP process during watchdog recovery")
            if client._loop is not None and client._loop.is_running():
                try:
                    client._loop.call_soon_threadsafe(client._loop.stop)
                except Exception:
                    logger.exception("Failed to stop ACP event loop during watchdog recovery")
