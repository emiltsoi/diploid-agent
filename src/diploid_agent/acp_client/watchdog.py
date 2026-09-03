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

            # If the subprocess has already exited, the transport is dead and
            # recovery should start immediately.
            if client._proc is not None and client._proc.returncode is not None:
                logger.warning(
                    "ACP process %s exited with code %s; watchdog recovering",
                    client._proc.pid,
                    client._proc.returncode,
                )
                self._stall_recovery()
                return

            now = time.monotonic()
            deadline = client._inflight_deadline
            last_request = client._last_request_at
            call_deadline = client._last_control_call_deadline
            has_prompt = bool(client._active_prompts)
            has_pending = bool(client._pending)

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

    def _stall_recovery(self) -> None:
        """Kill the ACP subprocess and unblock the in-flight caller (watchdog path)."""
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
