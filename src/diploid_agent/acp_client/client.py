"""ACP client: drive an ACP-compatible agent binary over stdio JSON-RPC.

Provides a long-lived ACP session, giving real-time `agent_message_chunk`
streaming, mid-turn cancellation via `session/cancel` notification, and
steering via `session/set_config_option` (mode/model).
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import os
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.acp_client.control import ControlListener
from diploid_agent.acp_client.errors import (
    AcpError,
    AcpMcpError,
    AcpModelError,
    AcpSessionStaleError,
    AcpTransportError,
    _acp_error_from_response,
)
from diploid_agent.acp_client.lifecycle import AcpLifecycleLog, AcpRestartHistory
from diploid_agent.acp_client.sandbox import AcpSandbox
from diploid_agent.acp_client.transport import AcpTransport
from diploid_agent.acp_client.types import AcpPromptResult, _Prompt
from diploid_agent.acp_client.utils import (
    _devin_default_start_args,
    _load_windsurf_api_key,
    _normalize_model,
    _resolve_agent_bin,
)
from diploid_agent.acp_client.watchdog import PromptWatchdog

logger = logging.getLogger(__name__)

_ACP_MODE_MAP = {
    "auto": "smart",
    "normal": "accept-edits",
    "accept-edits": "accept-edits",
    "smart": "smart",
    "dangerous": "bypass",
    "bypass": "bypass",
}


class _TransportAttr:
    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: AcpClient | None, owner: type | None = None) -> Any:
        if instance is None:
            raise AttributeError(self.name)
        return getattr(instance._transport, self.name)

    def __set__(self, instance: AcpClient, value: Any) -> None:
        setattr(instance._transport, self.name, value)


class AcpClient:
    """Synchronous wrapper around an ACP v1 agent binary.

    Spawns one long-lived agent subprocess and multiplexes sessions through it.
    Defaults to `devin acp` start arguments for backward compatibility; other
    binaries can override `start_args`.
    """

    def __init__(
        self,
        model: str = "swe-1-7",
        permission_mode: str = "dangerous",
        timeout: float | None = 900.0,
        startup_timeout: float = 30.0,
        control_timeout: float = 120.0,
        watchdog_interval: float = 10.0,
        watchdog_timeout: float = 120.0,
        max_restarts: int = 3,
        max_mcp_restarts: int = 5,
        max_user_restarts: int = 5,
        restart_backoff_window: float = 300.0,
        acp_resume_max_retries: int = 1,
        acp_resume_retry_base_seconds: float = 0.5,
        acp_resume_retry_max_seconds: float = 5.0,
        agent_bin: str | Path = "~/.local/bin/devin",
        start_args: list[str] | None = None,
        api_key: str | None = None,
        metrics: Any | None = None,
        service_name: str | None = None,
        on_service_restart: Callable[[str, str], None] | None = None,
        lifecycle_log: AcpLifecycleLog | None = None,
    ):
        self.model = _normalize_model(model)
        self.acp_mode = _ACP_MODE_MAP.get(permission_mode, "bypass")
        self.timeout = timeout
        self.agent_bin = _resolve_agent_bin(agent_bin)
        self.start_args = start_args or _devin_default_start_args(self.model)
        self.metrics = metrics
        self.acp_resume_max_retries = max(0, acp_resume_max_retries)
        self.acp_resume_retry_base_seconds = acp_resume_retry_base_seconds
        self.acp_resume_retry_max_seconds = acp_resume_retry_max_seconds

        self._api_key = (
            api_key
            or _load_windsurf_api_key()
            or os.environ.get("WINDSURF_API_KEY")
            or os.environ.get("ACP_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "No api_key provided and no Devin credentials or "
                "WINDSURF_API_KEY/ACP_API_KEY in environment."
            )

        # Session/prompt state.
        self._next_id = 0
        self._active_prompts: dict[str, _Prompt] = {}
        self._session_models: dict[str, str] = {}
        self._pending_cancels: set[str] = set()
        self._model_options: list[str] | None = None
        self._mcp_servers: list[dict[str, Any]] = []

        # Shared lock.
        self._lock = threading.RLock()

        # Low-level transport state.
        self._transport = AcpTransport(self)

        # Timing and control configuration.
        self._control_timeout = control_timeout
        self._startup_timeout = startup_timeout
        self._watchdog_interval = watchdog_interval
        self._watchdog_timeout = watchdog_timeout

        # Restart backoff per cause.
        self._max_restarts_by_reason: dict[str, int] = {
            "transport_error": max(0, max_restarts),
            "mcp_change": max(0, max_mcp_restarts),
            "user_restart": max(0, max_user_restarts),
        }
        self._restart_backoff_window = max(1.0, restart_backoff_window)

        # Service restart support: the subprocess can request a controlled restart
        # through a private Unix socket instead of killing the parent directly.
        self._service_name = service_name
        self._on_service_restart = on_service_restart
        self._lifecycle_log = lifecycle_log
        restart_history_path = None
        if lifecycle_log is not None:
            restart_history_path = lifecycle_log.path.parent / "acp_restart_history.jsonl"
        self._restart_history_store = AcpRestartHistory(
            restart_history_path,
            self._restart_backoff_window,
        )
        self._sandbox = AcpSandbox(service_name=service_name)
        self._control = ControlListener(
            service_name=service_name or "unknown.service",
            on_service_restart=on_service_restart,
            control_timeout=control_timeout,
            watchdog_timeout=watchdog_timeout,
        )

        self._watchdog = PromptWatchdog(self)

        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Backward-compatible aliases for transport state.

    _loop = _TransportAttr("_loop")
    _thread = _TransportAttr("_thread")
    _proc = _TransportAttr("_proc")
    _reader_task = _TransportAttr("_reader_task")
    _stderr_task = _TransportAttr("_stderr_task")
    _inflight_future = _TransportAttr("_inflight_future")
    _inflight_deadline = _TransportAttr("_inflight_deadline")
    _last_stdout_at = _TransportAttr("_last_stdout_at")
    _last_progress_at = _TransportAttr("_last_progress_at")
    _last_request_at = _TransportAttr("_last_request_at")
    _last_control_call_deadline = _TransportAttr("_last_control_call_deadline")
    _pending = _TransportAttr("_pending")
    _transport_healthy = _TransportAttr("_transport_healthy")
    _initialized = _TransportAttr("_initialized")
    _restart_history = _TransportAttr("_restart_history")

    # ------------------------------------------------------------------
    # Watchdog and control aliases.

    @property
    def _control_socket_path(self) -> Path:
        """Backward-compatible alias for tests that introspect the control socket."""
        return self._control.socket_path

    @property
    def _watchdog_running(self) -> bool:
        """Backward-compatible alias for tests that drive the watchdog directly."""
        return self._watchdog._running

    @_watchdog_running.setter
    def _watchdog_running(self, value: bool) -> None:
        self._watchdog._running = value

    def _check_watchdog(self) -> None:
        """Backward-compatible alias for tests that drive the watchdog directly."""
        self._watchdog.check()

    # ---------------------------------------------------------------- public

    def create_session(
        self,
        prompt_text: str,
        *,
        cwd: str | Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        soft_timeout: float | None = None,
        chat_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Create a new ACP session, send the first prompt, return the result."""
        self._ensure_started(mcp_servers)
        normalized_mcp_servers = self._sandbox.normalize_mcp_servers(mcp_servers)
        if cwd is not None:
            cwd = Path(cwd)
        return self._run(
            self._create_session(
                prompt_text,
                cwd=cwd,
                model=model,
                mcp_servers=normalized_mcp_servers,
                soft_timeout=soft_timeout,
                chat_id=chat_id,
                on_chunk=on_chunk,
                on_update=on_update,
            ),
            timeout=self.timeout + 30.0 if self.timeout is not None else None,
        )

    def send_message(
        self,
        session_id: str,
        prompt_text: str,
        *,
        cwd: str | Path | None = None,
        model: str | None = None,
        soft_timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Send a follow-up prompt to an existing ACP session."""
        self._ensure_started()
        if cwd is not None:
            cwd = Path(cwd)
        return self._run(
            self._send_message(
                session_id,
                prompt_text,
                cwd=cwd,
                model=model,
                soft_timeout=soft_timeout,
                on_chunk=on_chunk,
                on_update=on_update,
            ),
            timeout=self.timeout + 30.0 if self.timeout is not None else None,
        )

    def list_models(self) -> list[str]:
        """Return the model IDs advertised by the ACP server."""
        self._ensure_started()
        if self._model_options is not None:
            return self._model_options
        return self._run(self._list_models(), timeout=60.0)

    def health(self) -> bool:
        """Return True if the ACP transport is initialized and healthy."""
        with self._lock:
            return self._transport.healthy()

    def session_alive(self, session_id: str) -> bool:
        """Probe whether an ACP session id is still valid."""
        self._ensure_started()
        return self._run(self._session_alive(session_id), timeout=30.0)

    def resume_session(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        """Resume a persisted ACP session and return the active session id.

        Tries ``session/resume`` first, then falls back to ``session/load`` if
        the agent does not advertise the unstable ``session/resume`` method.
        After a successful resume the session mode and model are re-applied.
        """
        self._ensure_started(mcp_servers)
        return self._run(
            self._resume_session(
                session_id,
                cwd=cwd,
                model=model,
                mcp_servers=mcp_servers,
            ),
            timeout=self.timeout + 30.0 if self.timeout is not None else None,
        )

    def cancel(self, session_id: str) -> None:
        """Send a `session/cancel` notification for an in-flight prompt.

        `session/cancel` only works as a JSON-RPC notification (no `"id"`).
        When the server cancels the turn, the in-progress prompt resolves with
        `stopReason: "cancelled"` and the collected chunks form the partial
        reply.
        """
        if self._loop is None:
            return

        def _do_cancel() -> None:
            # Record the cancel request so a prompt that has not yet registered
            # can still be cancelled once it appears.
            self._pending_cancels.add(session_id)
            prompt = self._active_prompts.get(session_id)
            if prompt is None:
                return
            if prompt.cancel_done.done() or prompt.cancelled:
                # A soft timeout or another cancel already fired; just mark this
                # as an explicit caller cancellation.
                prompt.cancelled = True
                return
            self._pending_cancels.discard(session_id)
            prompt.cancelled = True
            # Wake up the _prompt waiter so it stops waiting for the full turn.
            if not prompt.cancel_done.done():
                prompt.cancel_done.set_result(None)
            # Fire the notification; this actually asks the server to abort.
            self._loop.create_task(self._send_cancel_notification(session_id))

        self._loop.call_soon_threadsafe(_do_cancel)

    def active_session_id(self) -> str | None:
        """Return the session id of the currently in-flight prompt, if any."""
        if not self._active_prompts:
            return None
        # There should only be one active prompt per transport; return the first.
        return next(iter(self._active_prompts.keys()), None)

    def list_sessions(self, *, cwd: Path | None = None) -> list[dict[str, Any]]:
        """ACP does not expose a directory-scoped session list."""
        return []

    def close(self) -> None:
        """Terminate the ACP subprocess and stop the background loop."""
        with self._lock:
            if not self._initialized or self._loop is None:
                return
            if self._lifecycle_log is not None:
                self._lifecycle_log.write("transport.stop")
            self._initialized = False
            self._transport_healthy = False
            try:
                if (
                    self._thread is not None
                    and self._thread.is_alive()
                    and self._loop is not None
                    and self._loop.is_running()
                ):
                    # Schedule _close_transport on the background loop and stop
                    # the loop only after the coroutine has actually completed.
                    # Stopping the loop from the main thread immediately after
                    # scheduling the close task can leave the coroutine unawaited
                    # and generate a RuntimeWarning.
                    future: concurrent.futures.Future[Any] = asyncio.run_coroutine_threadsafe(
                        self._close_transport(), self._loop
                    )

                    def _stop_loop_soon(_: Any) -> None:
                        if self._loop is not None and self._loop.is_running():
                            try:
                                self._loop.call_soon_threadsafe(self._loop.stop)
                            except RuntimeError:
                                pass

                    future.add_done_callback(_stop_loop_soon)
                    try:
                        future.result(timeout=10.0)
                    except (RuntimeError, TimeoutError) as exc:
                        logger.warning("ACP close transport failed: %s", exc)
                        if self._proc is not None and self._proc.returncode is None:
                            try:
                                self._proc.kill()
                            except Exception:
                                logger.exception("Failed to kill ACP process during close")
                        if self._loop is not None:
                            try:
                                self._loop.call_soon_threadsafe(self._loop.stop)
                            except RuntimeError:
                                pass
                else:
                    # The background loop is not running; kill the process
                    # directly and do not schedule a coroutine that can never
                    # be awaited.
                    if self._proc is not None and self._proc.returncode is None:
                        try:
                            self._proc.kill()
                        except Exception:
                            logger.exception("Failed to kill ACP process during close")
            finally:
                self._watchdog.stop()
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=10.0)
                self._loop = None
                self._thread = None
                self._proc = None
                self._reader_task = None
                self._stderr_task = None
                self._sandbox.cleanup()
                self._control.close()

    # ---------------------------------------------------------------- internal

    def _categorize_restart_reason(self, reason: str | None) -> str:
        """Map a free-text restart reason to a backoff bucket."""
        if reason == "mcp_change":
            return "mcp_change"
        if reason is not None and ("user" in reason.lower() or reason.startswith("/")):
            return "user_restart"
        return "transport_error"

    def _check_restart_backoff(self, reason: str | None = None) -> None:
        """Raise AcpTransportError if we have restarted too many times recently."""
        bucket = self._categorize_restart_reason(reason)
        max_restarts = self._max_restarts_by_reason.get(bucket, 0)
        self._restart_history_store.check(max_restarts, reason=bucket)

    def _record_restart_attempt(self, reason: str | None = None) -> None:
        """Record that we are about to (re)start the ACP transport."""
        bucket = self._categorize_restart_reason(reason)
        self._restart_history_store.record(reason=bucket)

    def _unblock_inflight(self, reason: str) -> None:
        """Set an exception on the in-flight _run future and cancel active prompts."""
        self._transport._unblock_inflight(reason)

    def _kill_process_group(self, proc: asyncio.subprocess.Process) -> None:
        """Kill the subprocess and any spawned descendants."""
        self._transport._kill_process_group(proc)

    def _run(self, coro: Any, timeout: float | None = None) -> Any:
        """Run a coroutine on the background loop and block for the result."""
        return self._transport.run(coro, timeout=timeout)

    async def _start_transport(self) -> None:
        """Start the ACP subprocess and run the initialize handshake."""
        await self._transport._start_transport()

    async def _close_transport(self) -> None:
        """Close the ACP subprocess and cancel I/O tasks."""
        await self._transport.close()

    def _ensure_started(
        self,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        """Start the ACP transport, writing the active MCP list first.

        If the transport is already running with a different MCP server list,
        restart it so `devin acp` picks up the new `mcp_config.json`.
        """
        target = self._sandbox.normalize_mcp_servers(
            mcp_servers if mcp_servers is not None else self._mcp_servers
        )

        while True:
            with self._lock:
                if self._transport._is_transport_healthy():
                    if self._sandbox.mcp_servers_key(target) == self._sandbox.mcp_servers_key(
                        self._mcp_servers
                    ):
                        self._watchdog.start()
                        return
                    # MCP list changed; restart outside the lock.
                    needs_restart = True
                else:
                    needs_restart = False
                    if self._initialized:
                        logger.warning(
                            "ACP transport was marked initialized but is not healthy; resetting"
                        )
                        self._transport._cleanup_stale_transport()

                    self._initialized = False
                    self._transport_healthy = False
                    self._loop = asyncio.new_event_loop()
                    self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
                    self._thread.start()

                    self._sandbox.prepare(target)

            if needs_restart:
                if self._lifecycle_log is not None:
                    self._lifecycle_log.write(
                        "transport.restart",
                        reason="mcp_change",
                        detail={"mcp_servers": [s.get("name") for s in (target or [])]},
                    )
                self._check_restart_backoff("mcp_change")
                self._record_restart_attempt("mcp_change")
                self.close()
                # close() sets _initialized=False and clears the transport. Loop
                # back to start a fresh one with the new target list.
                continue

            # Do not hold _lock while waiting for the transport to start; the
            # background _send coroutine needs to acquire it to record request time,
            # and holding it here would block the event loop.
            last_exc: Exception | None = None
            attempts = 2
            for attempt in range(1, attempts + 1):
                try:
                    self._run(self._start_transport(), timeout=self._startup_timeout)
                    break
                except AcpTransportError as exc:
                    last_exc = exc
                    logger.warning(
                        "ACP transport startup timed out (attempt %d/%d)", attempt, attempts
                    )
                    self.close()
                    if attempt < attempts:
                        with self._lock:
                            self._loop = asyncio.new_event_loop()
                            self._thread = threading.Thread(
                                target=self._loop.run_forever, daemon=True
                            )
                            self._thread.start()
                            self._sandbox.prepare(target)
                    else:
                        raise last_exc

            with self._lock:
                self._initialized = True
                self._mcp_servers = target
                self._transport_healthy = True
            if self._lifecycle_log is not None:
                self._lifecycle_log.write("transport.start")
            return

    def restart_transport(self, reason: str | None = None, chat_id: str | None = None) -> None:
        """Kill the ACP subprocess and start a fresh one."""
        logger.warning("Restarting ACP transport")
        if self._lifecycle_log is not None:
            self._lifecycle_log.write(
                "transport.restart",
                chat_id=chat_id,
                reason=reason,
                detail={"reason": reason} if reason else None,
            )
        self._check_restart_backoff(reason)
        self._record_restart_attempt(reason)
        if self.metrics is not None:
            self.metrics.inc("acp_restarts_total")
        self._unblock_inflight("ACP transport restarted")
        self.close()
        self._ensure_started()

    # ---------------------------------------------------------------- JSON-RPC

    async def _call(
        self,
        method: str,
        params: Any,
        timeout: float | None = None,
    ) -> Any:
        """Send a JSON-RPC request and return the result."""
        return await self._transport.call(method, params, timeout=timeout)

    def _route_update(self, msg: dict[str, Any]) -> None:
        """Route a `session/update` notification to its in-flight prompt."""
        self._transport._route_update(msg)

    # ---------------------------------------------------------------- session helpers

    @staticmethod
    def _is_stale_session_error(exc: BaseException) -> bool:
        """Return True if an ACP error indicates the session id is no longer valid."""
        if isinstance(exc, AcpSessionStaleError):
            return True
        if isinstance(exc, (AcpModelError, AcpMcpError, AcpTransportError)):
            return False
        msg = str(exc).lower()
        # Model or MCP schema failures are not stale-session signals; treating them
        # as such leads to futile rehydration loops.
        if "model" in msg or "mcp" in msg:
            return False
        return "session" in msg and (
            "not found" in msg
            or "invalid" in msg
            or "expired" in msg
            or "stale" in msg
            or "empty reply" in msg
        )

    @staticmethod
    def _is_method_not_found(exc: AcpError) -> bool:
        """Return True if the ACP agent reports a method it does not implement."""
        if exc.code == -32601:
            return True
        msg = str(exc.message or "").lower()
        return "method not found" in msg or (
            "not found" in msg and "session/resume" in str(exc.method or "").lower()
        )

    async def _send_cancel_notification(self, session_id: str) -> None:
        """Send a fire-and-forget `session/cancel` notification."""
        await self._transport._send(
            {
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": session_id},
            },
            timeout=self._control_timeout,
        )

    async def _session_alive(self, session_id: str) -> bool:
        """Try a cheap config update to see if the session still exists."""
        try:
            await self._call(
                "session/set_config_option",
                {
                    "sessionId": session_id,
                    "configId": "mode",
                    "value": self.acp_mode,
                },
                timeout=10.0,
            )
            return True
        except (AcpError, RuntimeError) as exc:
            if self._is_stale_session_error(exc):
                return False
            raise

    def _resume_jitter(self, attempt: int) -> float:
        """Return an exponential-backoff delay with a small amount of jitter."""
        base = self.acp_resume_retry_base_seconds
        cap = self.acp_resume_retry_max_seconds
        delay = min(base * (2**attempt), cap) + random.uniform(0, 0.1)
        return min(delay, cap)

    async def _call_with_resume_retry(
        self,
        method: str,
        params: dict[str, Any],
        call_timeout: float,
    ) -> Any:
        """Call an ACP resume method, retrying transient errors with jitter.

        Does not retry a JSON-RPC "method not found" error; that is the
        caller's signal to try a different method.
        """
        last_exc: Exception | None = None
        max_attempts = self.acp_resume_max_retries + 1
        for attempt in range(max_attempts):
            try:
                return await self._call(method, params, timeout=call_timeout)
            except AcpError as exc:
                if self._is_method_not_found(exc):
                    raise
                last_exc = exc
            except TimeoutError as exc:
                last_exc = exc
            if attempt < max_attempts - 1:
                await asyncio.sleep(self._resume_jitter(attempt))
        assert last_exc is not None
        raise last_exc

    async def _resume_session(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        """Resume a persisted ACP session.

        ``session/resume`` is the ACP-unstable method intended for continuing a
        user-visible session; ``session/load`` is the stable equivalent used by
        the Devin CLI.  We try ``session/resume`` first and fall back to
        ``session/load`` so the harness works with both current and future ACP
        servers.

        The active MCP server list is written to the ACP subprocess's
        ``mcp_config.json`` by ``_prepare_devin_home``; ``devin acp`` 3000.6.7+
        rejects inline ``mcpServers`` definitions in the resume/load payload, so
        we pass an empty list just like we do for ``session/new``.
        """
        use_cwd = str(cwd) if cwd else os.getcwd()
        use_model = _normalize_model(model or self.model)
        # Keep the client-side MCP list in sync so future transport restarts
        # write the correct mcp_config.json.
        if mcp_servers is not None:
            self._mcp_servers = self._sandbox.normalize_mcp_servers(mcp_servers)
        resume_params: dict[str, Any] = {
            "sessionId": session_id,
            "cwd": use_cwd,
            "mcpServers": [],
        }
        load_params: dict[str, Any] = {
            "sessionId": session_id,
            "cwd": use_cwd,
            "mcpServers": [],
        }

        if self._lifecycle_log is not None:
            self._lifecycle_log.write(
                "session.resume.attempt",
                session_id=session_id,
                model=use_model,
                detail={"cwd": str(use_cwd)},
            )

        resume_method = "resume"
        start = time.perf_counter()
        try:
            call_timeout = self._control.call_timeout()
            try:
                await self._call_with_resume_retry(
                    "session/resume",
                    resume_params,
                    call_timeout,
                )
            except AcpError as exc:
                if self._is_method_not_found(exc):
                    logger.debug(
                        "session/resume not supported; trying session/load for %s", session_id
                    )
                    resume_method = "load"
                    await self._call_with_resume_retry(
                        "session/load",
                        load_params,
                        call_timeout,
                    )
                else:
                    raise
            await self._apply_session_config(session_id, use_model, timeout=call_timeout)
        except (AcpError, TimeoutError) as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if self.metrics is not None:
                self.metrics.inc("acp_resume_total", result="failure", method=resume_method)
                self.metrics.set("acp_resume_latency_ms", duration_ms, result="failure")
            if self._lifecycle_log is not None:
                self._lifecycle_log.write(
                    "session.resume.failure",
                    session_id=session_id,
                    model=use_model,
                    detail={
                        "cwd": str(use_cwd),
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if self.metrics is not None:
            self.metrics.inc("acp_resume_total", result="success", method=resume_method)
            self.metrics.set("acp_resume_latency_ms", duration_ms, result="success")
        if self._lifecycle_log is not None:
            self._lifecycle_log.write(
                "session.resume.success",
                session_id=session_id,
                model=use_model,
                detail={
                    "cwd": str(use_cwd),
                    "method": resume_method,
                    "duration_ms": duration_ms,
                },
            )
        return session_id

    async def _session_load(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        """Load a persisted ACP session with ``session/load``.

        Exposed separately so callers can force the stable load path.
        """
        use_cwd = str(cwd) if cwd else os.getcwd()
        use_model = _normalize_model(model or self.model)
        # Keep the client-side MCP list in sync so future transport restarts
        # write the correct mcp_config.json.
        if mcp_servers is not None:
            self._mcp_servers = self._sandbox.normalize_mcp_servers(mcp_servers)
        if self._lifecycle_log is not None:
            self._lifecycle_log.write(
                "session.load.attempt",
                session_id=session_id,
                model=use_model,
                detail={"cwd": str(use_cwd)},
            )
        call_timeout = self._control.call_timeout()
        start = time.perf_counter()
        try:
            await self._call(
                "session/load",
                {
                    "sessionId": session_id,
                    "cwd": use_cwd,
                    "mcpServers": [],
                },
                timeout=call_timeout,
            )
            await self._apply_session_config(session_id, use_model, timeout=call_timeout)
        except (AcpError, TimeoutError) as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if self.metrics is not None:
                self.metrics.inc("acp_resume_total", result="failure", method="load")
                self.metrics.set("acp_resume_latency_ms", duration_ms, result="failure")
            if self._lifecycle_log is not None:
                self._lifecycle_log.write(
                    "session.load.failure",
                    session_id=session_id,
                    model=use_model,
                    detail={
                        "cwd": str(use_cwd),
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if self.metrics is not None:
            self.metrics.inc("acp_resume_total", result="success", method="load")
            self.metrics.set("acp_resume_latency_ms", duration_ms, result="success")
        if self._lifecycle_log is not None:
            self._lifecycle_log.write(
                "session.load.success",
                session_id=session_id,
                model=use_model,
                detail={"cwd": str(use_cwd), "duration_ms": duration_ms},
            )
        return session_id

    async def _apply_session_config(
        self,
        session_id: str,
        use_model: str,
        *,
        timeout: float | None = None,
    ) -> None:
        """Set mode and model on a freshly created or resumed session."""
        call_timeout = timeout or self._control.call_timeout()
        await self._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "mode", "value": self.acp_mode},
            timeout=call_timeout,
        )
        await self._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "model", "value": use_model},
            timeout=call_timeout,
        )
        self._session_models[session_id] = use_model

    async def _create_session(
        self,
        prompt_text: str,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        soft_timeout: float | None = None,
        chat_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        use_model = _normalize_model(model or self.model)
        use_cwd = str(cwd) if cwd else os.getcwd()
        if cwd is not None:
            cwd.mkdir(parents=True, exist_ok=True)

        # `devin acp` 3000.6.7+ loads MCP servers from the isolated
        # `mcp_config.json` written by `_prepare_devin_home`.  `session/new`
        # no longer accepts inline server definitions in its `mcpServers`
        # parameter; passing them produces "data did not match any variant of
        # untagged enum McpServer".  Pass an empty list and rely on the config
        # file so the active server list is still honored.
        # Cap session/new so a hung subprocess restart (e.g. slow MCP server init)
        # does not hold the harness lock for multiple minutes. The watchdog
        # stall threshold is also bounded by _watchdog_timeout, so align the
        # call timeout with that ceiling (at least 60s to allow normal init).
        if self._lifecycle_log is not None:
            self._lifecycle_log.write(
                "session.new.attempt",
                chat_id=chat_id,
                model=use_model,
                detail={"cwd": str(use_cwd)},
            )

        session_new_timeout = self._control.call_timeout()
        start = time.perf_counter()
        try:
            session = await self._call(
                "session/new",
                {"cwd": use_cwd, "mcpServers": []},
                timeout=session_new_timeout,
            )
            session_id = session["sessionId"]
        except (AcpError, TimeoutError) as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if self.metrics is not None:
                self.metrics.inc("acp_resume_total", result="failure", method="new")
                self.metrics.set("acp_resume_latency_ms", duration_ms, result="failure")
            if self._lifecycle_log is not None:
                self._lifecycle_log.write(
                    "session.new.failure",
                    chat_id=chat_id,
                    model=use_model,
                    detail={
                        "cwd": str(use_cwd),
                        "error": str(exc),
                        "duration_ms": duration_ms,
                    },
                )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if self.metrics is not None:
            self.metrics.inc("acp_resume_total", result="success", method="new")
            self.metrics.set("acp_resume_latency_ms", duration_ms, result="success")
        if self._lifecycle_log is not None:
            self._lifecycle_log.write(
                "session.new.success",
                chat_id=chat_id,
                session_id=session_id,
                model=use_model,
                detail={"cwd": str(use_cwd), "duration_ms": duration_ms},
            )

        if self._model_options is None:
            self._model_options = self._extract_model_options(session)

        # Honor the requested mode and model for this session.
        await self._apply_session_config(session_id, use_model, timeout=session_new_timeout)

        return await self._prompt(
            session_id,
            prompt_text,
            soft_timeout=soft_timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )

    async def _send_message(
        self,
        session_id: str,
        prompt_text: str,
        cwd: Path | None = None,
        model: str | None = None,
        soft_timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        use_model = _normalize_model(model or self.model)
        # Only set the session model on follow-up when it has changed. Repeated
        # no-op model changes can re-render the session's system prefix and
        # destabilize the ACP subprocess.
        if self._session_models.get(session_id) != use_model:
            await self._call(
                "session/set_config_option",
                {"sessionId": session_id, "configId": "model", "value": use_model},
            )
            self._session_models[session_id] = use_model

        return await self._prompt(
            session_id,
            prompt_text,
            soft_timeout=soft_timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )

    async def _list_models(self) -> list[str]:
        """Create a throwaway session to discover the advertised model list."""
        probe_cwd = Path(os.getcwd()) / ".acp_model_probe"
        probe_cwd.mkdir(exist_ok=True)

        session = await self._call(
            "session/new",
            {"cwd": str(probe_cwd), "mcpServers": []},
        )
        self._model_options = self._extract_model_options(session)
        return self._model_options

    @staticmethod
    def _extract_model_options(session_result: dict[str, Any]) -> list[str]:
        for opt in session_result.get("configOptions", []):
            if opt.get("id") == "model":
                return [o["value"] for o in opt.get("options", [])]
        return []

    async def _prompt(
        self,
        session_id: str,
        text: str,
        soft_timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Send `session/prompt` and return the streamed reply."""
        self._next_id += 1
        prompt_id = self._next_id
        prompt = _Prompt(
            session_id=session_id,
            prompt_id=prompt_id,
            text=text,
            future=self._loop.create_future(),
            cancel_done=self._loop.create_future(),
            soft_timeout=soft_timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )
        self._pending[prompt_id] = prompt.future
        self._active_prompts[session_id] = prompt
        with self._lock:
            self._last_stdout_at = time.monotonic()
            self._last_progress_at = time.monotonic()

        # If a cancel arrived before we registered the prompt, honor it now.
        if session_id in self._pending_cancels:
            self._pending_cancels.discard(session_id)
            prompt.cancelled = True
            if not prompt.cancel_done.done():
                prompt.cancel_done.set_result(None)
            self._loop.create_task(self._send_cancel_notification(session_id))

        # If the caller cancelled before we started, just return.
        if prompt.cancelled:
            return AcpPromptResult(
                reply="",
                session_id=session_id,
                cancelled=True,
                partial=True,
            )

        timeout_task: asyncio.Task[None] | None = None
        try:
            await self._transport._send(
                {
                    "jsonrpc": "2.0",
                    "id": prompt_id,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": text}],
                    },
                },
                timeout=self._control_timeout,
            )

            if soft_timeout is not None and soft_timeout > 0:
                timeout_task = self._loop.create_task(
                    self._soft_timeout_canceller(prompt, soft_timeout)
                )

            start = self._loop.time()
            done, _pending = await asyncio.wait(
                [prompt.future, prompt.cancel_done],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=self.timeout,
            )

            raw: dict[str, Any] | None = None
            if prompt.future in done:
                raw = prompt.future.result()
            elif prompt.cancel_done in done:
                # Cancel was requested. Give the server a short grace period to
                # finish the aborted turn and send the prompt response.
                elapsed = self._loop.time() - start
                if self.timeout is not None:
                    remaining = max(0.0, self.timeout - elapsed)
                    wait_for = min(5.0, remaining)
                else:
                    wait_for = 5.0
                try:
                    raw = await asyncio.wait_for(prompt.future, timeout=wait_for)
                except TimeoutError:
                    raw = None
            else:
                # Hard timeout on the prompt itself.
                logger.warning("ACP prompt hard timeout for session %s", session_id)
                try:
                    await self._send_cancel_notification(session_id)
                except Exception:
                    logger.exception("Failed to send cancel on hard timeout")
                try:
                    raw = await asyncio.wait_for(prompt.future, timeout=5.0)
                except TimeoutError:
                    raw = None

            if raw is None:
                # We never got a prompt response. Return the partial stream.
                return AcpPromptResult(
                    reply="".join(prompt.chunks),
                    session_id=session_id,
                    stop_reason="timeout",
                    cancelled=prompt.cancelled,
                    partial=True,
                    timed_out=True,
                    updates=prompt.updates,
                )

            if "error" in raw:
                raise _acp_error_from_response("session/prompt", raw["error"])
            result = raw.get("result", {})
            stop_reason = result.get("stopReason")
            stopped_early = stop_reason in ("cancelled", "timeout")
            cancelled = prompt.cancelled
            timed_out = prompt.timed_out or stop_reason == "timeout"
            reply = "".join(prompt.chunks)

            if stop_reason is None and not reply:
                # The server returned a prompt response with no text and no
                # stop reason. This usually means the ACP session is stale and
                # the prompt was not actually processed.
                logger.warning(
                    "ACP session %s returned an empty prompt response; treating as stale",
                    session_id,
                )
                raise AcpSessionStaleError(
                    "session/prompt",
                    {
                        "code": -32002,
                        "message": "Resource not found",
                        "data": {"uri": f"Session {session_id} returned an empty reply"},
                    },
                )

            # Mark as partial if the server stopped early (cancelled/timeout),
            # the client explicitly cancelled, or the soft/hard timeout fired.
            # A missing stopReason on a non-empty reply is a normal completion.
            partial = cancelled or stopped_early or prompt.timed_out
            return AcpPromptResult(
                reply=reply,
                session_id=session_id,
                stop_reason=stop_reason,
                usage=result.get("usage"),
                cancelled=cancelled,
                partial=partial,
                timed_out=timed_out,
                updates=prompt.updates,
            )
        except asyncio.CancelledError:
            logger.warning("ACP prompt cancelled by watchdog/timeout")
            return AcpPromptResult(
                reply="".join(prompt.chunks),
                session_id=session_id,
                stop_reason="timeout",
                cancelled=prompt.cancelled,
                partial=True,
                timed_out=True,
                updates=prompt.updates,
            )
        finally:
            if timeout_task is not None and not timeout_task.done():
                timeout_task.cancel()
            self._active_prompts.pop(session_id, None)
            self._pending.pop(prompt_id, None)

    async def _soft_timeout_canceller(self, prompt: _Prompt, delay: float) -> None:
        """Fire a `session/cancel` notification after `delay` seconds."""
        await asyncio.sleep(delay)
        if prompt.future.done() or prompt.cancel_done.done():
            return
        prompt.timed_out = True
        if not prompt.cancel_done.done():
            prompt.cancel_done.set_result(None)
        logger.debug("ACP soft timeout for session %s; sending cancel", prompt.session_id)
        try:
            await self._send_cancel_notification(prompt.session_id)
        except Exception:
            logger.exception("Failed to send soft timeout cancel for %s", prompt.session_id)
