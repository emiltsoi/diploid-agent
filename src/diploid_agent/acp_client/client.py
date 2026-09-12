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
)
from diploid_agent.acp_client.lifecycle import AcpLifecycleLog, AcpRestartHistory
from diploid_agent.acp_client.sandbox import AcpSandbox
from diploid_agent.acp_client.sessions import AcpSessionOps
from diploid_agent.acp_client.state import AcpClientState, _StateAttr
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
        silence_warn_after: float = 600.0,
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

        # Mutable state shared with the transport/session/watchdog
        # components; ``client._x`` names resolve onto it via ``_StateAttr``.
        self._state = AcpClientState()

        # Low-level transport state.
        self._transport = AcpTransport(self)

        # Timing and control configuration.
        self._control_timeout = control_timeout
        self._startup_timeout = startup_timeout
        self._watchdog_interval = watchdog_interval
        self._watchdog_timeout = watchdog_timeout
        self._silence_warn_after = silence_warn_after

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
        if lifecycle_log is not None:
            lifecycle_log.context = self._lifecycle_context
        # Generation of the transport for which we last logged transport.stop
        # is tracked on ``self._state._logged_stop_gen`` so repeated close()
        # calls do not emit duplicate stop events.
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
        self._sessions = AcpSessionOps(self)

        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Shared state aliases: ``client._x`` names resolve onto ``self._state``
    # so existing test seams and internal callers keep working.

    _lock = _StateAttr("_lock")
    _lifecycle_lock = _StateAttr("_lifecycle_lock")
    _next_id = _StateAttr("_next_id")
    _active_prompts = _StateAttr("_active_prompts")
    _session_models = _StateAttr("_session_models")
    _pending_cancels = _StateAttr("_pending_cancels")
    _model_options = _StateAttr("_model_options")
    _mcp_servers = _StateAttr("_mcp_servers")
    _logged_stop_gen = _StateAttr("_logged_stop_gen")

    _loop = _StateAttr("_loop")
    _thread = _StateAttr("_thread")
    _proc = _StateAttr("_proc")
    _reader_task = _StateAttr("_reader_task")
    _stderr_task = _StateAttr("_stderr_task")
    _inflight_future = _StateAttr("_inflight_future")
    _inflight_deadline = _StateAttr("_inflight_deadline")
    _last_stdout_at = _StateAttr("_last_stdout_at")
    _last_progress_at = _StateAttr("_last_progress_at")
    _last_request_at = _StateAttr("_last_request_at")
    _last_control_call_deadline = _StateAttr("_last_control_call_deadline")
    _pending = _StateAttr("_pending")
    _transport_healthy = _StateAttr("_transport_healthy")
    _initialized = _StateAttr("_initialized")
    _terminated = _StateAttr("_terminated")
    _restart_history = _StateAttr("_restart_history")

    # ------------------------------------------------------------------
    # Watchdog and control aliases.

    @property
    def transport_pid(self) -> int | None:
        """Return the OS pid of the ACP child process, or None if not running."""
        proc = self._proc
        if proc is None:
            return None
        return proc.pid

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
        timeout: float | None = None,
        chat_id: str | None = None,
        background: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Create a new ACP session, send the first prompt, return the result."""
        self._ensure_started(mcp_servers)
        normalized_mcp_servers = self._sandbox.normalize_mcp_servers(mcp_servers)
        if cwd is not None:
            cwd = Path(cwd)
        effective_timeout = timeout if timeout is not None else self.timeout
        result = self._run(
            self._create_session(
                prompt_text,
                cwd=cwd,
                model=model,
                mcp_servers=normalized_mcp_servers,
                soft_timeout=soft_timeout,
                timeout=timeout,
                chat_id=chat_id,
                on_chunk=on_chunk,
                on_update=on_update,
            ),
            timeout=effective_timeout + self._control_timeout + 30.0
            if effective_timeout is not None
            else None,
            background=background,
        )
        if result and result.stop_reason == "timeout" and not background:
            # Force a transport restart so the next turn does not hang on
            # session/new while the old child is still busy.  Background
            # calls skip this: their timeout must not poison the shared
            # transport under a live foreground session.
            with self._lock:
                self._transport_healthy = False
        return result

    def send_message(
        self,
        session_id: str,
        prompt_text: str,
        *,
        cwd: str | Path | None = None,
        model: str | None = None,
        soft_timeout: float | None = None,
        timeout: float | None = None,
        background: bool = False,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Send a follow-up prompt to an existing ACP session."""
        self._ensure_started()
        if cwd is not None:
            cwd = Path(cwd)
        effective_timeout = timeout if timeout is not None else self.timeout
        result = self._run(
            self._send_message(
                session_id,
                prompt_text,
                cwd=cwd,
                model=model,
                soft_timeout=soft_timeout,
                timeout=timeout,
                on_chunk=on_chunk,
                on_update=on_update,
            ),
            timeout=effective_timeout + self._control_timeout + 30.0
            if effective_timeout is not None
            else None,
            background=background,
        )
        if result and result.stop_reason == "timeout" and not background:
            with self._lock:
                self._transport_healthy = False
        return result

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
        timeout: float | None = None,
    ) -> str:
        """Resume a persisted ACP session and return the active session id.

        Tries ``session/resume`` first, then falls back to ``session/load`` if
        the agent does not advertise the unstable ``session/resume`` method.
        After a successful resume the session mode and model are re-applied.

        ``timeout`` is a real end-to-end budget shared by the resume/load
        attempts and the config re-apply, not per-call headroom: a stalled
        resume is an opportunistic optimization and should give up quickly so
        the caller can fall back to prompt rehydration.
        """
        self._ensure_started(mcp_servers)
        effective_timeout = timeout if timeout is not None else self.timeout
        return self._run(
            self._resume_session(
                session_id,
                cwd=cwd,
                model=model,
                mcp_servers=mcp_servers,
                timeout=effective_timeout,
            ),
            timeout=effective_timeout + 30.0 if effective_timeout is not None else None,
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

    def _lifecycle_context(self) -> dict[str, Any]:
        """Process/transport identity stamped onto every lifecycle event."""
        return {"pid": os.getpid(), "transport_gen": self._transport.generation}

    def close(self) -> None:
        """Terminate the ACP subprocess and stop the background loop."""
        with self._lifecycle_lock:
            with self._lock:
                if not self._initialized and self._loop is None and self._proc is None:
                    return
                gen = self._transport.generation
                if self._lifecycle_log is not None and gen > 0 and gen != self._logged_stop_gen:
                    self._lifecycle_log.write("transport.stop")
                    self._logged_stop_gen = gen
                self._initialized = False
                self._transport_healthy = False
                # Snapshot this generation's teardown handles once.  The
                # _close_transport done-callback fires on whichever thread
                # completes the future -- potentially after a follow-on
                # _ensure_started installed the next generation's loop -- and
                # the same applies to the slot clears in ``finally``.
                # Dereferencing ``self._loop`` at fire time stopped the NEW
                # loop out from under its _start_transport (the "ghost
                # generation" behind the 86 s resume-stall gap).
                loop = self._loop
                thread = self._thread
                proc = self._proc
                reader_task = self._reader_task
                stderr_task = self._stderr_task

            # ``_lock`` is released for the blocking section: _close_transport
            # needs it (via _unblock_inflight) to complete, and
            # ``_lifecycle_lock`` alone already serializes us against
            # _ensure_started/restart_transport/watchdog recovery.
            try:
                if (
                    thread is not None
                    and thread.is_alive()
                    and loop is not None
                    and loop.is_running()
                ):
                    # Schedule _close_transport on the background loop and stop
                    # the loop only after the coroutine has actually completed.
                    # Stopping the loop from the main thread immediately after
                    # scheduling the close task can leave the coroutine unawaited
                    # and generate a RuntimeWarning.
                    future: concurrent.futures.Future[Any] = asyncio.run_coroutine_threadsafe(
                        self._close_transport(), loop
                    )

                    def _stop_loop_soon(_: Any) -> None:
                        if loop.is_running():
                            try:
                                loop.call_soon_threadsafe(loop.stop)
                            except RuntimeError:
                                pass

                    future.add_done_callback(_stop_loop_soon)
                    try:
                        future.result(timeout=10.0)
                    except (RuntimeError, TimeoutError) as exc:
                        logger.warning("ACP close transport failed", exc_info=exc)
                        if not future.done():
                            # Cancel the close task so it is awaited (up to its
                            # first suspension) before the loop is stopped.
                            future.cancel()
                        if proc is not None and proc.returncode is None:
                            try:
                                proc.kill()
                            except (OSError, ProcessLookupError) as kill_exc:
                                logger.warning(
                                    "Failed to kill ACP process during close",
                                    exc_info=kill_exc,
                                )
                        if loop.is_running():
                            try:
                                loop.call_soon_threadsafe(loop.stop)
                            except RuntimeError:
                                pass
                else:
                    # The background loop is not running; kill the process
                    # directly and do not schedule a coroutine that can never
                    # be awaited.
                    if proc is not None and proc.returncode is None:
                        try:
                            proc.kill()
                        except (OSError, ProcessLookupError) as kill_exc:
                            logger.warning(
                                "Failed to kill ACP process during close",
                                exc_info=kill_exc,
                            )
            finally:
                self._watchdog.stop()
                if thread is not None and thread.is_alive():
                    thread.join(timeout=10.0)
                with self._lock:
                    # Clear only slots still pointing at this generation's
                    # handles; a racing restart may have installed replacements.
                    if self._loop is loop:
                        self._loop = None
                    if self._thread is thread:
                        self._thread = None
                    if self._proc is proc:
                        self._proc = None
                    if self._reader_task is reader_task:
                        self._reader_task = None
                    if self._stderr_task is stderr_task:
                        self._stderr_task = None
                if loop is not None and not loop.is_closed() and not loop.is_running():
                    try:
                        loop.close()
                    except RuntimeError:
                        pass
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

    def _run(self, coro: Any, timeout: float | None = None, background: bool = False) -> Any:
        """Run a coroutine on the background loop and block for the result."""
        return self._transport.run(coro, timeout=timeout, background=background)

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
        self._transport.ensure_started(mcp_servers)

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
        # Serialize with concurrent _ensure_started/close callers so a racing
        # restart cannot swap the loop mid-start or spawn a second child.
        with self._lifecycle_lock:
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

    # The session-level JSON-RPC operations live in ``AcpSessionOps``
    # (acp_client/sessions.py).  The delegates below preserve the historical
    # ``client._*`` call surface used by tests and by the transport/watchdog
    # collaborators, so monkeypatched seams keep intercepting.

    async def _send_cancel_notification(self, session_id: str) -> None:
        """Send a fire-and-forget `session/cancel` notification."""
        await self._sessions._send_cancel_notification(session_id)

    async def _session_alive(self, session_id: str) -> bool:
        """Try a cheap config update to see if the session still exists."""
        return await self._sessions._session_alive(session_id)

    def _resume_jitter(self, attempt: int) -> float:
        """Return an exponential-backoff delay with a small amount of jitter."""
        return self._sessions._resume_jitter(attempt)

    async def _call_with_resume_retry(
        self,
        method: str,
        params: dict[str, Any],
        call_timeout: float,
        budget: Callable[[], float] | None = None,
    ) -> Any:
        """Call an ACP resume method, retrying transient errors with jitter."""
        return await self._sessions._call_with_resume_retry(
            method, params, call_timeout, budget=budget
        )

    async def _resume_session(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Resume a persisted ACP session and return the active session id."""
        return await self._sessions._resume_session(
            session_id,
            cwd=cwd,
            model=model,
            mcp_servers=mcp_servers,
            timeout=timeout,
        )

    async def _session_load(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> str:
        """Load a persisted ACP session with ``session/load``."""
        return await self._sessions._session_load(
            session_id,
            cwd=cwd,
            model=model,
            mcp_servers=mcp_servers,
        )

    async def _apply_session_config(
        self,
        session_id: str,
        use_model: str,
        *,
        timeout: float | None = None,
    ) -> None:
        """Set mode and model on a freshly created or resumed session."""
        await self._sessions._apply_session_config(session_id, use_model, timeout=timeout)

    async def _create_session(
        self,
        prompt_text: str,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        soft_timeout: float | None = None,
        timeout: float | None = None,
        chat_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Create a new ACP session and run its first prompt."""
        return await self._sessions._create_session(
            prompt_text,
            cwd=cwd,
            model=model,
            mcp_servers=mcp_servers,
            soft_timeout=soft_timeout,
            timeout=timeout,
            chat_id=chat_id,
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
        timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Send a follow-up prompt to an existing ACP session."""
        return await self._sessions._send_message(
            session_id,
            prompt_text,
            cwd=cwd,
            model=model,
            soft_timeout=soft_timeout,
            timeout=timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )

    async def _list_models(self) -> list[str]:
        """Create a throwaway session to discover the advertised model list."""
        return await self._sessions._list_models()

    @staticmethod
    def _extract_model_options(session_result: dict[str, Any]) -> list[str]:
        """Extract the advertised model IDs from a ``session/new`` result."""
        return AcpSessionOps._extract_model_options(session_result)

    async def _prompt(
        self,
        session_id: str,
        text: str,
        soft_timeout: float | None = None,
        timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Send `session/prompt` and return the streamed reply."""
        return await self._sessions._prompt(
            session_id,
            text,
            soft_timeout=soft_timeout,
            timeout=timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )

    async def _soft_timeout_canceller(self, prompt: _Prompt, delay: float) -> None:
        """Fire a `session/cancel` notification after `delay` seconds."""
        await self._sessions._soft_timeout_canceller(prompt, delay)
