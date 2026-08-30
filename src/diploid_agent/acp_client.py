"""ACP client: drive an ACP-compatible agent binary over stdio JSON-RPC.

Provides a long-lived ACP session, giving real-time `agent_message_chunk`
streaming, mid-turn cancellation via `session/cancel` notification, and
steering via `session/set_config_option` (mode/model).
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import dataclasses
import json
import logging
import os
import shutil
import signal
import tempfile
import threading
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ACP_MODE_MAP = {
    "auto": "smart",
    "normal": "accept-edits",
    "accept-edits": "accept-edits",
    "smart": "smart",
    "dangerous": "bypass",
    "bypass": "bypass",
}


@dataclasses.dataclass
class AcpPromptResult:
    """Result of a single `session/prompt` turn."""

    reply: str
    session_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    cancelled: bool = False
    partial: bool = False
    timed_out: bool = False
    updates: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Prompt:
    """In-flight ACP prompt state."""

    session_id: str
    prompt_id: int
    text: str
    future: asyncio.Future[dict[str, Any]]
    cancel_done: asyncio.Future[None]
    soft_timeout: float | None = None
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    chunks: list[str] = dataclasses.field(default_factory=list)
    updates: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    on_chunk: Callable[[str], None] | None = None
    on_update: Callable[[dict[str, Any]], None] | None = None
    cancelled: bool = False
    timed_out: bool = False


class AcpError(RuntimeError):
    """Base class for ACP JSON-RPC errors returned by the child process."""

    def __init__(self, method: str, error: dict[str, Any]) -> None:
        self.method = method
        self.code = error.get("code")
        self.message = error.get("message", "")
        self.data = error.get("data") or {}
        self.error = error
        super().__init__(f"ACP {method} failed: {error}")


class AcpTransportError(AcpError):
    """The ACP transport itself did not respond or the child process died."""

    def __init__(self, method: str, error: dict[str, Any] | None = None, msg: str = "") -> None:
        if error is None:
            error = {"message": msg}
        super().__init__(method, error)


class AcpSessionStaleError(AcpError):
    """The ACP session id is no longer valid on an otherwise healthy transport."""


class AcpModelError(AcpError):
    """The requested model is not available to the ACP child."""


class AcpMcpError(AcpError):
    """The MCP server configuration was rejected by the ACP child."""


def _acp_error_from_response(method: str, error: dict[str, Any]) -> AcpError:
    """Return the appropriate typed exception for a JSON-RPC error response.

    Classifier rules (from observed `devin acp` 3000.6.7 payloads):

    - ``-32602`` / ``Invalid params`` with an ``mcpServers`` / ``McpServer``
      mention in the data or message -> ``AcpMcpError``.
    - ``-32002`` / ``Resource not found``
      - ``data.uri`` / message starts with ``Model not found`` or mentions
        ``model`` -> ``AcpModelError``.
      - ``data.uri`` / message contains ``Session not found`` or a session id
        and does *not* look like a model error -> ``AcpSessionStaleError``.
    - Other ACP protocol or transport failures -> ``AcpError``.
    """
    code = error.get("code")
    message = (error.get("message") or "").lower()
    data = error.get("data") or {}
    data_str = json.dumps(data, default=str).lower()
    combined = f"{message} {data_str}"

    if code == -32602:
        if "mcp" in combined or "mcpservers" in combined:
            return AcpMcpError(method, error)
        return AcpError(method, error)

    if code == -32002:
        # The `data.uri` field is sometimes the human-readable detail string.
        data_uri = str(data.get("uri", "")).lower()
        detail = data_uri or message

        # `devin acp 3000.6.7` reports a stale/non-existent session as
        # "Model not found: <model>. Available models: " with an *empty* model
        # list. A genuine "model not found" has a non-empty available-models list.
        available_models_str = ""
        if "available models:" in detail:
            available_models_str = detail.split("available models:", 1)[1].strip(" \"\'[]").strip()

        looks_like_model_error = (
            detail.startswith("model not found")
            or ("model" in detail and "not found" in detail)
        )
        if looks_like_model_error:
            if available_models_str == "":
                return AcpSessionStaleError(method, error)
            return AcpModelError(method, error)

        if detail.startswith("session not found") or (
            "session" in detail and "not found" in detail
        ):
            return AcpSessionStaleError(method, error)

        # Generic resource not found that does not name a session is probably a
        # model or other configuration error, not a stale session.
        if "model" in detail and available_models_str != "":
            return AcpModelError(method, error)
        if "session" in detail or (method and method.startswith("session/")):
            return AcpSessionStaleError(method, error)

    return AcpError(method, error)


def _load_windsurf_api_key() -> str | None:
    """Return the Windsurf API key from env or the Devin CLI credentials file."""
    if os.environ.get("WINDSURF_API_KEY"):
        return os.environ["WINDSURF_API_KEY"]

    creds_path = Path.home() / ".local" / "share" / "devin" / "credentials.toml"
    if creds_path.exists():
        try:
            data = tomllib.loads(creds_path.read_text())
            return data.get("windsurf_api_key")
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning("Failed to read Devin credentials from %s: %s", creds_path, exc)
    return None


def _normalize_model(model: str) -> str:
    """Return the canonical ACP model id for an alias.

    The Devin CLI and docs use dotted aliases like `swe-1.7`, but ACP's
    `session/set_config_option` expects the dashed form `swe-1-7`.
    """
    return model.replace(".", "-")


def _devin_default_start_args(model: str) -> list[str]:
    """Return the default start arguments for the Devin ACP binary."""
    return ["acp", "--model", model]


def _resolve_agent_bin(agent_bin: str | Path) -> Path:
    """Resolve an agent binary path, falling back to PATH by file name."""
    p = Path(agent_bin).expanduser()
    if p.exists():
        return p
    name = p.name if p.name != "." else str(agent_bin)
    found = shutil.which(name)
    if found:
        return Path(found)
    raise RuntimeError(f"agent binary not found: {agent_bin}")


# Backward-compatible alias (deprecated).
_resolve_devin_bin = _resolve_agent_bin


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
        timeout: float = 900.0,
        startup_timeout: float = 30.0,
        control_timeout: float = 120.0,
        watchdog_interval: float = 10.0,
        watchdog_timeout: float = 120.0,
        max_restarts: int = 3,
        restart_backoff_window: float = 300.0,
        agent_bin: str | Path = "~/.local/bin/devin",
        start_args: list[str] | None = None,
        api_key: str | None = None,
        metrics: Any | None = None,
    ):
        self.model = _normalize_model(model)
        self.acp_mode = _ACP_MODE_MAP.get(permission_mode, "bypass")
        self.timeout = timeout
        self.agent_bin = _resolve_agent_bin(agent_bin)
        self.start_args = start_args or _devin_default_start_args(self.model)
        self.metrics = metrics

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

        # Background loop state.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_running = False
        self._watchdog_interval = watchdog_interval
        self._watchdog_timeout = watchdog_timeout
        self._last_stdout_at: float = 0.0
        self._last_progress_at: float = 0.0
        self._last_request_at: float = 0.0
        self._last_control_call_deadline: float = 0.0
        self._inflight_future: concurrent.futures.Future[Any] | None = None
        self._inflight_deadline: float = 0.0
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._active_prompts: dict[str, _Prompt] = {}
        self._session_models: dict[str, str] = {}
        self._pending_cancels: set[str] = set()
        self._lock = threading.RLock()
        self._initialized = False
        self._transport_healthy = False
        self._control_timeout = control_timeout
        self._startup_timeout = startup_timeout
        self._model_options: list[str] | None = None
        self._devin_home: Path | None = None
        self._mcp_servers: list[dict[str, Any]] = []

        # Restart backoff: avoid an infinite kill/restart loop when the ACP child
        # cannot be made healthy.
        self._max_restarts = max(0, max_restarts)
        self._restart_backoff_window = max(1.0, restart_backoff_window)
        self._restart_history: list[float] = []

        atexit.register(self.close)

    def _prepare_devin_home(
        self,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        """Create an isolated HOME for the ACP child process.

        Devin's bundled MCP config (e.g. `~/.codeium/windsurf/mcp_config.json`)
        can include MCP servers that deadlock or saturate and block `devin acp`
        startup indefinitely. We create a sanitized home directory and write the
        active MCP server list into `mcp_config.json` before `devin acp` starts,
        because devin 3000.6.7+ loads servers from that file.
        """
        if self._devin_home is not None and self._devin_home.exists():
            # Re-sanitize the MCP config on every (re)start in case the previous
            # `devin` child wrote to it.
            self._write_mcp_configs(mcp_servers)
            return

        self._devin_home = Path(tempfile.mkdtemp(prefix="acp-home-"))
        config_dir = self._devin_home / ".config" / "devin"
        config_dir.mkdir(parents=True, exist_ok=True)
        codeium_dir = self._devin_home / ".codeium" / "windsurf"
        codeium_dir.mkdir(parents=True, exist_ok=True)

        user_config = Path.home() / ".config" / "devin" / "config.json"
        if user_config.exists():
            try:
                (config_dir / "config.json").write_text(user_config.read_text())
            except OSError:
                logger.warning("Failed to copy devin config to ACP home")
        else:
            (config_dir / "config.json").write_text(
                json.dumps({"version": 1, "permissions": {"allow": ["*"]}})
            )

        self._write_mcp_configs(mcp_servers)

    def _write_mcp_configs(
        self,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write MCP server definitions into the isolated HOME.

        `devin acp` 3000.6.7+ loads MCP servers from `mcp_config.json` at
        process startup, so the isolated home must contain the active servers
        before the child is spawned.
        """
        if self._devin_home is None:
            return

        servers: dict[str, dict[str, Any]] = {}
        for server in self._normalize_mcp_servers(mcp_servers):
            name = str(server.get("name", ""))
            if not name or server.get("disabled"):
                continue
            if name in servers:
                logger.warning("Duplicate MCP server %s in active list; using last", name)
            entry: dict[str, Any] = {
                "command": server.get("command", "python"),
                "args": server.get("args", []),
            }
            if "cwd" in server:
                entry["cwd"] = server["cwd"]
            env = server.get("env", [])
            if isinstance(env, list):
                env_dict: dict[str, str] = {}
                for e in env:
                    if isinstance(e, str) and "=" in e:
                        key, value = e.split("=", 1)
                        env_dict[key] = value
                if env_dict:
                    entry["env"] = env_dict
            elif isinstance(env, dict) and env:
                entry["env"] = dict(env)
            if "instructions" in server:
                entry["instructions"] = server["instructions"]
            servers[name] = entry

        mcp_config = json.dumps({"mcpServers": servers}, indent=2)
        try:
            (self._devin_home / ".config" / "devin" / "mcp_config.json").write_text(mcp_config)
            (self._devin_home / ".codeium" / "windsurf" / "mcp_config.json").write_text(mcp_config)
        except OSError:
            logger.warning("Failed to write mcp_config.json")

    def _normalize_mcp_servers(
        self,
        mcp_servers: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Return the requested MCP servers with any `lean-ctx` entries dropped.

        `lean-ctx` has been removed from this setup because the shared daemon is
        a single point of failure and can hang `devin acp` startup. If a caller
        still passes it, strip it out and keep the other servers.

        The ACP `session/new` payload expects `env` as a map of strings, while
        the harness stores it as a list of `KEY=VALUE` strings. Convert any
        non-empty list to a dict before sending it to the ACP child.
        """
        if not mcp_servers:
            return []

        out: list[dict[str, Any]] = []

        for server in list(mcp_servers):
            name = str(server.get("name", ""))
            command = str(server.get("command", ""))
            if name == "lean-ctx" or Path(command).name == "lean-ctx":
                logger.warning(
                    "lean-ctx MCP server requested but is disabled in this setup; dropping"
                )
                continue
            server = dict(server)
            env = server.get("env")
            if isinstance(env, list):
                env_map: dict[str, str] = {}
                for entry in env:
                    if isinstance(entry, str) and "=" in entry:
                        key, value = entry.split("=", 1)
                        env_map[key] = value
                server["env"] = env_map
            out.append(server)

        return out

    def _mcp_servers_key(self, mcp_servers: list[dict[str, Any]]) -> str:
        """Return a stable comparison key for a list of MCP server definitions."""
        simplified = [
            {
                "name": str(s.get("name", "")),
                "command": str(s.get("command", "")),
                "args": list(s.get("args", [])),
                "env": list(s.get("env", [])) if isinstance(s.get("env"), list) else dict(s.get("env", {})),
            }
            for s in mcp_servers
        ]
        return json.dumps(simplified, sort_keys=True)

    def _cleanup_devin_home(self) -> None:
        """Remove the isolated HOME created for the ACP child."""
        home = self._devin_home
        self._devin_home = None
        if home is not None and home.exists():
            try:
                shutil.rmtree(home)
            except OSError:
                logger.warning("Failed to remove ACP home %s", home)

    # ---------------------------------------------------------------- public

    def create_session(
        self,
        prompt_text: str,
        *,
        cwd: str | Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        soft_timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Create a new ACP session, send the first prompt, return the result."""
        normalized_mcp_servers = self._normalize_mcp_servers(mcp_servers)
        self._ensure_started(normalized_mcp_servers)
        if cwd is not None:
            cwd = Path(cwd)
        return self._run(
            self._create_session(
                prompt_text,
                cwd=cwd,
                model=model,
                mcp_servers=normalized_mcp_servers,
                soft_timeout=soft_timeout,
                on_chunk=on_chunk,
                on_update=on_update,
            ),
            timeout=self.timeout + 30.0,
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
            timeout=self.timeout + 30.0,
        )

    def list_models(self) -> list[str]:
        """Return the model IDs advertised by the ACP server."""
        self._ensure_started()
        if self._model_options is not None:
            return self._model_options
        return self._run(self._list_models(), timeout=60.0)

    def health(self) -> bool:
        """Return True if the ACP transport is initialized and healthy.

        Avoid making a live ACP call here: the event loop may be busy with a
        long prompt or a slow ``session/new``.  Those calls have their own
        timeouts; a blocking health probe would only cause the harness
        watchdog to misfire and restart a legitimately busy service.
        """
        with self._lock:
            if not self._initialized or self._proc is None or self._proc.returncode is not None:
                return False
            if self._inflight_future is not None and not self._inflight_future.done():
                now = time.monotonic()
                if now <= self._inflight_deadline:
                    return True
            return self._transport_healthy

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
        self._ensure_started()
        return self._run(
            self._resume_session(
                session_id,
                cwd=cwd,
                model=model,
                mcp_servers=mcp_servers,
            ),
            timeout=self.timeout + 30.0,
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
            self._initialized = False
            self._transport_healthy = False
            try:
                if self._thread is not None and self._thread.is_alive():
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
                self._stop_watchdog()
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=10.0)
                self._loop = None
                self._thread = None
                self._proc = None
                self._reader_task = None
                self._stderr_task = None
                self._cleanup_devin_home()

    def _check_restart_backoff(self) -> None:
        """Raise AcpTransportError if we have restarted too many times recently."""
        now = time.monotonic()
        cutoff = now - self._restart_backoff_window
        self._restart_history = [t for t in self._restart_history if t > cutoff]
        if self._max_restarts and len(self._restart_history) >= self._max_restarts:
            raise AcpTransportError(
                "acp.restart",
                msg=(
                    f"ACP transport has been restarted {len(self._restart_history)} times "
                    f"in the last {self._restart_backoff_window}s; giving up"
                ),
            )

    def _record_restart_attempt(self) -> None:
        """Record that we are about to (re)start the ACP transport."""
        now = time.monotonic()
        cutoff = now - self._restart_backoff_window
        self._restart_history = [t for t in self._restart_history if t > cutoff]
        self._restart_history.append(now)

    def restart_transport(self) -> None:
        """Kill the ACP subprocess and start a fresh one."""
        logger.warning("Restarting ACP transport")
        self._check_restart_backoff()
        self._record_restart_attempt()
        if self.metrics is not None:
            self.metrics.inc("acp_restarts_total")
        self.close()
        self._ensure_started()

    # ---------------------------------------------------------------- internal

    def _ensure_started(
        self,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        """Start the ACP transport, writing the active MCP list first.

        If the transport is already running with a different MCP server list,
        restart it so `devin acp` picks up the new `mcp_config.json`.
        """
        target = self._normalize_mcp_servers(
            mcp_servers if mcp_servers is not None else self._mcp_servers
        )

        while True:
            with self._lock:
                if self._initialized:
                    if self._mcp_servers_key(target) == self._mcp_servers_key(self._mcp_servers):
                        if self._watchdog_thread is None or not self._watchdog_thread.is_alive():
                            self._watchdog_running = True
                            self._watchdog_thread = threading.Thread(
                                target=self._watchdog, daemon=True
                            )
                            self._watchdog_thread.start()
                        return
                    # MCP list changed; restart outside the lock.
                    needs_restart = True
                else:
                    needs_restart = False

                    self._loop = asyncio.new_event_loop()
                    self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
                    self._thread.start()

                    self._prepare_devin_home(target)

            if needs_restart:
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
                            self._prepare_devin_home(target)
                    else:
                        raise last_exc

            with self._lock:
                self._initialized = True
                self._mcp_servers = target
                self._transport_healthy = True
            return

    def _run(self, coro: Any, timeout: float | None = None) -> Any:
        """Run a coroutine on the background loop and block for the result."""
        if self._loop is None:
            raise RuntimeError("ACP client not started")
        if timeout is None:
            timeout = self.timeout

        deadline = time.monotonic() + timeout
        future: concurrent.futures.Future[Any] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        with self._lock:
            self._inflight_future = future
            self._inflight_deadline = deadline
            if self._watchdog_thread is not None and not self._watchdog_thread.is_alive():
                self._watchdog_running = True
                self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
                self._watchdog_thread.start()
        try:
            result = future.result(timeout=timeout + 5.0)
            self._transport_healthy = True
            return result
        except TimeoutError as exc:
            self._transport_healthy = False
            if self.metrics is not None:
                self.metrics.inc("acp_transport_errors_total", reason="timeout")
            raise AcpTransportError(
                getattr(coro, "__name__", "acp.run"),
                msg=f"ACP call timed out after {timeout}s",
            ) from exc
        except AcpError as exc:
            if isinstance(exc, AcpTransportError):
                self._transport_healthy = False
            raise
        except RuntimeError as exc:
            # Anything that is still a plain RuntimeError at this point is
            # unexpected; treat it as a transport failure so the caller can
            # restart cleanly instead of crashing the service child.
            self._transport_healthy = False
            raise AcpTransportError(
                getattr(coro, "__name__", "acp.run"),
                msg=str(exc),
            ) from exc
        except Exception:
            self._transport_healthy = False
            raise
        finally:
            with self._lock:
                self._inflight_future = None
                self._inflight_deadline = 0.0

    async def _start_transport(self) -> None:
        env = os.environ.copy()
        env["WINDSURF_API_KEY"] = self._api_key
        env["ACP_API_KEY"] = self._api_key

        # A standalone `devin acp` must not believe it is inside the
        # Windsurf IDE, or it will wait for the IDE to authenticate.
        env.pop("ACP_BACKEND", None)
        env.pop("WINDSURF_IDE_TYPE", None)
        env.pop("WINDSURF_EXT_HOST_PID", None)

        # Use an isolated HOME to avoid loading user/channel default MCP
        # configs that can block `devin acp` startup (e.g. `lean-ctx`).
        if self._devin_home is not None:
            env["HOME"] = str(self._devin_home)
            env["XDG_CONFIG_HOME"] = str(self._devin_home / ".config")
            env["XDG_DATA_HOME"] = os.environ.get(
                "XDG_DATA_HOME",
                str(Path.home() / ".local" / "share"),
            )
            env["XDG_CACHE_HOME"] = os.environ.get(
                "XDG_CACHE_HOME",
                str(Path.home() / ".cache"),
            )

        self._proc = await asyncio.create_subprocess_exec(
            str(self.agent_bin),
            *self.start_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        self._reader_task = asyncio.create_task(self._reader())
        self._stderr_task = asyncio.create_task(self._stderr_drain())

        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

        init = await self._call(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {
                    "name": "diploid-agent",
                    "version": "0.1.0",
                },
            },
            timeout=self._startup_timeout,
        )
        logger.info(
            "ACP transport ready: %s v%s",
            init["agentInfo"].get("title", "Devin"),
            init["agentInfo"].get("version", "?"),
        )

    async def _reader(self) -> None:
        """Read JSON-RPC lines from `devin acp` stdout and route them."""
        while True:
            if self._proc is None or self._proc.stdout is None:
                break
            try:
                line = await self._proc.stdout.readline()
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("ACP reader closed: %s", exc)
                break
            if not line:
                break
            logger.debug("ACP RECV: %s", line.decode().strip()[:200])

            with self._lock:
                self._last_stdout_at = time.monotonic()

            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                continue

            # Agent-to-client request (e.g. permission prompt).
            if "method" in msg and "id" in msg:
                await self._handle_request(msg)
                continue

            # Notification.
            if "id" not in msg:
                if msg.get("method") == "session/update":
                    self._route_update(msg)
                else:
                    logger.debug("ACP notification: %s", msg.get("method"))
                continue

            # Response to one of our calls.
            future = self._pending.pop(msg["id"], None)
            if future is not None and not future.done():
                future.set_result(msg)
                self._last_progress_at = time.monotonic()

    async def _stderr_drain(self) -> None:
        """Discard stderr so the ACP process never blocks on a full pipe."""
        if self._proc is None or self._proc.stderr is None:
            return
        while True:
            try:
                data = await self._proc.stderr.read(8192)
            except (OSError, ValueError, RuntimeError):
                break
            if not data:
                break

    def _route_update(self, msg: dict[str, Any]) -> None:
        """Route a `session/update` notification to its in-flight prompt."""
        params = msg.get("params", {})
        session_id = params.get("sessionId")
        update = params.get("update", {})
        if not session_id:
            return

        with self._lock:
            self._last_progress_at = time.monotonic()

        prompt = self._active_prompts.get(session_id)
        if prompt is None:
            # Fallback: if only one prompt is active, route to it.
            if len(self._active_prompts) == 1:
                prompt = next(iter(self._active_prompts.values()))
            else:
                logger.debug("No prompt for update session %s", session_id)
                return

        prompt.updates.append(update)
        if prompt.on_update:
            try:
                prompt.on_update(update)
            except Exception:
                logger.exception("ACP on_update failed")

        def _text_from_content(content: Any) -> list[str]:
            """Return all text blocks from an ACP content payload."""
            if isinstance(content, list):
                return [b.get("text", "") for b in content if b.get("type") == "text"]
            if content.get("type") == "text":
                return [content.get("text", "")]
            return []

        kind = update.get("sessionUpdate")
        if kind in ("agent_message", "agent_message_chunk"):
            for text in _text_from_content(update.get("content", {})):
                if text:
                    prompt.chunks.append(text)
                    if prompt.on_chunk:
                        try:
                            prompt.on_chunk(text)
                        except Exception:
                            logger.exception("ACP on_chunk failed")

    async def _handle_request(self, msg: dict[str, Any]) -> None:
        method = msg["method"]
        req_id = msg["id"]
        params = msg.get("params", {})

        if method == "session/request_permission":
            options = params.get("options", [])
            option_id = options[0]["optionId"] if options else "allow"
            await self._respond(
                req_id,
                {"outcome": {"outcome": "selected", "optionId": option_id}},
            )
        else:
            await self._respond(
                req_id,
                {"error": {"code": -32601, "message": f"Method not found: {method}"}},
            )

    async def _respond(self, req_id: int, result: Any) -> None:
        await self._send(
            {"jsonrpc": "2.0", "id": req_id, "result": result},
            timeout=self._control_timeout,
        )

    async def _call(self, method: str, params: Any, timeout: float | None = None) -> Any:
        self._next_id += 1
        msg_id = self._next_id
        msg = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params,
        }
        future = self._loop.create_future()
        self._pending[msg_id] = future
        call_timeout = timeout or self._control_timeout
        with self._lock:
            self._last_request_at = time.monotonic()
            self._last_control_call_deadline = time.monotonic() + call_timeout
        try:
            await self._send(msg, timeout=timeout)
            with self._lock:
                # _send succeeded; reset the per-call deadline for the response wait.
                self._last_control_call_deadline = time.monotonic() + call_timeout
            resp = await asyncio.wait_for(future, timeout=call_timeout)
        except TimeoutError:
            self._pending.pop(msg_id, None)
            raise
        finally:
            with self._lock:
                self._last_control_call_deadline = 0.0
        if "error" in resp:
            raise _acp_error_from_response(method, resp["error"])
        return resp.get("result")

    async def _send(self, msg: dict[str, Any], timeout: float | None = None) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise AcpTransportError("acp.send", msg="ACP process not running")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode()
        logger.debug("ACP SEND: %s", data.decode().strip()[:200])
        self._proc.stdin.write(data)
        try:
            await asyncio.wait_for(
                self._proc.stdin.drain(),
                timeout=timeout or self._control_timeout,
            )
            with self._lock:
                self._last_request_at = time.monotonic()
        except TimeoutError:
            logger.warning(
                "ACP send timed out after %ss",
                timeout or self._control_timeout,
            )
            raise

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
        return "method not found" in msg or "not found" in msg and "session/resume" in str(exc.method or "").lower()

    async def _send_cancel_notification(self, session_id: str) -> None:
        """Send a fire-and-forget `session/cancel` notification."""
        await self._send(
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
        """
        use_cwd = str(cwd) if cwd else os.getcwd()
        use_model = _normalize_model(model or self.model)
        normalized_mcp = self._normalize_mcp_servers(mcp_servers)
        resume_params: dict[str, Any] = {"sessionId": session_id, "cwd": use_cwd}
        if normalized_mcp:
            resume_params["mcpServers"] = normalized_mcp
        load_params: dict[str, Any] = {
            "sessionId": session_id,
            "cwd": use_cwd,
            "mcpServers": normalized_mcp,
        }

        try:
            try:
                await self._call(
                    "session/resume",
                    resume_params,
                    timeout=self._control_timeout,
                )
            except AcpError as exc:
                if self._is_method_not_found(exc):
                    logger.debug(
                        "session/resume not supported; trying session/load for %s", session_id
                    )
                    await self._call(
                        "session/load",
                        load_params,
                        timeout=self._control_timeout,
                    )
                else:
                    raise
            await self._apply_session_config(session_id, use_model)
        except (AcpError, TimeoutError):
            if self.metrics is not None:
                self.metrics.inc("acp_resume_failures_total")
            raise

        if self.metrics is not None:
            self.metrics.inc("acp_resumes_total")
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
        normalized_mcp = self._normalize_mcp_servers(mcp_servers)
        try:
            await self._call(
                "session/load",
                {
                    "sessionId": session_id,
                    "cwd": use_cwd,
                    "mcpServers": normalized_mcp,
                },
                timeout=self._control_timeout,
            )
            await self._apply_session_config(session_id, use_model)
        except (AcpError, TimeoutError):
            if self.metrics is not None:
                self.metrics.inc("acp_resume_failures_total")
            raise

        if self.metrics is not None:
            self.metrics.inc("acp_resumes_total")
        return session_id

    async def _apply_session_config(self, session_id: str, use_model: str) -> None:
        """Set mode and model on a freshly created or resumed session."""
        await self._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "mode", "value": self.acp_mode},
        )
        await self._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "model", "value": use_model},
        )
        self._session_models[session_id] = use_model

    async def _create_session(
        self,
        prompt_text: str,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        soft_timeout: float | None = None,
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
        # Cap session/new so a hung child restart (e.g. slow MCP server init)
        # does not hold the harness lock for multiple minutes. The watchdog
        # stall threshold is also bounded by _watchdog_timeout, so align the
        # call timeout with that ceiling (at least 60s to allow normal init).
        session_new_timeout = max(60.0, min(self._control_timeout, self._watchdog_timeout))
        session = await self._call(
            "session/new",
            {"cwd": use_cwd, "mcpServers": []},
            timeout=session_new_timeout,
        )
        session_id = session["sessionId"]

        if self._model_options is None:
            self._model_options = self._extract_model_options(session)

        # Honor the requested mode and model for this session.
        await self._apply_session_config(session_id, use_model)

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
        # destabilize the ACP child.
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
            await self._send(
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
                remaining = max(0.0, self.timeout - elapsed)
                try:
                    raw = await asyncio.wait_for(prompt.future, timeout=min(5.0, remaining))
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

    def _kill_process_group(self, proc: asyncio.subprocess.Process) -> None:
        """Kill the child and any spawned descendants."""
        try:
            proc.kill()
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            # Already gone or process group no longer valid.
            pass

    async def _close_transport(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._kill_process_group(self._proc)

    def _stop_watchdog(self) -> None:
        """Signal the watchdog thread to stop and wait briefly."""
        with self._lock:
            self._watchdog_running = False
        if self._watchdog_thread is not None:
            if self._watchdog_thread.is_alive():
                self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None

    def _watchdog(self) -> None:
        """Monitor ACP transport I/O and kill stuck children."""
        while True:
            with self._lock:
                if not self._watchdog_running:
                    break
                interval = self._watchdog_interval
            time.sleep(interval)
            try:
                self._check_watchdog()
            except Exception:
                logger.exception("ACP watchdog check failed")

    def _check_watchdog(self) -> None:
        """Detect unresponsive ACP transport and trigger recovery."""
        with self._lock:
            if not self._watchdog_running:
                return
            if self._inflight_future is None or self._inflight_future.done():
                return

            # If the child process has already exited, the transport is dead and
            # recovery should start immediately.
            if self._proc is not None and self._proc.returncode is not None:
                logger.warning(
                    "ACP process %s exited with code %s; watchdog recovering",
                    self._proc.pid,
                    self._proc.returncode,
                )
                self._stall_recovery()
                return

            now = time.monotonic()
            deadline = self._inflight_deadline
            last_request = self._last_request_at
            call_deadline = self._last_control_call_deadline
            active_prompts = dict(self._active_prompts)
            has_prompt = bool(active_prompts)
            has_pending = bool(self._pending)

        if now > deadline:
            logger.warning("ACP call exceeded its deadline; watchdog recovering")
            self._stall_recovery()
            return

        if has_prompt:
            # Kill a prompt only if it has exceeded its own soft/hard timeout plus
            # a short grace. Long, legitimately silent prompts are allowed to run
            # up to that deadline; the watchdog is not a "no output" timer.
            kill_at = float("inf")
            for prompt in active_prompts.values():
                limit = (
                    prompt.soft_timeout
                    if prompt.soft_timeout is not None and prompt.soft_timeout > 0
                    else self.timeout
                )
                prompt_kill_at = prompt.started_at + max(self._watchdog_timeout, limit + 30.0)
                kill_at = min(kill_at, prompt_kill_at)
            if now > kill_at:
                logger.warning("ACP prompt exceeded its soft/hard timeout; watchdog recovering")
                self._stall_recovery()
                return

            # Transport-death fallback: if the child has produced no stdout at all
            # for an extended period while a prompt is in flight, the reader is
            # likely dead even though the prompt's own timeout has not been reached.
            dead_timeout = self._control_timeout + 30.0
            if now - self._last_stdout_at > dead_timeout:
                logger.warning(
                    "No ACP stdout for %.1fs; watchdog recovering",
                    now - self._last_stdout_at,
                )
                self._stall_recovery()
                return

        # The `_pending` map also holds the future for an in-flight prompt, so
        # only use the request timing for non-prompt control calls. Prompts are
        # allowed to run for up to `self.timeout` as long as they keep producing
        # progress (chunks/updates).
        if has_pending and not has_prompt:
            if call_deadline and now > call_deadline:
                logger.warning(
                    "ACP control call produced no response for %.1fs; watchdog recovering",
                    now - last_request,
                )
                self._stall_recovery()
                return
            if not call_deadline and now - last_request > self._watchdog_timeout:
                logger.warning(
                    "ACP control call produced no response for %ss; watchdog recovering",
                    self._watchdog_timeout,
                )
                self._stall_recovery()
                return

    def _stall_recovery(self) -> None:
        """Kill the ACP child and unblock the in-flight caller."""
        with self._lock:
            if self.metrics is not None:
                self.metrics.inc("acp_watchdog_fired_total")

            # Unblock the synchronous caller waiting in _run().
            inflight = self._inflight_future
            if inflight is not None and not inflight.done():
                try:
                    inflight.set_exception(TimeoutError("ACP transport watchdog detected a stall"))
                except Exception:
                    logger.exception("Failed to interrupt in-flight ACP future")

            # Attempt to cancel any in-flight prompt server-side.
            for prompt in list(self._active_prompts.values()):
                prompt.cancelled = True
                prompt.timed_out = True
                if not prompt.cancel_done.done():
                    prompt.cancel_done.set_result(None)

            def _cancel_all() -> None:
                for prompt in list(self._active_prompts.values()):
                    if self._loop is not None:
                        self._loop.create_task(self._send_cancel_notification(prompt.session_id))

            if self._loop is not None:
                try:
                    self._loop.call_soon_threadsafe(_cancel_all)
                except Exception:
                    logger.exception("Failed to schedule ACP cancel notifications")

            # Kill the process and stop the loop.
            self._transport_healthy = False
            self._initialized = False
            if self._proc is not None and self._proc.returncode is None:
                try:
                    logger.warning("Killing unresponsive ACP process %s", self._proc.pid)
                    self._kill_process_group(self._proc)
                    if self.metrics is not None:
                        self.metrics.inc("acp_transport_killed_total")
                except Exception:
                    logger.exception("Failed to kill ACP process during watchdog recovery")
            if self._loop is not None and self._loop.is_running():
                try:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                except Exception:
                    logger.exception("Failed to stop ACP event loop during watchdog recovery")
