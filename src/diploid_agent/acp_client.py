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
    chunks: list[str] = dataclasses.field(default_factory=list)
    updates: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    on_chunk: Callable[[str], None] | None = None
    on_update: Callable[[dict[str, Any]], None] | None = None
    cancelled: bool = False
    timed_out: bool = False


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
        watchdog_interval: float = 10.0,
        watchdog_timeout: float = 120.0,
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
            api_key or os.environ.get("WINDSURF_API_KEY") or os.environ.get("ACP_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "No api_key provided and no WINDSURF_API_KEY or ACP_API_KEY in environment."
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
        self._last_request_at: float = 0.0
        self._inflight_future: concurrent.futures.Future[Any] | None = None
        self._inflight_deadline: float = 0.0
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._active_prompts: dict[str, _Prompt] = {}
        self._pending_cancels: set[str] = set()
        self._lock = threading.RLock()
        self._initialized = False
        self._transport_healthy = False
        self._control_timeout = 30.0
        self._model_options: list[str] | None = None

        atexit.register(self.close)

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
        self._ensure_started()
        if cwd is not None:
            cwd = Path(cwd)
        return self._run(
            self._create_session(
                prompt_text,
                cwd=cwd,
                model=model,
                mcp_servers=mcp_servers,
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
        """Return True if the ACP transport is initialized and healthy."""
        if not self._initialized or not self._transport_healthy:
            return False
        if self._model_options is not None:
            return True
        try:
            self._run(self._list_models(), timeout=10.0)
            return True
        except Exception:  # noqa: BLE001
            return False

    def session_alive(self, session_id: str) -> bool:
        """Probe whether an ACP session id is still valid."""
        self._ensure_started()
        return self._run(self._session_alive(session_id), timeout=30.0)

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
                self._run(self._close_transport(), timeout=10.0)
            except (RuntimeError, TimeoutError) as exc:
                logger.warning("ACP close transport failed: %s", exc)
                if self._proc is not None and self._proc.returncode is None:
                    try:
                        self._proc.kill()
                    except Exception:
                        logger.exception("Failed to kill ACP process during close")
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._stop_watchdog()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)
            self._loop = None
            self._thread = None
            self._proc = None
            self._reader_task = None

    def restart_transport(self) -> None:
        """Kill the ACP subprocess and start a fresh one."""
        logger.warning("Restarting ACP transport")
        if self.metrics is not None:
            self.metrics.inc("acp_restarts_total")
        self.close()
        self._ensure_started()

    # ---------------------------------------------------------------- internal

    def _ensure_started(self) -> None:
        with self._lock:
            if self._initialized:
                return

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()

            self._run(self._start_transport(), timeout=30.0)
            self._initialized = True
            self._transport_healthy = True

    def _run(self, coro: Any, timeout: float | None = None) -> Any:
        """Run a coroutine on the background loop and block for the result."""
        if self._loop is None:
            raise RuntimeError("ACP client not started")
        if timeout is None:
            timeout = self.timeout

        deadline = time.monotonic() + timeout
        future: concurrent.futures.Future[Any] = asyncio.run_coroutine_threadsafe(
            coro, self._loop
        )
        with self._lock:
            self._inflight_future = future
            self._inflight_deadline = deadline
        try:
            result = future.result(timeout=timeout + 5.0)
            self._transport_healthy = True
            return result
        except TimeoutError:
            self._transport_healthy = False
            if self.metrics is not None:
                self.metrics.inc("acp_transport_errors_total", reason="timeout")
            raise
        except RuntimeError as exc:
            self._transport_healthy = False
            if self._is_stale_session_error(exc):
                raise
            raise TimeoutError(f"ACP transport failed: {exc}") from exc
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

        self._proc = await asyncio.create_subprocess_exec(
            str(self.agent_bin),
            *self.start_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )

        self._reader_task = asyncio.create_task(self._reader())

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

    def _route_update(self, msg: dict[str, Any]) -> None:
        """Route a `session/update` notification to its in-flight prompt."""
        params = msg.get("params", {})
        session_id = params.get("sessionId")
        update = params.get("update", {})
        if not session_id:
            return

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
        try:
            await self._send(msg, timeout=timeout)
            resp = await asyncio.wait_for(future, timeout=timeout or self._control_timeout)
        except TimeoutError:
            self._pending.pop(msg_id, None)
            raise
        if "error" in resp:
            raise RuntimeError(f"ACP {method} failed: {resp['error']}")
        return resp.get("result")

    async def _send(self, msg: dict[str, Any], timeout: float | None = None) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("ACP process not running")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode()
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
    def _is_stale_session_error(exc: RuntimeError) -> bool:
        """Return True if an ACP error indicates the session id is no longer valid."""
        msg = str(exc).lower()
        return "session" in msg and (
            "not found" in msg
            or "invalid" in msg
            or "expired" in msg
            or "stale" in msg
            or "empty reply" in msg
        )

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
            )
            return True
        except RuntimeError as exc:
            if self._is_stale_session_error(exc):
                return False
            raise

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

        session = await self._call(
            "session/new",
            {"cwd": use_cwd, "mcpServers": mcp_servers or []},
        )
        session_id = session["sessionId"]

        if self._model_options is None:
            self._model_options = self._extract_model_options(session)

        # Honor the requested mode and model for this session.
        await self._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "mode", "value": self.acp_mode},
        )
        await self._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "model", "value": use_model},
        )

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
        # Always set the session model on follow-up to keep it in sync with the
        # harness record (ACP ignores a no-op switch).
        await self._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "model", "value": use_model},
        )

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
            on_chunk=on_chunk,
            on_update=on_update,
        )
        self._pending[prompt_id] = prompt.future
        self._active_prompts[session_id] = prompt

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
                raise RuntimeError(f"session/prompt failed: {raw['error']}")
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
                raise RuntimeError(f"Session {session_id} returned an empty reply (stale session)")

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

    async def _close_transport(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()

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
            self._check_watchdog()

    def _check_watchdog(self) -> None:
        """Detect unresponsive ACP transport and trigger recovery."""
        with self._lock:
            if not self._watchdog_running:
                return
            if self._inflight_future is None or self._inflight_future.done():
                return

            now = time.monotonic()
            deadline = self._inflight_deadline
            last_stdout = self._last_stdout_at
            last_request = self._last_request_at
            has_prompt = bool(self._active_prompts)
            has_pending = bool(self._pending)
            timeout = self._watchdog_timeout

        if now > deadline:
            logger.warning("ACP call exceeded its deadline; watchdog recovering")
            self._stall_recovery()
            return

        if has_prompt and now - last_stdout > timeout:
            logger.warning(
                "ACP prompt produced no output for %ss; watchdog recovering", timeout
            )
            self._stall_recovery()
            return

        if has_pending and now - last_request > self._control_timeout:
            logger.warning(
                "ACP control call produced no response for %ss; watchdog recovering",
                self._control_timeout,
            )
            self._stall_recovery()

    def _stall_recovery(self) -> None:
        """Kill the ACP child and unblock the in-flight caller."""
        with self._lock:
            if self.metrics is not None:
                self.metrics.inc("acp_watchdog_fired_total")

            # Unblock the synchronous caller waiting in _run().
            inflight = self._inflight_future
            if inflight is not None and not inflight.done():
                try:
                    inflight.set_exception(
                        TimeoutError("ACP transport watchdog detected a stall")
                    )
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
                        self._loop.create_task(
                            self._send_cancel_notification(prompt.session_id)
                        )

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
                    self._proc.kill()
                    if self.metrics is not None:
                        self.metrics.inc("acp_transport_killed_total")
                except Exception:
                    logger.exception("Failed to kill ACP process during watchdog recovery")
            if self._loop is not None and self._loop.is_running():
                try:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                except Exception:
                    logger.exception("Failed to stop ACP event loop during watchdog recovery")
