"""Low-level ACP JSON-RPC stdio transport and process lifecycle."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.acp_client.errors import (
    AcpError,
    AcpTransportError,
    _acp_error_from_response,
)

logger = logging.getLogger(__name__)


class AcpTransport:
    """JSON-RPC stdio transport, background event loop, and process lifecycle.

    ``AcpTransport`` owns the subprocess and the JSON-RPC reader/writer.
    It does not manage session/prompt state directly; that lives on the
    ``AcpClient`` instance passed in as ``client``.
    """

    def __init__(
        self,
        client: Any,
        on_request: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._client = client
        self._on_request = on_request

        # Background loop state.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None

        # I/O and progress tracking.
        self._last_stdout_at: float = 0.0
        self._last_progress_at: float = 0.0
        self._last_request_at: float = 0.0
        self._last_control_call_deadline: float = 0.0

        # In-flight request tracking.
        self._inflight_future: concurrent.futures.Future[Any] | None = None
        self._inflight_deadline: float = 0.0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}

        # Transport health.
        self._initialized = False
        self._transport_healthy = False

    # ------------------------------------------------------------------ public

    def healthy(self) -> bool:
        """Return True if the ACP transport is initialized and healthy."""
        if not self._initialized or self._proc is None or self._proc.returncode is not None:
            return False
        if self._inflight_future is not None and not self._inflight_future.done():
            now = time.monotonic()
            if now <= self._inflight_deadline:
                return True
        return self._transport_healthy

    async def start(self, mcp_servers: list[dict[str, Any]] | None = None) -> None:
        """Prepare the sandbox and start the ACP transport process."""
        if mcp_servers is not None:
            self._client._sandbox.prepare(mcp_servers)
        await self._start_transport()

    async def restart(self, mcp_servers: list[dict[str, Any]] | None = None) -> None:
        """Close and re-open the ACP transport."""
        logger.warning("Restarting ACP transport")
        self._check_restart_backoff()
        self._record_restart_attempt()
        if self._client.metrics is not None:
            self._client.metrics.inc("acp_restarts_total")
        self._unblock_inflight("ACP transport restarted")
        await self.close()
        await self.start(mcp_servers)

    async def close(self) -> None:
        """Terminate the ACP subprocess and cancel I/O tasks."""
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._kill_process_group(self._proc)

        # Cancel reader and stderr drain so the loop does not keep them alive.
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    logger.debug("ACP task %s ended with %s", task.get_name(), exc)

        # Any request futures that have not been resolved by the reader should
        # be aborted now, including in-flight _run() callers.
        self._unblock_inflight("ACP transport closed")

    async def call(self, method: str, params: Any, timeout: float | None = None) -> Any:
        """Send a JSON-RPC request and return the result."""
        if self._loop is None:
            raise RuntimeError("ACP transport not started")

        self._client._next_id += 1
        msg_id = self._client._next_id
        msg = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params,
        }
        future = self._loop.create_future()
        self._pending[msg_id] = future
        call_timeout = timeout or self._client._control_timeout
        with self._client._lock:
            self._last_request_at = time.monotonic()
            self._last_control_call_deadline = time.monotonic() + call_timeout
        try:
            await self._send(msg, timeout=timeout)
            with self._client._lock:
                # _send succeeded; reset the per-call deadline for the response wait.
                self._last_control_call_deadline = time.monotonic() + call_timeout
            resp = await asyncio.wait_for(future, timeout=call_timeout)
        except TimeoutError:
            self._pending.pop(msg_id, None)
            raise
        finally:
            with self._client._lock:
                self._last_control_call_deadline = 0.0
        if "error" in resp:
            raise _acp_error_from_response(method, resp["error"])
        return resp.get("result")

    def run(self, coro: Any, timeout: float | None = None) -> Any:
        """Run a coroutine on the background loop and block for the result."""
        if self._loop is None:
            raise RuntimeError("ACP transport not started")
        if timeout is None:
            timeout = self._client.timeout

        deadline = time.monotonic() + timeout if timeout is not None else float("inf")
        result_timeout = timeout + 5.0 if timeout is not None else None
        future: concurrent.futures.Future[Any] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        with self._client._lock:
            self._inflight_future = future
            self._inflight_deadline = deadline
            self._client._watchdog.start()
        try:
            result = future.result(timeout=result_timeout)
            self._transport_healthy = True
            return result
        except TimeoutError as exc:
            self._transport_healthy = False
            if self._client.metrics is not None:
                self._client.metrics.inc("acp_transport_errors_total", reason="timeout")
            raise AcpTransportError(
                getattr(coro, "__name__", "acp.run"),
                msg=(
                    f"ACP call timed out after {timeout}s"
                    if timeout is not None
                    else "ACP call timed out"
                ),
            ) from exc
        except AcpError as exc:
            if isinstance(exc, AcpTransportError):
                self._transport_healthy = False
            raise
        except RuntimeError as exc:
            # Anything that is still a plain RuntimeError at this point is
            # unexpected; treat it as a transport failure so the caller can
            # restart cleanly instead of crashing the service subprocess.
            self._transport_healthy = False
            raise AcpTransportError(
                getattr(coro, "__name__", "acp.run"),
                msg=str(exc),
            ) from exc
        except (OSError, ConnectionError) as exc:
            # Broken pipes and closed transports surface as OS-level errors.
            # Convert them to transport errors so the harness treats them as
            # recoverable rather than crashing the service subprocess.
            self._transport_healthy = False
            raise AcpTransportError(
                getattr(coro, "__name__", "acp.run"),
                msg=f"ACP transport IO error: {exc}",
            ) from exc
        except Exception:
            self._transport_healthy = False
            raise
        finally:
            with self._client._lock:
                self._inflight_future = None
                self._inflight_deadline = 0.0
            # Cancel the underlying asyncio task if the caller timed out or
            # raised, so it is not left pending and garbage-collected when the
            # event loop shuts down.
            try:
                if future is not None and not future.done():
                    future.cancel()
            except (RuntimeError, OSError, ValueError) as exc:
                logger.debug("Could not cancel pending ACP task: %s", exc)

    # ---------------------------------------------------------------- internal

    def _is_transport_healthy(self) -> bool:
        """Check whether the running ACP subprocess and event loop are still usable."""
        if not self._initialized:
            return False
        if self._loop is None or self._loop.is_closed() or not self._loop.is_running():
            return False
        if self._proc is None:
            return False
        return self._proc.returncode is None

    def _cleanup_stale_transport(self) -> None:
        """Kill a dead subprocess and stop its event loop so a fresh one can start."""
        if self._proc is not None:
            if self._proc.returncode is None:
                try:
                    self._kill_process_group(self._proc)
                except OSError:
                    logger.warning("Failed to kill stale ACP process %s", self._proc.pid)
            self._proc = None
        if self._loop is not None and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                logger.warning("Failed to stop stale ACP event loop")
        self._client._watchdog.stop()

    async def _start_transport(self) -> None:
        env = os.environ.copy()
        env["WINDSURF_API_KEY"] = self._client._api_key
        env["ACP_API_KEY"] = self._client._api_key

        # A standalone `devin acp` must not believe it is inside the
        # Windsurf IDE, or it will wait for the IDE to authenticate.
        env.pop("ACP_BACKEND", None)
        env.pop("WINDSURF_IDE_TYPE", None)
        env.pop("WINDSURF_EXT_HOST_PID", None)

        # Use an isolated HOME to avoid loading user/channel default MCP
        # configs that can block `devin acp` startup (e.g. `lean-ctx`).
        if self._client._sandbox.devin_home is not None:
            env["HOME"] = str(self._client._sandbox.devin_home)
            env["XDG_CONFIG_HOME"] = str(self._client._sandbox.devin_home / ".config")
            env["XDG_DATA_HOME"] = os.environ.get(
                "XDG_DATA_HOME",
                str(Path.home() / ".local" / "share"),
            )
            env["XDG_CACHE_HOME"] = os.environ.get(
                "XDG_CACHE_HOME",
                str(Path.home() / ".cache"),
            )
            # Isolate the subprocess from the user's systemd/D-Bus session so it cannot
            # run raw `systemctl --user restart ...` directly. Restarts go through
            # the fake binaries in .local/bin and the harness control socket.
            env["XDG_RUNTIME_DIR"] = str(self._client._sandbox.devin_home / ".run")
            env.pop("DBUS_SESSION_BUS_ADDRESS", None)
            env.pop("DBUS_SYSTEM_BUS_ADDRESS", None)
            env.update(self._client._control.env())

            # Prepend the fake binary directory to PATH.
            fake_bin = str(self._client._sandbox.devin_home / ".local" / "bin")
            env["PATH"] = fake_bin + os.pathsep + env.get("PATH", "")

        self._proc = await asyncio.create_subprocess_exec(
            str(self._client.agent_bin),
            *self._client.start_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        self._reader_task = asyncio.create_task(self._reader())
        self._stderr_task = asyncio.create_task(self._stderr_drain())

        self._client._watchdog.start()

        init = await self.call(
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
            timeout=self._client._startup_timeout,
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

            with self._client._lock:
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

        with self._client._lock:
            self._last_progress_at = time.monotonic()

        prompt = self._client._active_prompts.get(session_id)
        if prompt is None:
            # Fallback: if only one prompt is active, route to it.
            if len(self._client._active_prompts) == 1:
                prompt = next(iter(self._client._active_prompts.values()))
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
            if isinstance(content, dict) and content.get("type") == "text":
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
            result = {"outcome": {"outcome": "selected", "optionId": option_id}}
            if self._on_request is not None:
                try:
                    result = self._on_request(msg)
                    if not isinstance(result, dict):
                        result = {"outcome": {"outcome": "selected", "optionId": option_id}}
                except Exception:
                    logger.exception("ACP on_request failed")
            await self._respond(req_id, result)
        else:
            await self._respond(
                req_id,
                {"error": {"code": -32601, "message": f"Method not found: {method}"}},
            )

    async def _respond(self, req_id: int, result: Any) -> None:
        await self._send(
            {"jsonrpc": "2.0", "id": req_id, "result": result},
            timeout=self._client._control_timeout,
        )

    async def _send(self, msg: dict[str, Any], timeout: float | None = None) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise AcpTransportError("acp.send", msg="ACP process not running")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode()
        logger.debug("ACP SEND: %s", data.decode().strip()[:200])
        self._proc.stdin.write(data)
        try:
            await asyncio.wait_for(
                self._proc.stdin.drain(),
                timeout=timeout or self._client._control_timeout,
            )
            with self._client._lock:
                self._last_request_at = time.monotonic()
        except TimeoutError:
            logger.warning(
                "ACP send timed out after %ss",
                timeout or self._client._control_timeout,
            )
            raise

    def _unblock_inflight(self, reason: str) -> None:
        """Set an exception on the in-flight _run future and cancel active prompts.

        This must be called before killing the ACP process so the synchronous
        caller blocked in _run() returns instead of hanging.
        """
        with self._client._lock:
            inflight = self._inflight_future
            if inflight is not None and not inflight.done():
                try:
                    inflight.set_exception(TimeoutError(reason))
                except Exception:
                    logger.exception("Failed to interrupt in-flight ACP future")

            # Abort any pending request futures so _send() does not hang.
            exc = AcpTransportError("acp.transport", msg=reason)
            for req_id, future in list(self._pending.items()):
                if not future.done():
                    try:
                        future.set_exception(exc)
                    except Exception:
                        logger.exception("Failed to unblock pending ACP request %s", req_id)
            self._pending.clear()

            for prompt in list(self._client._active_prompts.values()):
                prompt.cancelled = True
                prompt.timed_out = True
                if not prompt.cancel_done.done():
                    prompt.cancel_done.set_result(None)

            def _cancel_all() -> None:
                for prompt in list(self._client._active_prompts.values()):
                    if self._client._loop is not None:
                        self._client._loop.create_task(
                            self._client._send_cancel_notification(prompt.session_id)
                        )

            if self._client._loop is not None:
                try:
                    self._client._loop.call_soon_threadsafe(_cancel_all)
                except Exception:
                    logger.exception("Failed to schedule ACP cancel notifications")

    def _check_restart_backoff(self) -> None:
        """Raise AcpTransportError if we have restarted too many times recently."""
        self._client._check_restart_backoff()

    def _record_restart_attempt(self) -> None:
        """Record that we are about to (re)start the ACP transport."""
        self._client._record_restart_attempt()

    def _kill_process_group(self, proc: asyncio.subprocess.Process) -> None:
        """Kill the subprocess and any spawned descendants."""
        try:
            proc.kill()
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            # Already gone or process group no longer valid.
            pass
