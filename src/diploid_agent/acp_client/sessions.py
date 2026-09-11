"""Session-level ACP JSON-RPC operations.

``AcpSessionOps`` owns the async calls that sit above the raw transport:
``session/new``, ``session/resume``/``session/load``, ``session/prompt``,
``session/cancel``, and the per-session mode/model re-apply.  Shared state
(``_active_prompts``, ``_pending``, ``_mcp_servers``, ``_session_models``,
metrics, lifecycle log) stays on the owning ``AcpClient``; this collaborator
reaches it through ``self._client``, matching the ``AcpTransport`` /
``PromptWatchdog`` pattern.

Cross-calls deliberately go through the ``AcpClient._*`` delegates (e.g.
``self._client._call``, ``self._client._prompt``) rather than calling sibling
methods directly, so the historical client-level seams -- including methods
monkeypatched by tests -- keep intercepting.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.acp_client.errors import (
    AcpError,
    AcpSessionStaleError,
    _acp_error_from_response,
)
from diploid_agent.acp_client.types import AcpPromptResult, _Prompt
from diploid_agent.acp_client.utils import _normalize_model

logger = logging.getLogger(__name__)


class AcpSessionOps:
    """Session-level ACP calls bound to an ``AcpClient`` instance."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def _send_cancel_notification(self, session_id: str) -> None:
        """Send a fire-and-forget `session/cancel` notification."""
        await self._client._transport._send(
            {
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": session_id},
            },
            timeout=self._client._control_timeout,
        )

    async def _session_alive(self, session_id: str) -> bool:
        """Try a cheap config update to see if the session still exists."""
        try:
            await self._client._call(
                "session/set_config_option",
                {
                    "sessionId": session_id,
                    "configId": "mode",
                    "value": self._client.acp_mode,
                },
                timeout=10.0,
            )
            return True
        except (AcpError, RuntimeError) as exc:
            if self._client._is_stale_session_error(exc):
                return False
            raise

    def _resume_jitter(self, attempt: int) -> float:
        """Return an exponential-backoff delay with a small amount of jitter."""
        base = self._client.acp_resume_retry_base_seconds
        cap = self._client.acp_resume_retry_max_seconds
        delay = min(base * (2**attempt), cap) + random.uniform(0, 0.1)
        return min(delay, cap)

    async def _call_with_resume_retry(
        self,
        method: str,
        params: dict[str, Any],
        call_timeout: float,
        budget: Callable[[], float] | None = None,
    ) -> Any:
        """Call an ACP resume method, retrying transient errors with jitter.

        Does not retry a JSON-RPC "method not found" error; that is the
        caller's signal to try a different method.  ``budget`` is an optional
        callable returning the remaining seconds of the overall resume
        budget; each attempt is capped by it.
        """
        last_exc: Exception | None = None
        max_attempts = self._client.acp_resume_max_retries + 1
        for attempt in range(max_attempts):
            attempt_timeout = min(call_timeout, budget()) if budget is not None else call_timeout
            try:
                return await self._client._call(method, params, timeout=attempt_timeout)
            except AcpError as exc:
                if self._client._is_method_not_found(exc):
                    raise
                last_exc = exc
            except TimeoutError as exc:
                last_exc = exc
            if attempt < max_attempts - 1:
                await asyncio.sleep(self._client._resume_jitter(attempt))
        assert last_exc is not None
        raise last_exc

    async def _resume_session(
        self,
        session_id: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
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
        use_model = _normalize_model(model or self._client.model)
        # Keep the client-side MCP list in sync so future transport restarts
        # write the correct mcp_config.json.
        if mcp_servers is not None:
            self._client._mcp_servers = self._client._sandbox.normalize_mcp_servers(mcp_servers)
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

        if self._client._lifecycle_log is not None:
            self._client._lifecycle_log.write(
                "session.resume.attempt",
                session_id=session_id,
                model=use_model,
                detail={"cwd": str(use_cwd)},
            )

        resume_method = "resume"
        start = time.perf_counter()
        deadline = time.monotonic() + timeout if timeout is not None and timeout > 0 else None

        def _remaining() -> float:
            if deadline is None:
                return self._client._control.call_timeout()
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"ACP resume budget of {timeout}s exhausted for {session_id}")
            return left

        # Per-phase timing: when a resume eats its budget, the lifecycle log
        # should say which phase consumed it (resume/load call vs the mode and
        # model re-apply) rather than a single opaque duration.
        phase = "resume"
        resume_ms: float | None = None
        config_ms: float | None = None
        try:
            call_timeout = self._client._control.call_timeout()
            try:
                phase_start = time.perf_counter()
                await self._client._call_with_resume_retry(
                    "session/resume",
                    resume_params,
                    call_timeout,
                    budget=_remaining,
                )
                resume_ms = round((time.perf_counter() - phase_start) * 1000, 2)
            except AcpError as exc:
                if self._client._is_method_not_found(exc):
                    logger.debug(
                        "session/resume not supported; trying session/load for %s", session_id
                    )
                    resume_method = "load"
                    phase = "load"
                    phase_start = time.perf_counter()
                    await self._client._call_with_resume_retry(
                        "session/load",
                        load_params,
                        call_timeout,
                        budget=_remaining,
                    )
                    resume_ms = round((time.perf_counter() - phase_start) * 1000, 2)
                else:
                    raise
            phase = "config"
            phase_start = time.perf_counter()
            await self._client._apply_session_config(
                session_id, use_model, timeout=min(call_timeout, _remaining())
            )
            config_ms = round((time.perf_counter() - phase_start) * 1000, 2)
        except (AcpError, TimeoutError) as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if self._client.metrics is not None:
                self._client.metrics.inc("acp_resume_total", result="failure", method=resume_method)
                self._client.metrics.set("acp_resume_latency_ms", duration_ms, result="failure")
            if self._client._lifecycle_log is not None:
                self._client._lifecycle_log.write(
                    "session.resume.failure",
                    session_id=session_id,
                    model=use_model,
                    detail={
                        "cwd": str(use_cwd),
                        "error": str(exc),
                        "phase": phase,
                        "duration_ms": duration_ms,
                        "resume_ms": resume_ms,
                        "config_ms": config_ms,
                    },
                )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if self._client.metrics is not None:
            self._client.metrics.inc("acp_resume_total", result="success", method=resume_method)
            self._client.metrics.set("acp_resume_latency_ms", duration_ms, result="success")
        if self._client._lifecycle_log is not None:
            self._client._lifecycle_log.write(
                "session.resume.success",
                session_id=session_id,
                model=use_model,
                detail={
                    "cwd": str(use_cwd),
                    "method": resume_method,
                    "duration_ms": duration_ms,
                    "resume_ms": resume_ms,
                    "config_ms": config_ms,
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
        use_model = _normalize_model(model or self._client.model)
        # Keep the client-side MCP list in sync so future transport restarts
        # write the correct mcp_config.json.
        if mcp_servers is not None:
            self._client._mcp_servers = self._client._sandbox.normalize_mcp_servers(mcp_servers)
        if self._client._lifecycle_log is not None:
            self._client._lifecycle_log.write(
                "session.load.attempt",
                session_id=session_id,
                model=use_model,
                detail={"cwd": str(use_cwd)},
            )
        call_timeout = self._client._control.call_timeout()
        start = time.perf_counter()
        try:
            await self._client._call(
                "session/load",
                {
                    "sessionId": session_id,
                    "cwd": use_cwd,
                    "mcpServers": [],
                },
                timeout=call_timeout,
            )
            await self._client._apply_session_config(session_id, use_model, timeout=call_timeout)
        except (AcpError, TimeoutError) as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if self._client.metrics is not None:
                self._client.metrics.inc("acp_resume_total", result="failure", method="load")
                self._client.metrics.set("acp_resume_latency_ms", duration_ms, result="failure")
            if self._client._lifecycle_log is not None:
                self._client._lifecycle_log.write(
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
        if self._client.metrics is not None:
            self._client.metrics.inc("acp_resume_total", result="success", method="load")
            self._client.metrics.set("acp_resume_latency_ms", duration_ms, result="success")
        if self._client._lifecycle_log is not None:
            self._client._lifecycle_log.write(
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
        call_timeout = timeout or self._client._control.call_timeout()
        await self._client._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "mode", "value": self._client.acp_mode},
            timeout=call_timeout,
        )
        await self._client._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": "model", "value": use_model},
            timeout=call_timeout,
        )
        self._client._session_models[session_id] = use_model

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
        use_model = _normalize_model(model or self._client.model)
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
        if self._client._lifecycle_log is not None:
            self._client._lifecycle_log.write(
                "session.new.attempt",
                chat_id=chat_id,
                model=use_model,
                detail={"cwd": str(use_cwd)},
            )

        session_new_timeout = self._client._control.call_timeout()
        start = time.perf_counter()
        try:
            session = await self._client._call(
                "session/new",
                {"cwd": use_cwd, "mcpServers": []},
                timeout=session_new_timeout,
            )
            session_id = session["sessionId"]
        except (AcpError, TimeoutError) as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            if self._client.metrics is not None:
                self._client.metrics.inc("acp_resume_total", result="failure", method="new")
                self._client.metrics.set("acp_resume_latency_ms", duration_ms, result="failure")
            if self._client._lifecycle_log is not None:
                self._client._lifecycle_log.write(
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
        if self._client.metrics is not None:
            self._client.metrics.inc("acp_resume_total", result="success", method="new")
            self._client.metrics.set("acp_resume_latency_ms", duration_ms, result="success")
        if self._client._lifecycle_log is not None:
            self._client._lifecycle_log.write(
                "session.new.success",
                chat_id=chat_id,
                session_id=session_id,
                model=use_model,
                detail={"cwd": str(use_cwd), "duration_ms": duration_ms},
            )

        if self._client._model_options is None:
            self._client._model_options = self._extract_model_options(session)

        # Honor the requested mode and model for this session.
        await self._client._apply_session_config(session_id, use_model, timeout=session_new_timeout)

        return await self._client._prompt(
            session_id,
            prompt_text,
            soft_timeout=soft_timeout,
            timeout=timeout,
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
        use_model = _normalize_model(model or self._client.model)
        # Only set the session model on follow-up when it has changed. Repeated
        # no-op model changes can re-render the session's system prefix and
        # destabilize the ACP subprocess.
        if self._client._session_models.get(session_id) != use_model:
            await self._client._call(
                "session/set_config_option",
                {"sessionId": session_id, "configId": "model", "value": use_model},
            )
            self._client._session_models[session_id] = use_model

        return await self._client._prompt(
            session_id,
            prompt_text,
            soft_timeout=soft_timeout,
            timeout=timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )

    async def _list_models(self) -> list[str]:
        """Create a throwaway session to discover the advertised model list."""
        probe_cwd = Path(os.getcwd()) / ".acp_model_probe"
        probe_cwd.mkdir(exist_ok=True)

        session = await self._client._call(
            "session/new",
            {"cwd": str(probe_cwd), "mcpServers": []},
        )
        self._client._model_options = self._extract_model_options(session)
        return self._client._model_options

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
        timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Send `session/prompt` and return the streamed reply."""
        client = self._client
        client._next_id += 1
        prompt_id = client._next_id
        prompt = _Prompt(
            session_id=session_id,
            prompt_id=prompt_id,
            text=text,
            future=client._loop.create_future(),
            cancel_done=client._loop.create_future(),
            soft_timeout=soft_timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )
        client._pending[prompt_id] = prompt.future
        client._active_prompts[session_id] = prompt
        with client._lock:
            client._last_stdout_at = time.monotonic()
            client._last_progress_at = time.monotonic()

        # If a cancel arrived before we registered the prompt, honor it now.
        if session_id in client._pending_cancels:
            client._pending_cancels.discard(session_id)
            prompt.cancelled = True
            if not prompt.cancel_done.done():
                prompt.cancel_done.set_result(None)
            client._loop.create_task(client._send_cancel_notification(session_id))

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
            await client._transport._send(
                {
                    "jsonrpc": "2.0",
                    "id": prompt_id,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": text}],
                    },
                },
                timeout=client._control_timeout,
            )

            if soft_timeout is not None and soft_timeout > 0:
                timeout_task = client._loop.create_task(
                    client._soft_timeout_canceller(prompt, soft_timeout)
                )

            prompt_timeout = timeout if timeout is not None else client.timeout
            start = client._loop.time()
            done, _pending = await asyncio.wait(
                [prompt.future, prompt.cancel_done],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=prompt_timeout,
            )

            raw: dict[str, Any] | None = None
            if prompt.future in done:
                raw = prompt.future.result()
            elif prompt.cancel_done in done:
                # Cancel was requested. Give the server a short grace period to
                # finish the aborted turn and send the prompt response.
                elapsed = client._loop.time() - start
                if prompt_timeout is not None:
                    remaining = max(0.0, prompt_timeout - elapsed)
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
                    await client._send_cancel_notification(session_id)
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
                    updates=list(prompt.updates),
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
                updates=list(prompt.updates),
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
                updates=list(prompt.updates),
            )
        finally:
            if timeout_task is not None and not timeout_task.done():
                timeout_task.cancel()
            client._active_prompts.pop(session_id, None)
            client._pending.pop(prompt_id, None)

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
            await self._client._send_cancel_notification(prompt.session_id)
        except Exception:
            logger.exception("Failed to send soft timeout cancel for %s", prompt.session_id)
