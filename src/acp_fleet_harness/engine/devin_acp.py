"""Devin ACP implementation of AgentEngine."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from acp_fleet_harness.acp_client import AcpClient, AcpPromptResult
from acp_fleet_harness.config import DevinConfig
from acp_fleet_harness.engine.base import AgentEngine, TurnRequest, TurnResult

logger = logging.getLogger(__name__)


class DevinAcpEngine(AgentEngine):
    """AgentEngine that drives the Devin ACP binary over stdio."""

    def __init__(
        self,
        config: DevinConfig,
        *,
        api_key: str | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self._client = AcpClient(
            model=config.model,
            permission_mode=config.permission_mode,
            timeout=config.timeout,
            devin_bin=config.bin,
            api_key=api_key,
            metrics=metrics,
        )

    def _to_result(self, result: AcpPromptResult) -> TurnResult:
        return TurnResult(
            reply=result.reply,
            session_id=result.session_id,
            stop_reason=result.stop_reason,
            usage=result.usage,
            cancelled=result.cancelled,
            partial=result.partial,
            timed_out=result.timed_out,
            updates=result.updates,
        )

    def create_session(
        self,
        prompt: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        soft_timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Create a new ACP session. Exposed for test compatibility."""
        if cwd is not None:
            cwd = Path(cwd)
        return self._client.create_session(
            prompt,
            cwd=cwd,
            model=model,
            mcp_servers=mcp_servers,
            soft_timeout=soft_timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )

    def send_message(
        self,
        session_id: str,
        prompt: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        soft_timeout: float | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> AcpPromptResult:
        """Send a follow-up prompt. Exposed for test compatibility."""
        if cwd is not None:
            cwd = Path(cwd)
        return self._client.send_message(
            session_id,
            prompt,
            cwd=cwd,
            model=model,
            soft_timeout=soft_timeout,
            on_chunk=on_chunk,
            on_update=on_update,
        )

    def _coerce_result(self, result: AcpPromptResult | TurnResult) -> TurnResult:
        if isinstance(result, TurnResult):
            return result
        return self._to_result(result)

    def prompt(
        self,
        request: TurnRequest,
        *,
        session_id: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> TurnResult:
        if session_id is None:
            result = self.create_session(
                request.prompt,
                cwd=request.cwd,
                model=request.model,
                mcp_servers=request.mcp_servers,
                soft_timeout=request.soft_timeout,
                on_chunk=on_chunk,
                on_update=on_update,
            )
        else:
            result = self.send_message(
                session_id,
                request.prompt,
                cwd=request.cwd,
                model=request.model,
                soft_timeout=request.soft_timeout,
                on_chunk=on_chunk,
                on_update=on_update,
            )
        return self._coerce_result(result)

    def cancel(self, session_id: str) -> None:
        self._client.cancel(session_id)

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def health(self) -> bool:
        return self._client.health()

    def session_alive(self, session_id: str) -> bool:
        return self._client.session_alive(session_id)

    def restart(self) -> None:
        self.restart_transport()

    def restart_transport(self) -> None:
        """Restart the ACP transport."""
        self._client.restart_transport()

    def close(self) -> None:
        self._client.close()

    def is_stale_session_error(self, exc: BaseException) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        return bool(self._client._is_stale_session_error(exc))  # type: ignore[attr-defined]
