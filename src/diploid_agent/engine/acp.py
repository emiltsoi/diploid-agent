"""ACP engine implementation for the Devin CLI and other ACP-compatible binaries."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from diploid_agent.acp_client import AcpClient, AcpPromptResult
from diploid_agent.config import EngineConfig
from diploid_agent.engine.base import AgentEngine, TurnRequest, TurnResult

logger = logging.getLogger(__name__)


def _load_credentials() -> str | None:
    """Return the Windsurf API key from the ACP binary credentials file."""
    # Devin CLI writes its credentials here; other ACP-compatible binaries may
    # use WINDSURF_API_KEY or ACP_API_KEY instead.
    creds_path = Path.home() / ".local" / "share" / "devin" / "credentials.toml"
    if creds_path.exists():
        try:
            data = tomllib.loads(creds_path.read_text())
            return data.get("windsurf_api_key")
        except (OSError, tomllib.TOMLDecodeError) as exc:
            logger.warning("Failed to read credentials from %s: %s", creds_path, exc)
    return None


def _default_start_args(model: str) -> list[str]:
    """Return the default start arguments for a `devin acp` style ACP binary."""
    return ["acp", "--model", model.replace(".", "-")]


class AcpEngine(AgentEngine):
    """AgentEngine that drives an ACP-compatible binary over stdio.

    When `provider` is `"diploid"` (the default), the engine reads the
    Windsurf API key from the Devin credentials file and spawns `devin acp`.
    """

    def __init__(
        self,
        config: EngineConfig,
        *,
        api_key: str | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self._context_window_cache: dict[str, int | None] | None = None

        if api_key is None:
            api_key = _load_credentials()
            if not api_key:
                api_key = os.environ.get("WINDSURF_API_KEY") or os.environ.get("ACP_API_KEY")

        if config.start_args is None:
            start_args = _default_start_args(config.model)
        else:
            start_args = config.start_args

        self._client = AcpClient(
            model=config.model,
            permission_mode=config.permission_mode,
            timeout=config.timeout,
            startup_timeout=config.acp_startup_timeout,
            control_timeout=config.acp_control_timeout,
            watchdog_interval=config.acp_watchdog_interval,
            watchdog_timeout=config.acp_watchdog_timeout,
            agent_bin=config.bin,
            start_args=start_args,
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

    def model_context_window(self, model: str) -> int | None:
        """Return the context-window size for ``model`` in tokens.

        Uses the configured ``context_window`` override if set, otherwise
        queries the underlying Devin CLI ``models list`` output and caches it.
        """
        if self.config.context_window is not None:
            return self.config.context_window
        if self._context_window_cache is None:
            self._context_window_cache = self._load_model_context_windows()
        canonical = model.replace(".", "-")
        return self._context_window_cache.get(canonical)

    def _load_model_context_windows(self) -> dict[str, int | None]:
        """Parse ``<bin> models list --format json`` into a uid -> token map."""
        cache: dict[str, int | None] = {}
        bin_path = Path(self.config.bin).expanduser()
        if not bin_path.exists():
            return cache
        try:
            proc = subprocess.run(
                [str(bin_path), "models", "list", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            if proc.returncode != 0:
                logger.debug("Failed to list models: %s", proc.stderr)
                return cache
            data = json.loads(proc.stdout)
            for family in data.get("families", []):
                for variant in family.get("variants", []):
                    uid = variant.get("model_uid")
                    tokens = variant.get("max_context_tokens")
                    if uid and isinstance(tokens, int):
                        cache[uid] = tokens
            # Allow dotted aliases like ``swe-1.7`` by normalizing to dashed uids.
            dotted_cache = {uid.replace("-", "."): tokens for uid, tokens in cache.items() if uid}
            cache.update(dotted_cache)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not load model context windows: %s", exc)
        return cache


# Backward-compatible alias.
