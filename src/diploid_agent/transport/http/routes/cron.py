"""Read-only cron inspection endpoint."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from diploid_agent.config import Config
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler


def register_cron(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.get("/cron")
    def cron_get() -> dict[str, Any]:
        """Merged cron job list plus per-job state and reload warnings."""
        service = getattr(runtime, "cron_service", None)
        if service is None:
            return {"enabled": False, "jobs": [], "warnings": []}
        return service.snapshot()
