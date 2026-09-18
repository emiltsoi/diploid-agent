"""Cron inspection and manual-fire endpoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

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
    @app.get("/cron", dependencies=[Depends(_require_api_key)])
    def cron_get() -> dict[str, Any]:
        """Merged cron job list plus per-job state and reload warnings."""
        service = getattr(runtime, "cron_service", None)
        if service is None:
            return {"enabled": False, "jobs": [], "warnings": []}
        return service.snapshot()

    @app.post("/cron/{job_id}/run", dependencies=[Depends(_require_api_key)])
    def cron_run(job_id: str) -> dict[str, Any]:
        """Fire a cron job now, ignoring its schedule (operator door).

        Works on disabled and auto-disabled jobs too; an in-flight run is a
        conflict. The scheduled ``next_due_at`` is not consumed.
        """
        service = getattr(runtime, "cron_service", None)
        if service is None:
            raise HTTPException(status_code=404, detail="cron service unavailable")
        try:
            return service.run_now(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown cron job {job_id!r}") from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
