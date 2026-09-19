from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status

from diploid_agent.config import (
    AuthorshipConfig,
    Config,
    ConfigPersistenceError,
    NotificationsConfig,
    TaskConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.models import WakeEvent
from diploid_agent.runtime.wake_queue import WAKE_BUDGET_REASON_PREFIXES
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *

SELF_WAKE_REASON = "self_wake"


def _self_wake_permitted(config: Config) -> bool:
    """The enabled authorship plugin's ``self_wake_enabled`` toggle is the
    master switch for agent-initiated self-wakes."""
    for plugin in config.harness.plugins:
        if plugin.name == "authorship" and plugin.enabled:
            try:
                authorship = AuthorshipConfig.model_validate(plugin.config or {})
            except Exception:  # noqa: BLE001
                return False
            return authorship.self_wake_enabled
    return False


def register_config(
    app: FastAPI,
    runtime: RuntimeAPI,
    command_handler: CommandHandler,
    config: Config,
    _require_api_key: Callable[[str | None], None],
) -> None:
    @app.get("/config", dependencies=[Depends(_require_api_key)])
    def config_get() -> dict[str, Any]:
        """Return the current live runtime configuration (excluding secrets)."""
        return command_handler.call(
            method="get_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.patch(
        "/config",
        dependencies=[Depends(_require_api_key)],
    )
    def config_update(patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a partial live update to Telegram and/or plugin configuration."""
        try:
            return command_handler.call(
                method="update_config",
                patch=patch,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    def _section_get(get_method: str) -> Callable[[], Any]:
        def _get() -> Any:
            return command_handler.call(method=get_method, requires_chat_id=False, catch=False)

        return _get

    def _section_update(
        get_method: str, update_method: str, kwarg: str, cfg_cls: type
    ) -> Callable[..., Any]:
        def _update(req: Any) -> Any:
            try:
                command_handler.call(
                    method=update_method,
                    requires_chat_id=False,
                    catch=False,
                    **{kwarg: req},
                )
            except ConfigPersistenceError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from exc
            return command_handler.call(method=get_method, requires_chat_id=False, catch=False)

        # FastAPI reads the request-body model from this annotation.
        _update.__annotations__ = {"req": cfg_cls, "return": cfg_cls}
        return _update

    for section, cfg_cls in (
        ("task", TaskConfig),
        ("waker", WakerConfig),
        ("timer", TimerConfig),
        ("notifications", NotificationsConfig),
    ):
        path = f"/{section}/config"
        get_method = f"get_{section}_config"
        update_method = f"update_{section}_config"
        app.get(path, response_model=cfg_cls, dependencies=[Depends(_require_api_key)])(
            _section_get(get_method)
        )
        app.post(path, response_model=cfg_cls, dependencies=[Depends(_require_api_key)])(
            _section_update(get_method, update_method, f"{section}_config", cfg_cls)
        )

    def _enforce_self_wake_policy(req: TimerRequest, now: float) -> None:
        if not _self_wake_permitted(config):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="self_wake is not enabled for this persona (authorship toggle).",
            )
        timer_cfg = config.harness.timer
        if req.scheduled_at > now + timer_cfg.self_wake_max_delay_seconds:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "self_wake scheduled too far out "
                    f"(max {timer_cfg.self_wake_max_delay_seconds:.0f}s)"
                ),
            )
        self_wakes = [
            e
            for e in runtime.wake_queue.pending(chat_id=req.chat_id)
            if e.reason.startswith(WAKE_BUDGET_REASON_PREFIXES)
        ]
        if len(self_wakes) >= timer_cfg.self_wake_max_pending:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"self_wake budget reached: {len(self_wakes)} pending "
                    f"(max {timer_cfg.self_wake_max_pending})"
                ),
            )
        latest = max((e.created_at for e in self_wakes), default=0.0)
        if latest and now - latest < timer_cfg.self_wake_min_interval_seconds:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    "self_wake rate limit: a self-wake was armed "
                    f"{now - latest:.0f}s ago "
                    f"(min {timer_cfg.self_wake_min_interval_seconds:.0f}s)"
                ),
            )

    def _enqueue_timer(req: TimerRequest, now: float) -> dict[str, str]:
        event = WakeEvent(
            id="",
            chat_id=req.chat_id,
            reason=req.reason,
            priority=req.priority,
            scheduled_at=req.scheduled_at,
            payload=req.payload,
            silent=req.silent,
            created_at=now,
            ready=True,
        )
        enqueued = runtime.wake_queue.enqueue(event)
        return {"event_id": enqueued.id}

    @app.post("/timer", dependencies=[Depends(_require_api_key)])
    def timer_create(req: TimerRequest) -> dict[str, str]:
        now = time.time()
        if req.reason.startswith(SELF_WAKE_REASON):
            _enforce_self_wake_policy(req, now)
        return _enqueue_timer(req, now)

    @app.post("/timer/self", dependencies=[Depends(_require_api_key)])
    def timer_create_self(req: TimerRequest) -> dict[str, str]:
        """Agent-facing wake door: the self-wake toggle and budgets always
        apply here regardless of the requested reason, and the reason is
        normalized under a budget prefix so the event counts toward the
        same budgets it was checked against. ``/timer`` remains the
        operator door.
        """
        now = time.time()
        if not req.reason.startswith(WAKE_BUDGET_REASON_PREFIXES):
            req = req.model_copy(
                update={"reason": f"{SELF_WAKE_REASON}:{req.reason}"}
            )
        _enforce_self_wake_policy(req, now)
        return _enqueue_timer(req, now)

    @app.get("/timer/pending", dependencies=[Depends(_require_api_key)])
    def timer_pending(chat_id: str, reason: str | None = None) -> dict[str, Any]:
        events = runtime.wake_queue.pending(chat_id=chat_id)
        if reason is not None:
            events = [e for e in events if e.reason == reason]
        return {
            "events": [
                {
                    "id": e.id,
                    "chat_id": e.chat_id,
                    "reason": e.reason,
                    "scheduled_at": e.scheduled_at,
                    "created_at": e.created_at,
                    "silent": e.silent,
                    "priority": e.priority,
                    "attempts": e.attempts,
                }
                for e in events
            ]
        }

    @app.post("/timer/cancel", dependencies=[Depends(_require_api_key)])
    def timer_cancel(req: TimerCancelRequest) -> dict[str, Any]:
        event = runtime.wake_queue.cancel_event(req.event_id, chat_id=req.chat_id)
        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No such pending wake event for this chat.",
            )
        return {"ok": True, "event_id": req.event_id}
