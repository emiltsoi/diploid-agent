from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status

from diploid_agent.config import (
    Config,
    ConfigPersistenceError,
    NotificationsConfig,
    TaskConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.models import WakeEvent
from diploid_agent.transport.base import RuntimeAPI
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.http.models import *


def register_config(app: FastAPI, runtime: RuntimeAPI, command_handler: CommandHandler, config: Config, _require_api_key: Callable[[str | None], None]) -> None:
    @app.get("/config")
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

    @app.get(
        "/task/config",
        response_model=TaskConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def task_config_get() -> TaskConfig:
        return command_handler.call(
            method="get_task_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/task/config",
        response_model=TaskConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def task_config_update(req: TaskConfig) -> TaskConfig:
        try:
            command_handler.call(
                method="update_task_config",
                task_config=req,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return command_handler.call(
            method="get_task_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.get(
        "/waker/config",
        response_model=WakerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def waker_config_get() -> WakerConfig:
        return command_handler.call(
            method="get_waker_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/waker/config",
        response_model=WakerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def waker_config_update(req: WakerConfig) -> WakerConfig:
        try:
            command_handler.call(
                method="update_waker_config",
                waker_config=req,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return command_handler.call(
            method="get_waker_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.get(
        "/timer/config",
        response_model=TimerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def timer_config_get() -> TimerConfig:
        return command_handler.call(
            method="get_timer_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/timer/config",
        response_model=TimerConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def timer_config_update(req: TimerConfig) -> TimerConfig:
        try:
            command_handler.call(
                method="update_timer_config",
                timer_config=req,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return command_handler.call(
            method="get_timer_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.get(
        "/notifications/config",
        response_model=NotificationsConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def notifications_config_get() -> NotificationsConfig:
        return command_handler.call(
            method="get_notifications_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post(
        "/notifications/config",
        response_model=NotificationsConfig,
        dependencies=[Depends(_require_api_key)],
    )
    def notifications_config_update(req: NotificationsConfig) -> NotificationsConfig:
        try:
            command_handler.call(
                method="update_notifications_config",
                notifications_config=req,
                requires_chat_id=False,
                catch=False,
            )
        except ConfigPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        return command_handler.call(
            method="get_notifications_config",
            requires_chat_id=False,
            catch=False,
        )

    @app.post("/timer", dependencies=[Depends(_require_api_key)])
    def timer_create(req: TimerRequest) -> dict[str, str]:
        event = WakeEvent(
            id="",
            chat_id=req.chat_id,
            reason=req.reason,
            priority=req.priority,
            scheduled_at=req.scheduled_at,
            payload=req.payload,
            silent=req.silent,
            created_at=time.time(),
            ready=True,
        )
        enqueued = runtime.wake_queue.enqueue(event)
        return {"event_id": enqueued.id}
