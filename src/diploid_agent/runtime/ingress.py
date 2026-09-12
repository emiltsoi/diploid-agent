"""RuntimeIngress: public turn-entry surface and transport ingress routing."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from diploid_agent.dispatch import DispatchStatus
from diploid_agent.locking import locked
from diploid_agent.models import ChatResult, WakeEvent

logger = logging.getLogger(__name__)


class RuntimeIngress:
    """Owns the public turn-entry methods every transport calls.

    ``process``/``wake``/``continue_turn``/``dispatch`` acquire the per-chat
    instance lock and drive ``TurnController``; ``handle_ingress`` routes
    inbound HTTP to the registered protocol handler. ``AgentRuntime`` keeps
    thin delegates for the ``RuntimeAPI`` surface.
    """

    def __init__(
        self,
        *,
        config: Any,
        lock: Any,
        instance_manager: Any,
        wake_queue: Any,
        dispatch_store: Any,
        turn_controller: Any,
        planning: Any,
        ingress_handlers: dict[str, Any],
    ) -> None:
        self._config = config
        self._lock = lock
        self._instance_manager = instance_manager
        self._wake_queue = wake_queue
        self._dispatch_store = dispatch_store
        self._turn_controller = turn_controller
        self._planning = planning
        self._ingress_handlers = ingress_handlers

    def register_ingress_handler(self, protocol: str, handler: Any) -> None:
        """Register a protocol-specific inbound HTTP handler."""
        self._ingress_handlers[protocol] = handler

    async def handle_ingress(self, protocol: str, request: Any) -> Any:
        """Dispatch an inbound HTTP request to the registered handler."""
        from fastapi import HTTPException, Request, status

        if not isinstance(request, Request):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid ingress request object",
            )
        handler = self._ingress_handlers.get(protocol)
        if handler is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown ingress protocol: {protocol}",
            )
        return await handler.handle(request)

    def process(
        self,
        chat_id: str,
        user_message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        notify: bool = True,
    ) -> ChatResult:
        if not self._instance_manager.acquire(chat_id):
            # The chat is busy with another turn. Rather than dropping the user
            # message, queue it as a high-priority wake so it runs when the
            # current turn releases the lock.
            payload = {
                "user_message": user_message,
                "model": model,
                "reply_to": reply_to,
                "reply_to_is_bot": reply_to_is_bot,
                "reply_to_message_id": reply_to_message_id,
                "notify": True,
                "retry_after": 2.0,
            }
            self._wake_queue.enqueue(
                WakeEvent(
                    id="",
                    chat_id=chat_id,
                    reason="user_request",
                    priority=10,
                    scheduled_at=time.time(),
                    payload=payload,
                    silent=False,
                    created_at=time.time(),
                    ready=True,
                )
            )
            return ChatResult(
                reply="I'll get back to you in a moment.",
                notice="This chat is busy; your message was queued.",
            )
        try:
            result = self._turn_controller.process(
                chat_id,
                user_message,
                model=model,
                reply_to=reply_to,
                reply_to_is_bot=reply_to_is_bot,
                reply_to_message_id=reply_to_message_id,
                notify=notify,
                other_instance_running=False,
            )
            if result is not None and reply_to_message_id is not None:
                result.reply_to_message_id = reply_to_message_id
            return result
        finally:
            self._instance_manager.release(chat_id)

    @locked
    def dispatch(
        self,
        chat_id: str,
        context: str | None = None,
    ) -> ChatResult:
        return self._turn_controller.dispatch(chat_id, context=context)

    def continue_turn(self, dispatch_id: str, result: str) -> ChatResult:
        dispatch = self._dispatch_store.get(dispatch_id)
        if dispatch is None:
            return ChatResult(reply="Unknown dispatch.")
        chat_id = dispatch.chat_id
        if not self._instance_manager.acquire(chat_id):
            return ChatResult(reply="Another instance is currently handling this chat.")
        wake_id = f"wake-{dispatch_id}"
        self._wake_queue.ready(wake_id, now=time.time())
        try:
            chat_result = self._turn_controller.continue_turn(dispatch_id, result)
            if chat_result.turn_number is not None:
                self._wake_queue.complete(wake_id)
            return chat_result
        except Exception:
            self._wake_queue.fail(
                wake_id,
                retry_after=self._config.harness.waker.retry_after,
            )
            raise
        finally:
            self._instance_manager.release(chat_id)

    def wake(
        self,
        chat_id: str,
        event_id: str | None = None,
        reason: str | None = None,
        silent: bool | None = None,
    ) -> ChatResult:
        event = None
        if event_id is not None:
            event = self._wake_queue.get(event_id)
            if event is None:
                return ChatResult(reply="Unknown or already completed wake event.")
            chat_id = event.chat_id
            reason = event.reason
            if silent is None:
                silent = event.silent

        if silent is None:
            silent = False
        reason = reason or "user_request"

        if not self._instance_manager.acquire(chat_id):
            return ChatResult(reply="Chat is busy; wake re-enqueued.")

        try:
            payload = event.payload if event else {}
            if reason == "plan_task_update":
                user_message = self._planning._build_plan_task_update_message(payload)
            elif reason == "plan_completed":
                user_message = self._planning._build_plan_completed_message(payload)
            else:
                if payload and isinstance(payload.get("user_message"), str):
                    user_message = payload["user_message"]
                else:
                    user_message = f"[system wake: {reason}]"
                    if payload:
                        user_message += f"\n{json.dumps(payload, default=str)}"

            model = payload.get("model")
            reply_to = payload.get("reply_to")
            reply_to_is_bot = payload.get("reply_to_is_bot")
            reply_to_message_id = payload.get("reply_to_message_id")
            wake_notify = payload.get("notify", not silent)

            if reason == "dispatch" and "dispatch_id" in payload:
                dispatch = self._dispatch_store.get(payload["dispatch_id"])
                if dispatch and dispatch.status in (
                    DispatchStatus.PENDING,
                    DispatchStatus.TIMEOUT,
                    DispatchStatus.CANCELLED,
                    DispatchStatus.FAILED,
                ):
                    return self.continue_turn(
                        payload["dispatch_id"],
                        payload.get("result", dispatch.result or ""),
                    )

            result = self._turn_controller.process(
                chat_id,
                user_message,
                model=model,
                reply_to=reply_to,
                reply_to_is_bot=reply_to_is_bot,
                reply_to_message_id=reply_to_message_id,
                wake_event=event,
                notify=wake_notify,
                other_instance_running=False,
            )
            if result is not None and reply_to_message_id is not None:
                result.reply_to_message_id = reply_to_message_id
            return result
        finally:
            self._instance_manager.release(chat_id)
