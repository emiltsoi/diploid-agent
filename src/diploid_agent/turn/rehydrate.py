"""TurnRehydrate: stale-session recovery and prompt rehydration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from diploid_agent.engine import TurnRequest, TurnResult
from diploid_agent.models import ChatResult, SessionRecord, WakeEvent
from diploid_agent.plugins.contexts import PromptContext, RehydrationReason

if TYPE_CHECKING:
    from diploid_agent.turn.controller import TurnController

logger = logging.getLogger(__name__)


class TurnRehydrate:
    """Recover from stale ACP sessions by resuming or re-creating them."""

    def __init__(self, controller: TurnController) -> None:
        self.controller = controller

    @property
    def runtime(self) -> Any:
        return self.controller.runtime

    @property
    def acp_client(self) -> Any:
        return getattr(self.runtime, "acp_client", None)

    @property
    def _lock(self) -> Any:
        return self.runtime._lock

    @property
    def _chat_store(self) -> Any:
        return self.runtime._chat_store

    @property
    def _prompts(self) -> Any:
        return self.runtime._prompts

    @property
    def _mcp_skills(self) -> Any:
        return self.runtime._mcp_skills

    @property
    def _outbox(self) -> Any:
        return self.runtime._outbox

    @property
    def context_builder(self) -> Any:
        return self.runtime.context_builder

    @property
    def engine(self) -> Any:
        return self.runtime.engine

    def _rehydrate(
        self,
        chat_id: str,
        user_message: str,
        old_record: SessionRecord | None,
        use_model: str,
        *,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        continuation_anchor: str | None = None,
        wake_event: WakeEvent | None = None,
        other_instance_running: bool = False,
        on_chunk: Callable[[str], None],
        on_update: Callable[[dict[str, Any]], None],
        restart_first: bool = False,
        log_prefix: str = "Rehydrating",
        rehydration_reason: RehydrationReason | None = None,
    ) -> tuple[TurnResult, str, PromptContext] | ChatResult:
        """Resume a persisted ACP session if possible, otherwise start a new one."""
        if rehydration_reason is None:
            if restart_first:
                rehydration_reason = RehydrationReason.RESTART
            elif "stale" in log_prefix.lower() or "empty" in log_prefix.lower():
                rehydration_reason = RehydrationReason.STALE
            else:
                rehydration_reason = RehydrationReason.STALE
        if self.runtime.lifecycle_log is not None:
            self.runtime.lifecycle_log.write(
                "rehydrate.start",
                chat_id=chat_id,
                reason=rehydration_reason.value,
                detail={"log_prefix": log_prefix, "restart_first": restart_first},
            )
        if restart_first:
            logger.warning("%s; restarting ACP transport for %s", log_prefix, chat_id)
            try:
                self.runtime.engine.restart(reason=log_prefix)
                self.runtime._record_restart_memory(chat_id, reason=log_prefix)
            except Exception:
                logger.exception("Failed to restart ACP transport for %s", chat_id)
                if self.runtime.lifecycle_log is not None:
                    self.runtime.lifecycle_log.write(
                        "rehydrate.transport_restart_failure",
                        chat_id=chat_id,
                        reason=rehydration_reason.value,
                    )
                return ChatResult(
                    reply="Could not restart the ACP transport.",
                    notice="The agent is unavailable after a transport restart failure.",
                )

        if (
            self.runtime.config.engine.acp_resume_enabled
            and self.controller.session._can_resume_record(chat_id, old_record, use_model)
        ):
            assert old_record is not None
            try:
                logger.warning("%s; attempting ACP session resume for %s", log_prefix, chat_id)
                resumed_id = self.runtime.call_engine_unlocked(
                    self.runtime.engine.resume_session,
                    old_record.session_id,
                    cwd=self.runtime._chat_dir(chat_id),
                    model=use_model,
                    mcp_servers=self.runtime._active_mcp_servers(chat_id),
                )
                logger.warning("Resumed ACP session %s for %s", resumed_id, chat_id)
                if self.runtime.lifecycle_log is not None:
                    self.runtime.lifecycle_log.write(
                        "rehydrate.resume.success",
                        chat_id=chat_id,
                        session_id=resumed_id,
                        reason=rehydration_reason.value,
                    )
                pctx = self.runtime.context_builder.build_follow_up(
                    chat_id,
                    user_message,
                    old_record,
                    reply_to=reply_to,
                    reply_to_is_bot=reply_to_is_bot,
                    reply_to_message_id=reply_to_message_id,
                    continuation_anchor=continuation_anchor,
                    rehydrated=True,
                    rehydration_reason=RehydrationReason.RESUMED,
                    wake_event=wake_event,
                    other_instance_running=other_instance_running,
                )
                follow_model = pctx.model or use_model
                request = TurnRequest(
                    prompt=pctx.prompt,
                    cwd=self.runtime._chat_dir(chat_id),
                    model=follow_model,
                    mcp_servers=None,
                    soft_timeout=self.runtime.config.engine.soft_timeout,
                )
                result = self.runtime.call_engine_unlocked(
                    self.runtime.engine.prompt,
                    request,
                    session_id=resumed_id,
                    on_chunk=on_chunk,
                    on_update=on_update,
                )
                return result, resumed_id, pctx
            except (RuntimeError, TimeoutError) as exc:
                logger.warning("%s: ACP session resume failed for %s: %s", log_prefix, chat_id, exc)
                if self.runtime.lifecycle_log is not None:
                    self.runtime.lifecycle_log.write(
                        "rehydrate.resume.failure",
                        chat_id=chat_id,
                        session_id=old_record.session_id,
                        reason=rehydration_reason.value,
                        detail={"error": str(exc)},
                    )

        pctx = self.runtime.context_builder.build_first(
            chat_id,
            user_message,
            old_record,
            model=use_model,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            continuation_anchor=continuation_anchor,
            wake_event=wake_event,
            other_instance_running=other_instance_running,
            rehydrated=True,
            rehydration_reason=rehydration_reason,
        )
        model = pctx.model or use_model

        for attempt in range(2):
            try:
                result, session_id = self.runtime.call_engine_unlocked(
                    self.runtime._start_new_session,
                    chat_id,
                    pctx.prompt,
                    model,
                    on_chunk=on_chunk,
                    on_update=on_update,
                )
                if self.runtime.lifecycle_log is not None:
                    self.runtime.lifecycle_log.write(
                        "rehydrate.new_session.success",
                        chat_id=chat_id,
                        session_id=session_id,
                        reason=rehydration_reason.value,
                    )
                return result, session_id, pctx
            except (RuntimeError, TimeoutError) as exc:
                if self.runtime.engine.is_fatal_acp_error(exc) or (
                    not isinstance(exc, TimeoutError)
                    and self.runtime.engine.is_acp_error(exc)
                    and not self.runtime.engine.is_stale_session_error(exc)
                    and not self.runtime.engine.is_transport_error(exc)
                ):
                    logger.exception(
                        "Unrecoverable ACP error during %s for %s", log_prefix, chat_id
                    )
                    if old_record is not None:
                        old_record.last_stop_reason = "error"
                    if self.runtime.lifecycle_log is not None:
                        self.runtime.lifecycle_log.write(
                            "rehydrate.new_session.error",
                            chat_id=chat_id,
                            reason=rehydration_reason.value,
                            detail={"error": str(exc)},
                        )
                    return ChatResult(
                        reply=f"Could not continue: {exc}",
                        notice="An ACP configuration error prevented the turn from completing.",
                    )

                logger.warning(
                    "%s: ACP call failed for %s (attempt %d): %s",
                    log_prefix,
                    chat_id,
                    attempt + 1,
                    exc,
                )

                if attempt == 0:
                    try:
                        self.runtime.engine.restart(reason=log_prefix)
                        self.runtime._record_restart_memory(chat_id, reason=log_prefix)
                    except Exception:
                        logger.exception("Failed to restart ACP transport for %s", chat_id)
                        if old_record is not None:
                            old_record.last_stop_reason = "error"
                        if self.runtime.lifecycle_log is not None:
                            self.runtime.lifecycle_log.write(
                                "rehydrate.transport_restart_failure",
                                chat_id=chat_id,
                                reason=rehydration_reason.value,
                                detail={"attempt": attempt + 1},
                            )
                        return ChatResult(
                            reply="Could not restart the ACP transport.",
                            notice="The agent is unavailable after a transport restart failure.",
                        )
                    continue

                logger.error(
                    "%s: ACP transport still unresponsive for %s after retry", log_prefix, chat_id
                )
                if old_record is not None:
                    old_record.last_stop_reason = "error"
                return ChatResult(
                    reply="The ACP transport is not responding after multiple attempts.",
                    notice="Please check the agent configuration and try again.",
                )
