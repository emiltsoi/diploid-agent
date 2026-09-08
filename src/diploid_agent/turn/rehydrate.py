"""TurnRehydrate: stale-session recovery and prompt rehydration."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from diploid_agent.engine import TurnRequest, TurnResult
from diploid_agent.models import ActiveTurn, ChatResult, PartialTurn, SessionRecord, WakeEvent
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

    def _active_partial(self, chat_id: str) -> PartialTurn | None:
        """Return a PartialTurn snapshot of the in-flight turn, if any."""
        active: ActiveTurn | None = self.runtime._active_turns.get(chat_id)
        if active is None:
            return None
        record = self.runtime._active_record(chat_id)
        return PartialTurn(
            chat_id=chat_id,
            session_number=record.session_number if record else 0,
            turn_number=(record.turn_number + 1) if record else 1,
            user_message=active.user_message,
            message_text=active.message_text,
            thought_text=active.thought_text,
            thought_prefix=active.thought_prefix,
            thought_total=active.thought_total,
            full_text_offset=active.full_text_offset,
            updated_at=time.time(),
            current_intent=active.current_intent,
            last_side_effect=active.last_side_effect,
            last_side_effect_at=active.last_side_effect_at,
        )

    def _persisted_partial(self, chat_id: str, user_message: str) -> PartialTurn | None:
        """Load a persisted interrupted-turn snapshot for the same user message.

        The continuity plugin writes ``chat_active_turn.json`` while a turn
        streams and promotes a leftover to ``chat_interrupted_turn.json`` on the
        next wake. When the harness process itself died mid-turn, the in-process
        ``ActiveTurn`` is fresh, so the on-disk snapshot is the only record of
        what the interrupted turn had already produced.
        """
        chat_dir = self.runtime._chat_dir(chat_id)
        for name in ("chat_interrupted_turn.json", "chat_active_turn.json"):
            try:
                data = json.loads((chat_dir / name).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            persisted_message = data.get("user_message")
            if not isinstance(persisted_message, str):
                continue
            if persisted_message.strip() != (user_message or "").strip():
                continue
            text_fields = {
                key: data.get(key)
                for key in (
                    "message_text",
                    "thought_text",
                    "current_intent",
                    "last_side_effect",
                )
            }
            if any(v is not None and not isinstance(v, str) for v in text_fields.values()):
                continue
            try:
                return PartialTurn(
                    chat_id=chat_id,
                    session_number=int(data.get("session_number") or 0),
                    turn_number=int(data.get("turn_number") or 0),
                    user_message=persisted_message,
                    message_text=text_fields["message_text"] or "",
                    thought_text=text_fields["thought_text"] or "",
                    updated_at=float(data.get("updated_at") or 0.0),
                    current_intent=text_fields["current_intent"] or "",
                    last_side_effect=text_fields["last_side_effect"] or "",
                    last_side_effect_at=float(data.get("last_side_effect_at") or 0.0),
                )
            except (TypeError, ValueError):
                continue
        return None

    def _interrupted_turn_anchor(
        self,
        chat_id: str,
        reason: RehydrationReason | None,
        user_message: str = "",
    ) -> str | None:
        """Build an interrupted-turn anchor for the in-flight or persisted turn."""
        partial = self._active_partial(chat_id)
        if partial is None or not (
            partial.message_text or partial.thought_text or partial.last_side_effect
        ):
            # The live turn has produced nothing yet — a leftover on-disk
            # snapshot from a previous attempt at the same message is the
            # better record of what was lost.
            persisted = self._persisted_partial(chat_id, user_message)
            if persisted is not None:
                partial = persisted
        return self.context_builder.interrupted_turn_anchor(partial, reason)

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
        interrupted_anchor = self._interrupted_turn_anchor(
            chat_id, rehydration_reason, user_message
        )
        if self.runtime.lifecycle_log is not None:
            self.runtime.lifecycle_log.write(
                "rehydrate.start",
                chat_id=chat_id,
                reason=rehydration_reason.value,
                detail={
                    "log_prefix": log_prefix,
                    "restart_first": restart_first,
                    "interrupted_anchor": bool(interrupted_anchor),
                },
            )
        if restart_first:
            logger.warning("%s; restarting ACP transport for %s", log_prefix, chat_id)
            self.runtime._snapshot_plugin_states(chat_id)
            try:
                self.runtime.engine.restart(reason=log_prefix, chat_id=chat_id)
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

        resumed_id: str | None = None
        if (
            self.runtime.config.engine.acp_resume_enabled
            and self.controller.session._can_resume_record(chat_id, old_record, use_model)
        ):
            assert old_record is not None
            self.runtime._restore_plugin_states(chat_id)
            # After a transport restart the old session is likely gone from the
            # fresh child; give session/load a short budget instead of stalling
            # for the full acp_resume_timeout.
            resume_timeout = (
                self.runtime.config.engine.acp_resume_after_restart_timeout
                if restart_first
                else None
            )
            try:
                logger.warning(
                    "%s; attempting ACP session resume for %s (timeout=%s)",
                    log_prefix,
                    chat_id,
                    resume_timeout if resume_timeout is not None else "default",
                )
                resumed_id = self.runtime.call_engine_unlocked(
                    self.runtime.engine.resume_session,
                    old_record.session_id,
                    cwd=self.runtime._chat_dir(chat_id),
                    model=use_model,
                    mcp_servers=self.runtime._active_mcp_servers(chat_id),
                    timeout=resume_timeout,
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
                    interrupted_turn=interrupted_anchor,
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
                    timeout=self.runtime.config.engine.timeout,
                    chat_id=chat_id,
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
                        detail={"error": str(exc), "resume_timeout": resume_timeout},
                    )

        # If resume failed or was skipped, probe the old ACP session directly.
        # A live session can be reused without running prompt rehydration.
        if not resumed_id and old_record is not None and old_record.session_id:
            try:
                logger.warning(
                    "%s; probing ACP session %s for %s",
                    log_prefix,
                    old_record.session_id,
                    chat_id,
                )
                alive = self.runtime.call_engine_unlocked(
                    self.runtime.engine.session_alive,
                    old_record.session_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s: ACP session alive probe failed for %s: %s",
                    log_prefix,
                    chat_id,
                    exc,
                )
                alive = False

            if alive:
                logger.warning(
                    "ACP session %s is still alive for %s",
                    old_record.session_id,
                    chat_id,
                )
                if self.runtime.lifecycle_log is not None:
                    self.runtime.lifecycle_log.write(
                        "rehydrate.session_alive.success",
                        chat_id=chat_id,
                        session_id=old_record.session_id,
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
                    interrupted_turn=interrupted_anchor,
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
                    timeout=self.runtime.config.engine.timeout,
                    chat_id=chat_id,
                )
                result = self.runtime.call_engine_unlocked(
                    self.runtime.engine.prompt,
                    request,
                    session_id=old_record.session_id,
                    on_chunk=on_chunk,
                    on_update=on_update,
                )
                return result, old_record.session_id, pctx

        self.runtime._restore_plugin_states(chat_id)
        pctx = self.runtime.context_builder.build_first(
            chat_id,
            user_message,
            old_record,
            model=use_model,
            reply_to=reply_to,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
            continuation_anchor=continuation_anchor,
            interrupted_turn=interrupted_anchor,
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
                    self.runtime._snapshot_plugin_states(chat_id)
                    try:
                        self.runtime.engine.restart(reason=log_prefix, chat_id=chat_id)
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
