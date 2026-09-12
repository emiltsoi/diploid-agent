"""TurnProcess: per-turn ACP loop and helpers."""

from __future__ import annotations

import logging
import time

from diploid_agent.models import (
    ActiveTurn,
    ChatResult,
    SessionRecord,
    WakeEvent,
)
from diploid_agent.plugins.contexts import (
    RehydrationReason,
    TurnStartContext,
)
from diploid_agent.turn.notifier import _NotifyStream, _OutboxHeartbeat
from diploid_agent.turn.pipeline import TurnPipeline
from diploid_agent.turn.stream import TurnStream

logger = logging.getLogger(__name__)

class TurnProcess(TurnPipeline):
    """Main per-turn ACP loop for a single chat."""

    def _has_pending_continuation(self, chat_id: str, wake_event: WakeEvent | None = None) -> bool:
        """Return True if another auto-continue wake is pending for this chat."""
        exclude_id = wake_event.id if wake_event else None
        for event in self.runtime.wake_queue.pending(chat_id=chat_id):
            if event.id == exclude_id:
                continue
            if event.reason == "auto_continue" and not event.silent:
                return True
        return False

    def process(
        self,
        chat_id: str,
        user_message: str,
        *,
        model: str | None = None,
        reply_to: str | None = None,
        reply_to_is_bot: bool | None = None,
        reply_to_message_id: int | None = None,
        wake_event: WakeEvent | None = None,
        other_instance_running: bool = False,
        notify: bool = True,
    ) -> ChatResult:
        """Send a message to the persona session and return the reply.

        The RLock is released while the ACP call is in flight so `stop` can
        acquire it and request cancellation.
        """
        notifier_stream: _NotifyStream | None = None
        with self.runtime._lock:
            if self.runtime._restart_draining.is_set():
                return ChatResult(
                    reply=(
                        "The service is draining for a restart; "
                        "please send your message again in a moment."
                    ),
                    notice="Restart in progress; your turn was not started.",
                )
            if chat_id in self.runtime._active_turns:
                return ChatResult(
                    reply="A turn is already in progress for this chat.",
                    notice="Send /stop to cancel it, or wait for it to finish.",
                )

            record = self.runtime._active_record(chat_id)
            current_model = self.runtime._prompts._model(record)
            route = self.runtime._prompts.resolve_model(chat_id, user_message, record)
            if route.budget_exceeded:
                return ChatResult(reply="", notice=route.notice)
            budget_notice = route.notice
            if model is None:
                current_model = route.model
            start_ctx = self.runtime._plugins.before_turn(
                chat_id,
                TurnStartContext(
                    chat_id=chat_id,
                    user_message=user_message,
                    model=model or current_model,
                    record=record,
                    reply_to=reply_to,
                    reply_to_is_bot=reply_to_is_bot,
                    reply_to_message_id=reply_to_message_id,
                    now=time.time(),
                ),
            )
            if isinstance(start_ctx, ChatResult):
                return start_ctx

            user_message = start_ctx.user_message
            use_model = start_ctx.model or current_model
            reply_to = start_ctx.reply_to
            reply_to_is_bot = start_ctx.reply_to_is_bot
            reply_to_message_id = start_ctx.reply_to_message_id

            self.runtime._mcp_skills.match_and_activate_skills(chat_id, user_message)
            active_skill_names = self.runtime._mcp_skills._active_skill_names(chat_id)

            notice: str | None = None
            memory_flags: dict[str, bool] = {}
            continuation_anchor = self.runtime.context_builder.continuation_anchor(
                record, user_message
            )

            hard_timeout_before = record is not None and record.last_stop_reason == "timeout"
            model_changed = record is not None and use_model != current_model
            previous_skills = set(record.enabled_skills or []) if record else set()
            skills_changed = record is not None and previous_skills != active_skill_names
            force_new_session = False

            continue_word = (
                self.runtime.config.engine.continuation_triggers[0].capitalize()
                if self.runtime.config.engine.continuation_triggers
                else "Continue"
            )

            if (
                hard_timeout_before
                and not self.runtime.config.engine.acp_timeout_auto_resend
                and not self.runtime.context_builder.is_continuation_message(user_message)
            ):
                return ChatResult(
                    reply=(
                        f"The previous turn was interrupted by the hard time limit and did not "
                        f"complete. Reply `{continue_word}` to retry it, or send a new message to "
                        f"start fresh."
                    ),
                    notice="Waiting for confirmation before resending the interrupted turn.",
                    session_id=record.session_id,
                    session_number=record.session_number,
                    turn_number=record.turn_number,
                )

            resend_system_note: str | None = None
            if hard_timeout_before:
                resend_system_note = (
                    "Resuming the previous turn after it was interrupted by a hard timeout."
                )

            if record is None or model_changed or hard_timeout_before or skills_changed:
                if record and (model_changed or hard_timeout_before or skills_changed):
                    self.runtime._archive_active_session(chat_id, record)
                self.runtime._restore_plugin_states(chat_id)
                pctx = self.runtime.context_builder.build_first(
                    chat_id,
                    user_message,
                    record,
                    model=use_model,
                    reply_to=reply_to,
                    reply_to_is_bot=reply_to_is_bot,
                    reply_to_message_id=reply_to_message_id,
                    continuation_anchor=continuation_anchor,
                    wake_event=wake_event,
                    other_instance_running=other_instance_running,
                    rehydrated=hard_timeout_before,
                    rehydration_reason=RehydrationReason.TIMEOUT if hard_timeout_before else None,
                )
                if hard_timeout_before and self.runtime.lifecycle_log is not None:
                    self.runtime.lifecycle_log.write(
                        "rehydrate.timeout",
                        chat_id=chat_id,
                        reason=RehydrationReason.TIMEOUT.value,
                    )
                prompt = pctx.prompt
                notice = pctx.notice
                memory_flags = pctx.memory_flags
                use_model = pctx.model or use_model
                session_number = self.runtime._next_session_number(chat_id)
                cwd = self.runtime._chat_dir(chat_id)
                cwd.mkdir(parents=True, exist_ok=True)
                self.runtime.skills.sync_to_chat(chat_id, cwd, active_skill_names)
                active = ActiveTurn(chat_id, None, user_message, time.time())
                active.seed_from_wake(wake_event)
                self.runtime._active_turns[chat_id] = active
                is_new = True
                old_record: SessionRecord | None = record
            else:
                self.runtime.skills.refresh_to_chat(
                    chat_id, self.runtime._chat_dir(chat_id), active_skill_names
                )
                pctx = self.runtime.context_builder.build_follow_up(
                    chat_id,
                    user_message,
                    record,
                    reply_to=reply_to,
                    reply_to_is_bot=reply_to_is_bot,
                    reply_to_message_id=reply_to_message_id,
                    continuation_anchor=continuation_anchor,
                    wake_event=wake_event,
                    other_instance_running=other_instance_running,
                )
                prompt = pctx.prompt
                use_model = pctx.model or use_model
                force_new_session = pctx.force_new_session
                active = ActiveTurn(chat_id, record.session_id, user_message, time.time())
                active.seed_from_wake(wake_event)
                self.runtime._active_turns[chat_id] = active
                is_new = False
                session_number = record.session_number
                old_record = record

            # Reserve a turn number up front and persist it. If the turn is
            # killed, the record on disk will carry the reserved number so the
            # next wake does not reuse it.
            turn_number = 1
            previous_updated_at = 0.0
            if record is not None:
                previous_updated_at = record.updated_at if record.turn_number > 0 else 0.0
                turn_number = record.reserve_turn_number()
                self.runtime._append_record(record)

        telegram_config = self.runtime.config.harness.telegram
        notifier_stream = _NotifyStream(
            self.controller,
            chat_id,
            notify,
            telegram_config.stream_thoughts,
            telegram_config.min_edit_message_interval,
            wake_event=wake_event,
        )
        notifier_stream.start()

        outbox_heartbeat: _OutboxHeartbeat | None = None
        if notify and self.runtime._outbox_delivery_enabled:
            outbox_heartbeat = _OutboxHeartbeat(self.runtime, chat_id, active)
            outbox_heartbeat.start()

        stream = TurnStream(self.runtime, chat_id)

        turn_start = time.perf_counter()

        try:
            outcome = self._call_engine(
                chat_id=chat_id,
                user_message=user_message,
                prompt=prompt,
                use_model=use_model,
                record=record,
                old_record=old_record,
                is_new=is_new,
                force_new_session=force_new_session,
                active=active,
                stream=stream,
                memory_flags=memory_flags,
                request_timeout=self.runtime.config.engine.timeout,
                rehydrate_kwargs={
                    "reply_to": reply_to,
                    "reply_to_is_bot": reply_to_is_bot,
                    "reply_to_message_id": reply_to_message_id,
                    "continuation_anchor": continuation_anchor,
                    "wake_event": wake_event,
                    "other_instance_running": other_instance_running,
                },
            )
            if isinstance(outcome, ChatResult):
                return outcome

            def _set_continuation(cr: ChatResult, _record: SessionRecord) -> None:
                cr.continuation = self._has_pending_continuation(chat_id, wake_event)

            chat_result, record = self._finalize_turn(
                chat_id=chat_id,
                user_message=user_message,
                old_record=old_record,
                outcome=outcome,
                turn_number=turn_number,
                session_number=session_number,
                previous_updated_at=previous_updated_at,
                continue_word=continue_word,
                turn_start=turn_start,
                force_new_session=force_new_session,
                system_note=resend_system_note,
                budget_notice=budget_notice,
                notice=notice,
                after_result=_set_continuation,
            )
        except Exception as exc:
            self._emit_turn_error(chat_id, record, user_message, exc)
            raise
        finally:
            if outbox_heartbeat is not None:
                outbox_heartbeat.stop()
            self._cleanup_turn(chat_id, record)
            if notifier_stream is not None:
                if "chat_result" in vars():
                    notifier_stream.finish(chat_result)
                else:
                    notifier_stream.finish(ChatResult(reply=""))

        if notify and "chat_result" in vars() and self.runtime._outbox_delivery_enabled:
            self.runtime._deliver_chat_result(chat_id, chat_result)

        return chat_result
