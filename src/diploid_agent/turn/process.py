"""TurnProcess: per-turn ACP loop and helpers."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from diploid_agent.engine import TurnRequest, TurnResult
from diploid_agent.models import ActiveTurn, ChatResult, PartialTurn, SessionRecord, WakeEvent
from diploid_agent.plugins.base import TurnInfo
from diploid_agent.plugins.contexts import (
    EngineCallContext,
    EngineResultContext,
    RecordTurnContext,
    RehydrationReason,
    TurnErrorContext,
    TurnStartContext,
)
from diploid_agent.turn.notifier import _NotifyStream, _OutboxHeartbeat

if TYPE_CHECKING:
    from diploid_agent.turn.controller import TurnController

logger = logging.getLogger(__name__)

# Cap the in-memory thought stream so a runaway model cannot exhaust memory
# or produce giant wake/auto-continue payloads and Telegram status messages.
_MAX_THOUGHT_TEXT_CHARS = 20000


def _join_notices(*parts: str | None) -> str | None:
    """Concatenate non-empty notice strings with a blank line between them."""
    joined = "\n\n".join(p for p in parts if p)
    return joined or None


class TurnProcess:
    """Main per-turn ACP loop for a single chat."""

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
    def _runtime_metrics(self) -> Any:
        return self.runtime._runtime_metrics

    @property
    def _planning(self) -> Any:
        return self.runtime._planning

    @property
    def _subagent(self) -> Any:
        return self.runtime._subagent

    @property
    def session(self) -> Any:
        return self.controller.session

    @property
    def rehydrate(self) -> Any:
        return self.controller.rehydrate

    @property
    def _dispatch(self) -> Any:
        return self.controller._dispatch

    def _has_pending_continuation(self, chat_id: str, wake_event: WakeEvent | None = None) -> bool:
        """Return True if another auto-continue wake is pending for this chat."""
        exclude_id = wake_event.id if wake_event else None
        for event in self.runtime.wake_queue.pending(chat_id=chat_id):
            if event.id == exclude_id:
                continue
            if event.reason == "auto_continue" and not event.silent:
                return True
        return False

    def _seed_active_turn(self, active: ActiveTurn, wake_event: WakeEvent | None) -> None:
        """Pre-populate the active turn with content from a previous partial turn."""
        if not wake_event or not isinstance(wake_event.payload, dict):
            return
        message_text = wake_event.payload.get("message_text") or ""
        thought_text = wake_event.payload.get("thought_text") or ""
        thought_prefix = wake_event.payload.get("thought_prefix") or ""
        active.thought_text = thought_text[-_MAX_THOUGHT_TEXT_CHARS:]
        active.thought_prefix = thought_prefix or thought_text[:_MAX_THOUGHT_TEXT_CHARS]
        active.full_text = active.thought_text + message_text
        active.thought_total = wake_event.payload.get("thought_total") or len(active.thought_text)
        active.full_text_offset = wake_event.payload.get("full_text_offset") or 0
        self._recompute_message_text(active)

    @staticmethod
    def _append_full_text(active: ActiveTurn, text: str) -> None:
        """Append an agent_message chunk, keeping full_text as a rolling window.

        When the buffer grows past _MAX_THOUGHT_TEXT_CHARS we drop the oldest
        prefix (which is part of the thought) and adjust full_text_offset so
        message_text can still be computed correctly.
        """
        active.full_text += text
        excess = len(active.full_text) - _MAX_THOUGHT_TEXT_CHARS
        if excess > 0:
            dropped = active.full_text[:excess]
            active.full_text_offset += len(dropped)
            active.full_text = active.full_text[excess:]

    @staticmethod
    def _full_text_has_thought_prefix(active: ActiveTurn) -> bool:
        """Return True if the retained full_text window starts within the thought.

        For short thoughts, full_text must simply start with the thought_text.
        For long (capped) thoughts, we compare the first _MAX chars of the
        thought against the retained window starting at full_text_offset.
        """
        if not active.thought_total or not active.full_text:
            return False
        if active.thought_total <= _MAX_THOUGHT_TEXT_CHARS:
            return active.full_text.startswith(active.thought_text)
        # The thought is longer than the cap. The first _MAX chars are in
        # thought_prefix; the retained window starts at full_text_offset.
        prefix_start = active.full_text_offset
        prefix_end = min(_MAX_THOUGHT_TEXT_CHARS, prefix_start + len(active.full_text))
        if prefix_end <= prefix_start:
            return False
        return (
            active.full_text[: prefix_end - prefix_start]
            == active.thought_prefix[prefix_start:prefix_end]
        )

    @staticmethod
    def _recompute_message_text(active: ActiveTurn) -> None:
        """Keep message_text as the part of full_text after the thought prefix.

        thought_total is the cumulative length of agent_thought updates.
        full_text_offset is how much of the agent_message stream has been
        discarded from the front. We only remove the thought prefix if we can
        verify that full_text actually contains it; otherwise the ACP subprocess
        streams the answer and thought on separate channels.
        """
        if not active.full_text:
            active.message_text = ""
            return
        if active.thought_total > 0 and TurnProcess._full_text_has_thought_prefix(active):
            start = max(0, active.thought_total - active.full_text_offset)
            active.message_text = active.full_text[start:]
        else:
            active.message_text = active.full_text
        if len(active.message_text) > _MAX_THOUGHT_TEXT_CHARS:
            active.message_text = active.message_text[-_MAX_THOUGHT_TEXT_CHARS:]

    @staticmethod
    def _append_thought_text(active: ActiveTurn, text: str) -> None:
        """Append an agent_thought update and bump the cumulative counter."""
        active.thought_text += text
        active.thought_prefix += text
        active.thought_total += len(text)
        if len(active.thought_text) > _MAX_THOUGHT_TEXT_CHARS:
            active.thought_text = active.thought_text[-_MAX_THOUGHT_TEXT_CHARS:]
        if len(active.thought_prefix) > _MAX_THOUGHT_TEXT_CHARS:
            active.thought_prefix = active.thought_prefix[:_MAX_THOUGHT_TEXT_CHARS]

    @staticmethod
    def _final_reply_text(active: ActiveTurn, reply: str) -> str:
        """Return the final reply with any leading thought text removed."""
        if not active.thought_total:
            return reply
        # For short thoughts the full thought_text is still accurate; for long
        # (capped) thoughts we verify with the first _MAX chars prefix.
        if active.thought_total <= _MAX_THOUGHT_TEXT_CHARS:
            thought = active.thought_text
        else:
            thought = active.thought_prefix
        if thought and reply.startswith(thought):
            return reply[active.thought_total :].lstrip("\n")
        return reply

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
            if chat_id in self.runtime._active_turns:
                return ChatResult(
                    reply="A turn is already in progress for this chat.",
                    notice="Send /stop to cancel it, or wait for it to finish.",
                )

            record = self.runtime._active_record(chat_id)
            current_model = self.runtime._model(record)
            route = self.runtime.resolve_model(chat_id, user_message, record)
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

            self.runtime.match_and_activate_skills(chat_id, user_message)
            active_skill_names = self.runtime._active_skill_names(chat_id)

            notice: str | None = None
            partial: str | None = None
            memory_flags: dict[str, bool] = {}
            continuation_anchor = self.runtime.context_builder.continuation_anchor(
                record, user_message
            )

            hard_timeout_before = record is not None and record.last_stop_reason == "timeout"
            model_changed = record is not None and use_model != current_model
            previous_skills = set(record.enabled_skills or []) if record else set()
            skills_changed = record is not None and previous_skills != active_skill_names
            force_new_session = False

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
                self._seed_active_turn(active, wake_event)
                self.runtime._active_turns[chat_id] = active
                is_new = True
                old_record: SessionRecord | None = record
            else:
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
                self._seed_active_turn(active, wake_event)
                self.runtime._active_turns[chat_id] = active
                is_new = False
                session_number = record.session_number
                old_record = record

        rehydrate_notice: str | None = None

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

        def _maybe_emit_partial() -> None:
            a = self.runtime._active_turns.get(chat_id)
            if a is None:
                return
            record = self.runtime._active_record(chat_id)
            turn_number = record.turn_number + 1 if record else 1
            self.runtime._plugins.on_partial(
                chat_id,
                PartialTurn(
                    chat_id=chat_id,
                    session_number=record.session_number if record else 0,
                    turn_number=turn_number,
                    user_message=user_message,
                    message_text=a.message_text,
                    thought_text=a.thought_text,
                    thought_prefix=a.thought_prefix,
                    thought_total=a.thought_total,
                    full_text_offset=a.full_text_offset,
                    updated_at=time.time(),
                ),
            )

        def _on_chunk(text: str) -> None:
            with self.runtime._lock:
                a = self.runtime._active_turns.get(chat_id)
                if a:
                    self._append_full_text(a, text)
                    self._recompute_message_text(a)
            if a:
                with a._condition:
                    a._condition.notify_all()
            _maybe_emit_partial()

        def _on_update(update: dict[str, Any]) -> None:
            if update.get("sessionUpdate") not in ("agent_thought", "agent_thought_chunk"):
                return
            content = update.get("content", {})
            if isinstance(content, list):
                text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            elif content.get("type") == "text":
                text = content.get("text", "")
            else:
                text = ""
            if not text:
                return
            with self.runtime._lock:
                a = self.runtime._active_turns.get(chat_id)
                if a:
                    self._append_thought_text(a, text)
                    self._recompute_message_text(a)
            if a:
                with a._condition:
                    a._condition.notify_all()
            _maybe_emit_partial()

        turn_start = time.perf_counter()
        continue_word = (
            self.runtime.config.engine.continuation_triggers[0].capitalize()
            if self.runtime.config.engine.continuation_triggers
            else "Continue"
        )

        rehydrate_notice: str | None = None

        try:
            try:
                request = TurnRequest(
                    prompt=prompt,
                    cwd=self.runtime._chat_dir(chat_id),
                    model=use_model,
                    mcp_servers=(
                        self.runtime._active_mcp_servers(chat_id)
                        if is_new or force_new_session
                        else None
                    ),
                    soft_timeout=self.runtime.config.engine.soft_timeout,
                    chat_id=chat_id,
                )
                call_ctx = self.runtime._plugins.before_engine_call(
                    chat_id,
                    EngineCallContext(
                        chat_id=chat_id,
                        request=request,
                        session_id=(None if is_new or force_new_session else old_record.session_id),
                        record=record,
                        on_chunk=_on_chunk,
                        on_update=_on_update,
                    ),
                )
                if isinstance(call_ctx, ChatResult):
                    result = TurnResult(
                        reply=call_ctx.reply,
                        session_id=call_ctx.session_id,
                        stop_reason=None,
                        usage=None,
                        cancelled=False,
                        partial=False,
                    )
                    is_short_circuit = True
                else:
                    is_short_circuit = False
                    if is_new or force_new_session:
                        result = self.runtime.call_engine_unlocked(
                            self.runtime.engine.prompt,
                            call_ctx.request,
                            on_chunk=call_ctx.on_chunk,
                            on_update=call_ctx.on_update,
                        )
                        active.session_id = result.session_id
                        session_id = result.session_id
                    else:
                        result = self.runtime.call_engine_unlocked(
                            self.runtime.engine.prompt,
                            call_ctx.request,
                            session_id=call_ctx.session_id or old_record.session_id,
                            on_chunk=call_ctx.on_chunk,
                            on_update=call_ctx.on_update,
                        )
                        session_id = result.session_id or old_record.session_id
                    reply = result.reply

                result_ctx = self.runtime._plugins.after_engine_call(
                    chat_id,
                    EngineResultContext(
                        chat_id=chat_id,
                        record=record,
                        result=result,
                        reply=result.reply,
                        usage=result.usage,
                        stop_reason=result.stop_reason,
                    ),
                )
                if is_short_circuit:
                    result = result_ctx.result
                    session_id = result_ctx.result.session_id or session_id
                else:
                    result = result_ctx.result
                    result.reply = result_ctx.reply
                    result.usage = result_ctx.usage
                    result.stop_reason = result_ctx.stop_reason
                    session_id = result.session_id or session_id
                reply = result.reply
            except (RuntimeError, TimeoutError) as exc:
                if isinstance(exc, TimeoutError) or self.runtime.engine.is_transport_error(exc):
                    log_prefix = "ACP transport unresponsive"
                    restart_first = True
                elif self.runtime.engine.is_stale_session_error(exc):
                    stale_id = old_record.session_id if old_record else "unknown"
                    logger.warning("ACP session %s stale; rehydrating for %s", stale_id, chat_id)
                    log_prefix = "ACP session stale"
                    restart_first = False
                elif self.runtime.engine.is_acp_error(exc):
                    logger.exception("Unrecoverable ACP error for %s", chat_id)
                    if record is not None:
                        record.last_stop_reason = "error"
                    return ChatResult(
                        reply=f"Could not continue: {exc}",
                        notice="An ACP error prevented the turn from completing.",
                    )
                else:
                    logger.exception("Unexpected RuntimeError during ACP call for %s", chat_id)
                    return ChatResult(
                        reply=f"Unexpected error: {exc}",
                        notice="The turn stopped due to an unexpected error.",
                    )

                ret = self.rehydrate._rehydrate(
                    chat_id,
                    user_message,
                    old_record,
                    use_model,
                    reply_to=reply_to,
                    reply_to_is_bot=reply_to_is_bot,
                    reply_to_message_id=reply_to_message_id,
                    continuation_anchor=continuation_anchor,
                    wake_event=wake_event,
                    other_instance_running=other_instance_running,
                    on_chunk=_on_chunk,
                    on_update=_on_update,
                    restart_first=restart_first,
                    log_prefix=log_prefix,
                )
                if isinstance(ret, ChatResult):
                    return ret
                result, session_id, pctx = ret
                rehydrate_notice = pctx.notice
                memory_flags = pctx.memory_flags
                use_model = pctx.model or use_model
                is_new = session_id != (old_record.session_id if old_record else None)
                active.session_id = result.session_id
                reply = self._final_reply_text(active, result.reply)

            # If a prompt came back with a genuinely empty reply,
            # the ACP session is almost certainly stale. Rehydrate once.
            # Use the raw ACP reply so a thought-only prefix does not look empty.
            if not result.reply and not result.partial:
                empty_id = session_id or (old_record.session_id if old_record else "unknown")
                logger.warning(
                    "ACP session %s returned an empty reply; rehydrating for %s",
                    empty_id,
                    chat_id,
                )
                ret = self.rehydrate._rehydrate(
                    chat_id,
                    user_message,
                    old_record,
                    use_model,
                    reply_to=reply_to,
                    reply_to_is_bot=reply_to_is_bot,
                    reply_to_message_id=reply_to_message_id,
                    continuation_anchor=continuation_anchor,
                    wake_event=wake_event,
                    other_instance_running=other_instance_running,
                    on_chunk=_on_chunk,
                    on_update=_on_update,
                    restart_first=False,
                    log_prefix="ACP empty-reply rehydration",
                )
                if isinstance(ret, ChatResult):
                    return ret
                result, session_id, pctx = ret
                rehydrate_notice = _join_notices(rehydrate_notice, pctx.notice)
                memory_flags = pctx.memory_flags
                use_model = pctx.model or use_model
                is_new = session_id != (old_record.session_id if old_record else None)
                active.session_id = result.session_id
                reply = self._final_reply_text(active, result.reply)

            if result.partial:
                partial = self.runtime._partial_notice(result, continue_word=continue_word)
                notice = partial if notice is None else f"{notice}\n\n{partial}"
            if rehydrate_notice:
                notice = rehydrate_notice if notice is None else f"{rehydrate_notice}\n\n{notice}"

            latency = time.perf_counter() - turn_start
            turn_metrics = self.runtime._record_turn_metrics(
                chat_id,
                record.turn_number + 1 if record else 1,
                use_model,
                result.usage,
                latency,
                prompt_chars=len(request.prompt) if request.prompt else 0,
            )

            mcp_names = self.runtime._active_mcp_server_names(chat_id)
            skill_names = self.runtime._active_skill_names(chat_id)
            with self.runtime._lock:
                if is_new:
                    record = self.runtime._create_record(
                        chat_id,
                        session_number,
                        session_id,
                        use_model,
                        reply,
                        memory_flags,
                        label=self.runtime.context_builder.generate_label(chat_id, user_message),
                    )
                    record.enabled_mcp_servers = mcp_names
                    record.enabled_skills = sorted(skill_names)
                    self.runtime._chat_state(chat_id).sessions[record.session_number] = record
                else:
                    record = old_record
                    record.enabled_mcp_servers = mcp_names
                    record.enabled_skills = sorted(skill_names)
                    record.session_id = session_id
                    record.model = use_model
                    record.updated_at = time.time()

                previous_turn_number = record.turn_number
                record.turn_number += 1
                previous_updated_at = record.updated_at if previous_turn_number > 0 else 0.0
                record.updated_at = time.time()
                record.cumulative_metrics = self.runtime._per_chat_metrics[chat_id].get(
                    "cumulative", {}
                )
                if not result.partial:
                    record.last_stop_reason = "completed"
                elif active.stopped:
                    record.last_stop_reason = "stopped"
                elif result.cancelled:
                    record.last_stop_reason = "cancelled"
                elif result.stop_reason:
                    record.last_stop_reason = result.stop_reason
                else:
                    record.last_stop_reason = "timeout"

                record_ctx = self.runtime._plugins.before_record_turn(
                    chat_id,
                    RecordTurnContext(
                        chat_id=chat_id,
                        record=record,
                        turn_number=record.turn_number,
                        reply=reply,
                        notice=notice,
                        memory_flags=memory_flags,
                        metrics=turn_metrics,
                    ),
                )
                record.turn_number = record_ctx.turn_number
                record.last_turn_metrics = (
                    record_ctx.metrics if record_ctx.metrics is not None else turn_metrics
                )
                record.persona_memory_exceeded = (record_ctx.memory_flags or {}).get(
                    "persona_memory_exceeded", False
                )
                record.chat_memory_exceeded = (record_ctx.memory_flags or {}).get(
                    "chat_memory_exceeded", False
                )
                reply = record_ctx.reply

                # Preserve the split between rehydrate/transition notices (non-deferred)
                # and the partial/timeout notice (deferred by auto-continue). If a plugin
                # explicitly rewrote the combined notice, treat the whole thing as
                # non-deferred and let the plugin own it.
                turn_notice = rehydrate_notice
                turn_partial = partial
                combined = _join_notices(rehydrate_notice, partial)
                if record_ctx.notice is not None and record_ctx.notice != combined:
                    turn_notice = record_ctx.notice
                    turn_partial = None

                # Collect plugin memory items since the previous turn.
                extra_items = self.runtime._plugins.memory_items(chat_id, since=previous_updated_at)

                assistant_notice = _join_notices(turn_notice, turn_partial)
                self.runtime._memory_manager(chat_id).record_turn(
                    user_message=user_message,
                    reply=reply,
                    model=use_model,
                    session_number=record.session_number,
                    turn_number=record.turn_number,
                    extra_items=extra_items,
                    notice=assistant_notice,
                )

                turn = TurnInfo(
                    chat_id=chat_id,
                    session_id=record.session_id,
                    session_number=record.session_number,
                    turn_number=record.turn_number,
                    updated_at=record.updated_at,
                    last_stop_reason=record.last_stop_reason,
                    user_message=user_message,
                    reply=reply,
                    notice=turn_notice,
                    partial_notice=turn_partial,
                )
                self.runtime._plugins.on_turn_end(chat_id, turn)

                transition = self.runtime._check_chat_memory_transition(chat_id, record)
                if transition:
                    turn.notice = _join_notices(turn.notice, transition)

                self.runtime._append_record(record)
                turn.reply = reply
                turn.turn_number = record.turn_number
                self.runtime._plugins.after_turn(chat_id, turn)
                chat_result = ChatResult(
                    reply=reply,
                    notice=_join_notices(budget_notice, turn.notice, turn.partial_notice),
                    session_id=record.session_id,
                    session_number=record.session_number,
                    turn_number=record.turn_number,
                    metrics=record.last_turn_metrics,
                )
                chat_result.continuation = self._has_pending_continuation(chat_id, wake_event)
        except Exception as exc:
            self.runtime._plugins.on_turn_error(
                chat_id,
                TurnErrorContext(
                    chat_id=chat_id,
                    record=record,
                    user_message=user_message,
                    exception=exc,
                    now=time.time(),
                ),
            )
            raise
        finally:
            if outbox_heartbeat is not None:
                outbox_heartbeat.stop()
            with self.runtime._lock:
                active = self.runtime._active_turns.get(chat_id)
                self.runtime._active_turns.pop(chat_id, None)
                self.runtime._plugins.on_sleeping(chat_id, record, reason="turn_end")
            if active is not None:
                with active._condition:
                    active._condition.notify_all()
            if notifier_stream is not None:
                if "chat_result" in vars():
                    notifier_stream.finish(chat_result)
                else:
                    notifier_stream.finish(ChatResult(reply=""))

        if notify and "chat_result" in vars() and self.runtime._outbox_delivery_enabled:
            self.runtime._deliver_chat_result(chat_id, chat_result)

        return chat_result
