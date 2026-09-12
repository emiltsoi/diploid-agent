"""Shared per-turn ACP pipeline for TurnProcess and TurnDispatch.

``_call_engine`` runs the engine invocation half of a turn (before/after
engine hooks, ACP error classification, stale/empty-reply rehydration) and
``_finalize_turn`` runs the recording half (metrics, record mutation,
before_record_turn, memory write, turn hooks, ChatResult assembly). Callers
supply their divergent bits as parameters or small hooks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from diploid_agent.engine import TurnRequest, TurnResult
from diploid_agent.models import ActiveTurn, ChatResult, SessionRecord, final_segment_reply
from diploid_agent.plugins.base import TurnInfo
from diploid_agent.plugins.contexts import (
    EngineCallContext,
    EngineResultContext,
    RecordTurnContext,
    TurnErrorContext,
)
from diploid_agent.turn.base import TurnComponent
from diploid_agent.turn.stream import TurnStream
from diploid_agent.turn.utils import join_notices

logger = logging.getLogger(__name__)


@dataclass
class EngineCallOutcome:
    """Result of ``_call_engine`` once the turn's reply is settled."""

    result: TurnResult
    session_id: str | None
    reply: str
    use_model: str
    is_new: bool
    rehydrate_notice: str | None
    memory_flags: dict[str, bool] = field(default_factory=dict)
    prompt_chars: int = 0


class TurnPipeline(TurnComponent):
    """Shared engine-call + finalize spine for the per-turn ACP loop."""

    def _call_engine(
        self,
        *,
        chat_id: str,
        user_message: str,
        prompt: str,
        use_model: str,
        record: SessionRecord | None,
        old_record: SessionRecord | None,
        is_new: bool,
        force_new_session: bool,
        active: ActiveTurn,
        stream: TurnStream,
        memory_flags: dict[str, bool],
        request_timeout: float | None = None,
        rehydrate_kwargs: dict[str, Any] | None = None,
    ) -> EngineCallOutcome | ChatResult:
        """Run the engine call with hooks, error classification, and rehydration.

        Returns an ``EngineCallOutcome`` on success or a ``ChatResult`` when the
        turn must end early (plugin short-circuit is still an outcome; ACP error
        returns and rehydrate bail-outs are ChatResults).
        """
        rehydrate_kwargs = rehydrate_kwargs or {}
        rehydrate_notice: str | None = None
        session_id: str | None = None
        reply = ""
        try:
            request = TurnRequest(
                prompt=prompt,
                cwd=self.runtime._chat_dir(chat_id),
                model=use_model,
                mcp_servers=(
                    self.runtime._mcp_skills._active_mcp_servers(chat_id)
                    if is_new or force_new_session
                    else None
                ),
                soft_timeout=self.runtime.config.engine.soft_timeout,
                timeout=request_timeout,
                chat_id=chat_id,
            )
            call_ctx = self.runtime._plugins.before_engine_call(
                chat_id,
                EngineCallContext(
                    chat_id=chat_id,
                    request=request,
                    session_id=(None if is_new or force_new_session else old_record.session_id),
                    record=record,
                    on_chunk=stream.on_chunk,
                    on_update=stream.on_update,
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
                reply = active.final_reply_text(result.reply)

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
            reply = active.final_reply_text(result.reply)
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
                on_chunk=stream.on_chunk,
                on_update=stream.on_update,
                restart_first=restart_first,
                log_prefix=log_prefix,
                **rehydrate_kwargs,
            )
            if isinstance(ret, ChatResult):
                return ret
            result, session_id, pctx = ret
            rehydrate_notice = join_notices(rehydrate_notice, pctx.notice)
            memory_flags = pctx.memory_flags
            use_model = pctx.model or use_model
            is_new = session_id != (old_record.session_id if old_record else None)
            active.session_id = result.session_id
            reply = active.final_reply_text(result.reply)

        # If a prompt came back with a genuinely empty reply, the ACP session is
        # almost certainly stale. Rehydrate once. Use the raw ACP reply so a
        # thought-only prefix does not look empty.
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
                on_chunk=stream.on_chunk,
                on_update=stream.on_update,
                restart_first=False,
                log_prefix="ACP empty-reply rehydration",
                **rehydrate_kwargs,
            )
            if isinstance(ret, ChatResult):
                return ret
            result, session_id, pctx = ret
            rehydrate_notice = join_notices(rehydrate_notice, pctx.notice)
            memory_flags = pctx.memory_flags
            use_model = pctx.model or use_model
            is_new = session_id != (old_record.session_id if old_record else None)
            active.session_id = result.session_id
            reply = active.final_reply_text(result.reply)

        return EngineCallOutcome(
            result=result,
            session_id=session_id,
            reply=reply,
            use_model=use_model,
            is_new=is_new,
            rehydrate_notice=rehydrate_notice,
            memory_flags=memory_flags,
            prompt_chars=len(request.prompt) if request.prompt else 0,
        )

    def _finalize_turn(
        self,
        *,
        chat_id: str,
        user_message: str,
        old_record: SessionRecord | None,
        outcome: EngineCallOutcome,
        turn_number: int,
        session_number: int,
        previous_updated_at: float,
        continue_word: str,
        turn_start: float,
        force_new_session: bool = False,
        label: str | None = None,
        system_note: str | None = None,
        budget_notice: str | None = None,
        notice: str | None = None,
        after_turn_end: Callable[[], None] | None = None,
        after_append: Callable[[], None] | None = None,
        after_result: Callable[[ChatResult, SessionRecord], None] | None = None,
    ) -> tuple[ChatResult, SessionRecord]:
        """Record the completed turn and build its ChatResult.

        Returns ``(chat_result, record)`` — the record may be a *new* object
        when the turn crossed an ACP session boundary, so callers rebind their
        local. ``after_turn_end`` runs inside the record lock right after
        ``on_turn_end``; ``after_append`` right after ``_append_record``;
        ``after_result`` after the ChatResult is assembled.
        """
        result = outcome.result
        reply = outcome.reply
        is_new = outcome.is_new
        record_is_new = is_new or force_new_session
        session_id = outcome.session_id
        use_model = outcome.use_model
        memory_flags = outcome.memory_flags
        rehydrate_notice = outcome.rehydrate_notice

        partial: str | None = None
        if result.partial:
            partial = self.runtime._prompts._partial_notice(result, continue_word=continue_word)
            notice = partial if notice is None else f"{notice}\n\n{partial}"
        if rehydrate_notice:
            notice = rehydrate_notice if notice is None else f"{rehydrate_notice}\n\n{notice}"

        latency = time.perf_counter() - turn_start
        turn_metrics = self.runtime._runtime_metrics._record_turn_metrics(
            chat_id,
            turn_number,
            use_model,
            result.usage,
            latency,
            prompt_chars=outcome.prompt_chars,
        )

        mcp_names = self.runtime._mcp_skills._active_mcp_server_names(chat_id)
        skill_names = self.runtime._mcp_skills._active_skill_names(chat_id)
        with self._lock:
            # `force_new_session` (context-pressure fresh mode) also crosses an
            # ACP session boundary even though it took the follow-up prompt
            # path — give it a new record so session_number tracks real ACP
            # sessions instead of mutating the old record's session_id.
            if record_is_new:
                if not is_new:
                    session_number = self.runtime._next_session_number(chat_id)
                record = self.runtime._prompts._create_record(
                    chat_id,
                    session_number,
                    session_id,
                    use_model,
                    reply,
                    memory_flags,
                    label=(
                        label
                        if label is not None
                        else self.runtime.context_builder.generate_label(chat_id, user_message)
                    ),
                )
                record.pending_turn_number = turn_number
                record.enabled_mcp_servers = mcp_names
                record.enabled_skills = sorted(skill_names)
                self.runtime._chat_state(chat_id).sessions[record.session_number] = record
            else:
                record = old_record
                record.enabled_mcp_servers = mcp_names
                record.enabled_skills = sorted(skill_names)
                record.session_id = session_id
                record.model = use_model

            record.consume_turn_number()
            record.updated_at = time.time()
            record.cumulative_metrics = self.runtime._runtime_metrics._per_chat_metrics[
                chat_id
            ].get("cumulative", {})
            if not result.partial:
                record.last_stop_reason = "completed"
            elif (active := self.runtime._active_turns.get(chat_id)) is not None and active.stopped:
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
            # The first turn on a new ACP session is the cleanest
            # prompt_chars/input_tokens sample — later turns accumulate
            # history in input_tokens and the ratio collapses.
            if record_is_new and record.first_turn_metrics is None and record.last_turn_metrics:
                record.first_turn_metrics = dict(record.last_turn_metrics)
            record.persona_memory_exceeded = (record_ctx.memory_flags or {}).get(
                "persona_memory_exceeded", False
            )
            record.chat_memory_exceeded = (record_ctx.memory_flags or {}).get(
                "chat_memory_exceeded", False
            )
            reply = record_ctx.reply

            # Preserve the split between rehydrate/transition notices
            # (non-deferred) and the partial/timeout notice (deferred by
            # auto-continue). If a plugin explicitly rewrote the combined
            # notice, treat the whole thing as non-deferred and let the plugin
            # own it.
            turn_notice = rehydrate_notice
            turn_partial = partial
            combined = join_notices(rehydrate_notice, partial)
            if record_ctx.notice is not None and record_ctx.notice != combined:
                turn_notice = record_ctx.notice
                turn_partial = None

            # Collect plugin memory items since the previous turn.
            extra_items = self.runtime._plugins.memory_items(chat_id, since=previous_updated_at)

            assistant_notice = join_notices(turn_notice, turn_partial)
            self.runtime._memory_manager(chat_id).record_turn(
                user_message=user_message,
                reply=reply,
                model=use_model,
                session_number=record.session_number,
                turn_number=record.turn_number,
                extra_items=extra_items,
                notice=assistant_notice,
                system_note=system_note,
                final_segment=final_segment_reply(result, reply),
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

            if after_turn_end is not None:
                after_turn_end()

            transition = self.runtime._prompts._check_chat_memory_transition(chat_id, record)
            if transition:
                turn.notice = join_notices(turn.notice, transition)

            self.runtime._append_record(record)
            if after_append is not None:
                after_append()

            turn.reply = reply
            turn.turn_number = record.turn_number
            self.runtime._plugins.after_turn(chat_id, turn)
            chat_result = ChatResult(
                reply=reply,
                notice=join_notices(budget_notice, turn.notice, turn.partial_notice),
                session_id=record.session_id,
                session_number=record.session_number,
                turn_number=record.turn_number,
                metrics=record.last_turn_metrics,
            )
            if after_result is not None:
                after_result(chat_result, record)
        return chat_result, record

    def _emit_turn_error(
        self,
        chat_id: str,
        record: SessionRecord | None,
        user_message: str,
        exc: Exception,
    ) -> None:
        """Fire the on_turn_error plugin hook; caller re-raises."""
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

    def _cleanup_turn(self, chat_id: str, record: SessionRecord | None) -> ActiveTurn | None:
        """Pop the active turn, emit on_sleeping, and wake stream waiters."""
        with self._lock:
            active = self.runtime._active_turns.get(chat_id)
            self.runtime._active_turns.pop(chat_id, None)
            self.runtime._plugins.on_sleeping(chat_id, record, reason="turn_end")
        if active is not None:
            with active._condition:
                active._condition.notify_all()
        return active
