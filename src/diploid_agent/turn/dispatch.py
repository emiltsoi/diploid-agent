"""TurnDispatch: background dispatch and continue-turn logic."""

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Any

from diploid_agent.dispatch import DispatchStatus
from diploid_agent.engine import TurnRequest, TurnResult
from diploid_agent.models import ActiveTurn, ChatResult, PartialTurn, WakeEvent
from diploid_agent.plugins.base import TurnInfo
from diploid_agent.plugins.contexts import (
    DispatchCompleteContext,
    DispatchContinueContext,
    DispatchCreateContext,
    EngineCallContext,
    EngineResultContext,
    RecordTurnContext,
    TurnErrorContext,
    TurnStartContext,
)

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


def _locked(method):
    """Run a TurnDispatch method under the runtime RLock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class TurnDispatch:
    """Background dispatch / continue-turn logic for a single chat."""

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

    @_locked
    def dispatch(
        self,
        chat_id: str,
        context: str | None = None,
    ) -> ChatResult:
        """Register a new dispatch for this chat and return its id."""
        record = self.runtime._active_record(chat_id)
        if record is None:
            return ChatResult(reply="No active session for this chat.")

        create_ctx = self.runtime._plugins.before_dispatch(
            chat_id,
            DispatchCreateContext(chat_id=chat_id, record=record, context=context),
        )

        dispatch = self.runtime.dispatch_store.add(
            chat_id, record.session_id, context=create_ctx.context
        )

        event = WakeEvent(
            id=f"wake-{dispatch.id}",
            chat_id=chat_id,
            reason="dispatch",
            priority=1,
            scheduled_at=time.time(),
            created_at=time.time(),
            silent=True,
            payload={"dispatch_id": dispatch.id},
            ready=False,
        )
        self.runtime.wake_queue.enqueue(event)

        create_ctx.dispatch = dispatch
        self.runtime._plugins.after_dispatch(chat_id, create_ctx)
        self.runtime._plugins.on_dispatch(chat_id, dispatch)
        return ChatResult(reply="Dispatched.", dispatch_id=dispatch.id)

    def continue_turn(
        self,
        dispatch_id: str,
        result: str,
        *,
        notify: bool = True,
    ) -> ChatResult:
        """Resume the ACP session after a background dispatch completes."""
        with self._lock:
            dispatch = self.runtime.dispatch_store.get(dispatch_id)
            if dispatch is None:
                return ChatResult(reply="Unknown dispatch.")
            if dispatch.status == DispatchStatus.COMPLETED:
                return ChatResult(reply="Dispatch already completed.")

            dispatch.result = result

            continue_ctx = self.runtime._plugins.before_dispatch_continue(
                dispatch.chat_id,
                DispatchContinueContext(
                    chat_id=dispatch.chat_id,
                    dispatch=dispatch,
                    result=result,
                ),
            )
            dispatch.result = continue_ctx.result
            result = continue_ctx.result

            # Persist the full result and a short summary so the continuation prompt
            # can reference a clean file path even for manually-continued dispatches.
            self.runtime._persist_subagent_result(dispatch, result)

            chat_id = dispatch.chat_id
            if chat_id in self.runtime._active_turns:
                return ChatResult(
                    reply="A turn is already in progress; continuation queued.",
                    notice=f"dispatch:{dispatch_id}",
                )

            record = self.runtime._active_record(chat_id)
            if record is None:
                return ChatResult(reply="No active session for this chat.")

            use_model = self.runtime._model(record)
            user_message = "Continue"
            route = self.runtime.resolve_model(chat_id, user_message, record)
            if route.budget_exceeded:
                return ChatResult(reply="", notice=route.notice)
            budget_notice = route.notice
            start_ctx = self.runtime._plugins.before_turn(
                chat_id,
                TurnStartContext(
                    chat_id=chat_id,
                    user_message=user_message,
                    model=use_model,
                    record=record,
                    now=time.time(),
                ),
            )
            if isinstance(start_ctx, ChatResult):
                return start_ctx
            user_message = start_ctx.user_message
            use_model = start_ctx.model or use_model

            continuation_anchor = self._planning._build_dispatch_continuation(dispatch)
            memory_flags: dict[str, bool] = {}
            turn_start = time.perf_counter()

            pctx = self.runtime.context_builder.build_follow_up(
                chat_id,
                user_message,
                record,
                continuation_anchor=continuation_anchor,
            )
            prompt = pctx.prompt
            use_model = pctx.model or use_model
            active = ActiveTurn(chat_id, record.session_id, user_message, time.time())
            self.runtime._active_turns[chat_id] = active
            is_new = False
            old_record = record
            session_number = record.session_number

        rehydrate_notice: str | None = None
        partial: str | None = None
        turn_result: TurnResult | None = None
        session_id: str | None = None
        reply: str = ""

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
                    updated_at=time.time(),
                ),
            )

        def _on_chunk(text: str) -> None:
            with self._lock:
                a = self.runtime._active_turns.get(chat_id)
                if a:
                    a.message_text += text
                    if len(a.message_text) > _MAX_THOUGHT_TEXT_CHARS:
                        a.message_text = a.message_text[-_MAX_THOUGHT_TEXT_CHARS:]
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
            with self._lock:
                a = self.runtime._active_turns.get(chat_id)
                if a:
                    a.thought_text += text
                    if len(a.thought_text) > _MAX_THOUGHT_TEXT_CHARS:
                        a.thought_text = a.thought_text[-_MAX_THOUGHT_TEXT_CHARS:]
            if a:
                with a._condition:
                    a._condition.notify_all()
            _maybe_emit_partial()

        try:
            try:
                request = TurnRequest(
                    prompt=prompt,
                    cwd=self.runtime._chat_dir(chat_id),
                    model=use_model,
                    soft_timeout=self.runtime.config.engine.soft_timeout,
                )
                call_ctx = self.runtime._plugins.before_engine_call(
                    chat_id,
                    EngineCallContext(
                        chat_id=chat_id,
                        request=request,
                        session_id=old_record.session_id,
                        record=record,
                        on_chunk=_on_chunk,
                        on_update=_on_update,
                    ),
                )
                if isinstance(call_ctx, ChatResult):
                    turn_result = TurnResult(
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
                    turn_result = self.runtime.call_engine_unlocked(
                        self.runtime.engine.prompt,
                        call_ctx.request,
                        session_id=call_ctx.session_id or old_record.session_id,
                        on_chunk=call_ctx.on_chunk,
                        on_update=call_ctx.on_update,
                    )
                    session_id = turn_result.session_id or old_record.session_id
                    reply = turn_result.reply

                result_ctx = self.runtime._plugins.after_engine_call(
                    chat_id,
                    EngineResultContext(
                        chat_id=chat_id,
                        record=record,
                        result=turn_result,
                        reply=turn_result.reply,
                        usage=turn_result.usage,
                        stop_reason=turn_result.stop_reason,
                    ),
                )
                if is_short_circuit:
                    turn_result = result_ctx.result
                    session_id = result_ctx.result.session_id or session_id
                else:
                    turn_result = result_ctx.result
                    turn_result.reply = result_ctx.reply
                    turn_result.usage = result_ctx.usage
                    turn_result.stop_reason = result_ctx.stop_reason
                    session_id = turn_result.session_id or session_id
                reply = turn_result.reply
            except (RuntimeError, TimeoutError) as exc:
                if isinstance(exc, TimeoutError) or self.runtime.engine.is_transport_error(exc):
                    log_prefix = "ACP transport unresponsive"
                    restart_first = True
                elif self.runtime.engine.is_stale_session_error(exc):
                    logger.warning(
                        "ACP session %s stale; rehydrating for %s",
                        old_record.session_id,
                        chat_id,
                    )
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
                    continuation_anchor=continuation_anchor,
                    on_chunk=_on_chunk,
                    on_update=_on_update,
                    restart_first=restart_first,
                    log_prefix=log_prefix,
                )
                if isinstance(ret, ChatResult):
                    return ret
                turn_result, session_id, pctx = ret
                rehydrate_notice = _join_notices(rehydrate_notice, pctx.notice)
                memory_flags = pctx.memory_flags
                use_model = pctx.model or use_model
                is_new = session_id != (old_record.session_id if old_record else None)
                active.session_id = turn_result.session_id
                reply = turn_result.reply

            if not reply and turn_result and not turn_result.partial:
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
                    continuation_anchor=continuation_anchor,
                    on_chunk=_on_chunk,
                    on_update=_on_update,
                    restart_first=False,
                    log_prefix="ACP empty-reply rehydration",
                )
                if isinstance(ret, ChatResult):
                    return ret
                turn_result, session_id, pctx = ret
                rehydrate_notice = _join_notices(rehydrate_notice, pctx.notice)
                memory_flags = pctx.memory_flags
                use_model = pctx.model or use_model
                is_new = session_id != (old_record.session_id if old_record else None)
                active.session_id = turn_result.session_id
                reply = turn_result.reply

            continue_word = (
                self.runtime.config.engine.continuation_triggers[0].capitalize()
                if self.runtime.config.engine.continuation_triggers
                else "Continue"
            )
            notice: str | None = None
            if turn_result and turn_result.partial:
                partial = self.runtime._partial_notice(turn_result, continue_word=continue_word)
                notice = partial if notice is None else f"{notice}\n\n{partial}"
            if rehydrate_notice:
                notice = rehydrate_notice if notice is None else f"{rehydrate_notice}\n\n{notice}"

            latency = time.perf_counter() - turn_start
            turn_metrics = self.runtime._record_turn_metrics(
                chat_id,
                old_record.turn_number + 1 if old_record else 1,
                use_model,
                turn_result.usage if turn_result else None,
                latency,
            )

            mcp_names = self.runtime._active_mcp_server_names(chat_id)
            skill_names = self.runtime._active_skill_names(chat_id)

            with self._lock:
                if is_new:
                    record = self.runtime._create_record(
                        chat_id,
                        session_number,
                        session_id,
                        use_model,
                        reply,
                        memory_flags,
                        label="dispatch continuation",
                    )
                    record.enabled_mcp_servers = mcp_names
                    record.enabled_skills = sorted(skill_names)
                    # Preserve the turn sequence across rehydration.
                    record.turn_number = old_record.turn_number if old_record else 0
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
                if previous_turn_number > 0:
                    previous_updated_at = (
                        old_record.updated_at if is_new and old_record else record.updated_at
                    )
                else:
                    previous_updated_at = 0.0
                record.updated_at = time.time()
                record.cumulative_metrics = self.runtime._per_chat_metrics[chat_id].get(
                    "cumulative", {}
                )
                if not turn_result or not turn_result.partial:
                    record.last_stop_reason = "completed"
                elif active.stopped:
                    record.last_stop_reason = "stopped"
                elif turn_result.cancelled:
                    record.last_stop_reason = "cancelled"
                elif turn_result.stop_reason:
                    record.last_stop_reason = turn_result.stop_reason
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
                # and the partial/timeout notice (deferred by auto-continue).
                turn_notice = rehydrate_notice
                turn_partial = partial
                combined = _join_notices(rehydrate_notice, partial)
                if record_ctx.notice is not None and record_ctx.notice != combined:
                    turn_notice = record_ctx.notice
                    turn_partial = None

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

                # Retain the dispatch result for future recall.
                memory = self.runtime._memory_manager(chat_id)
                memory.retain(
                    content=result,
                    tags=["dispatch"],
                    context=dispatch.context or "continuation",
                )

                transition = self.runtime._check_chat_memory_transition(chat_id, record)
                if transition:
                    turn.notice = _join_notices(turn.notice, transition)

                self.runtime._append_record(record)
                self.runtime._prune_and_compact(chat_id)

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

                dispatch = self.runtime.dispatch_store.complete(dispatch_id, result)
                if dispatch is None:
                    logger.warning("Dispatch %s was removed during continuation", dispatch_id)
                else:
                    self.runtime._plugins.after_dispatch_continue(
                        chat_id,
                        DispatchCompleteContext(
                            chat_id=chat_id,
                            dispatch=dispatch,
                            record=record,
                            result=chat_result,
                        ),
                    )
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
            with self._lock:
                active = self.runtime._active_turns.get(chat_id)
                self.runtime._active_turns.pop(chat_id, None)
                self.runtime._plugins.on_sleeping(chat_id, record, reason="turn_end")
            if active is not None:
                with active._condition:
                    active._condition.notify_all()

        if notify:
            self.runtime._deliver_chat_result(chat_id, chat_result)

        return chat_result
