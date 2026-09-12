"""TurnDispatch: background dispatch and continue-turn logic."""

from __future__ import annotations

import logging
import time

from diploid_agent.dispatch import DispatchStatus
from diploid_agent.models import (
    ActiveTurn,
    ChatResult,
    SessionRecord,
    WakeEvent,
)
from diploid_agent.plugins.contexts import (
    DispatchCompleteContext,
    DispatchContinueContext,
    DispatchCreateContext,
    TurnStartContext,
)
from diploid_agent.turn.pipeline import TurnPipeline
from diploid_agent.turn.stream import TurnStream

logger = logging.getLogger(__name__)


class TurnDispatch(TurnPipeline):
    """Background dispatch / continue-turn logic for a single chat."""

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
            if self.runtime._restart_draining.is_set():
                return ChatResult(
                    reply="The service is draining for a restart; continuation deferred.",
                    notice=f"dispatch:{dispatch_id}",
                )
            if chat_id in self.runtime._active_turns:
                return ChatResult(
                    reply="A turn is already in progress; continuation queued.",
                    notice=f"dispatch:{dispatch_id}",
                )

            record = self.runtime._active_record(chat_id)
            if record is None:
                return ChatResult(reply="No active session for this chat.")

            use_model = self.runtime._prompts._model(record)
            user_message = "Continue"
            route = self.runtime._prompts.resolve_model(chat_id, user_message, record)
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

            self.runtime.skills.refresh_to_chat(
                chat_id,
                self.runtime._chat_dir(chat_id),
                set(record.enabled_skills or []),
            )
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

            # Reserve a turn number up front and persist it.
            previous_updated_at = record.updated_at if record.turn_number > 0 else 0.0
            turn_number = record.reserve_turn_number()
            self.runtime._append_record(record)

        stream = TurnStream(self.runtime, chat_id)

        try:
            outcome = self._call_engine(
                chat_id=chat_id,
                user_message=user_message,
                prompt=prompt,
                use_model=use_model,
                record=record,
                old_record=old_record,
                is_new=is_new,
                force_new_session=False,
                active=active,
                stream=stream,
                memory_flags=memory_flags,
                rehydrate_kwargs={"continuation_anchor": continuation_anchor},
            )
            if isinstance(outcome, ChatResult):
                return outcome

            def _retain_dispatch() -> None:
                # Retain the dispatch result for future recall.
                self.runtime._memory_manager(chat_id).retain(
                    content=result,
                    tags=["dispatch"],
                    context=dispatch.context or "continuation",
                )

            def _after_result(cr: ChatResult, rec: SessionRecord) -> None:
                completed = self.runtime.dispatch_store.complete(dispatch_id, result)
                if completed is None:
                    logger.warning("Dispatch %s was removed during continuation", dispatch_id)
                else:
                    self.runtime._plugins.after_dispatch_continue(
                        chat_id,
                        DispatchCompleteContext(
                            chat_id=chat_id,
                            dispatch=completed,
                            record=rec,
                            result=cr,
                        ),
                    )

            continue_word = (
                self.runtime.config.engine.continuation_triggers[0].capitalize()
                if self.runtime.config.engine.continuation_triggers
                else "Continue"
            )
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
                budget_notice=budget_notice,
                label="dispatch continuation",
                after_turn_end=_retain_dispatch,
                after_append=lambda: self.runtime._prune_and_compact(chat_id),
                after_result=_after_result,
            )
        except Exception as exc:
            self._emit_turn_error(chat_id, record, user_message, exc)
            raise
        finally:
            self._cleanup_turn(chat_id, record)

        if notify:
            self.runtime._deliver_chat_result(chat_id, chat_result)

        return chat_result
