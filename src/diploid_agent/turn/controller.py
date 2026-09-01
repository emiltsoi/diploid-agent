"""TurnController: per-chat turn and session logic."""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from diploid_agent.acp_client import AcpTransportError
from diploid_agent.engine import TurnRequest, TurnResult
from diploid_agent.models import ActiveTurn, ChatResult, PartialTurn, SessionRecord, WakeEvent
from diploid_agent.plugins.base import TurnInfo
from diploid_agent.plugins.contexts import (
    EngineCallContext,
    EngineResultContext,
    RecordTurnContext,
    TurnErrorContext,
    TurnStartContext,
)
from diploid_agent.turn.dispatch import TurnDispatch
from diploid_agent.turn.rehydrate import TurnRehydrate
from diploid_agent.turn.session import TurnSession

if TYPE_CHECKING:
    from diploid_agent.runtime.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)

# Cap the in-memory thought stream so a runaway model cannot exhaust memory
# or produce giant wake/auto-continue payloads and Telegram status messages.
_MAX_THOUGHT_TEXT_CHARS = 20000


def _join_notices(*parts: str | None) -> str | None:
    """Concatenate non-empty notice strings with a blank line between them."""
    joined = "\n\n".join(p for p in parts if p)
    return joined or None


def _format_elapsed_short(seconds: float) -> str:
    """Return a short, human-readable elapsed time."""
    total = int(seconds)
    mins, secs = divmod(total, 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def _locked(method):
    """Run a TurnController method under the runtime RLock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.runtime._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _run_unlocked(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run a method while releasing the runtime RLock during its execution."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        self.runtime._lock.release()
        try:
            return method(self, *args, **kwargs)
        finally:
            self.runtime._lock.acquire()

    return wrapper


class _NotifyStream:
    """Stream a turn to a Telegram-like notifier when ``notify=True``.

    This is used for turns that are not driven by the Telegram poller's
    ``TurnWorker`` -- for example, an ``auto_continue`` wake or a dispatch
    continuation.  It creates a reply placeholder (and a thought placeholder
    when ``stream_thoughts`` is enabled), keeps the typing indicator alive,
    and edits the placeholders as the turn produces output.
    """

    def __init__(
        self,
        turn_controller: TurnController,
        chat_id: str,
        notify: bool,
        stream_thoughts: bool,
        min_edit_interval: float,
        wake_event: WakeEvent | None = None,
    ) -> None:
        self.turn_controller = turn_controller
        self.chat_id = chat_id
        self.notify = notify
        self.notifier = turn_controller.runtime.notifier
        self.stream_thoughts = stream_thoughts
        self.min_edit_interval = min_edit_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_edit = 0.0
        self._lock = threading.Lock()

        payload = wake_event.payload if wake_event and isinstance(wake_event.payload, dict) else {}
        self._reply_id: int | None = payload.get("message_id")
        self._thought_id: int | None = payload.get("thought_id")
        self._last_text: str = payload.get("message_text") or ""
        self._last_thought: str = payload.get("thought_text") or ""

    def start(self) -> None:
        if not self.notify:
            return
        if not hasattr(self.notifier, "send_placeholder"):
            return

        if self._reply_id is None:
            self._reply_id = self.notifier.send_placeholder(self.chat_id, "...")
        if self._reply_id is None:
            return

        if (
            self.stream_thoughts
            and self._thought_id is None
            and hasattr(self.notifier, "send_placeholder")
        ):
            self._thought_id = self.notifier.send_placeholder(self.chat_id, "Thinking...")
        if hasattr(self.notifier, "begin_typing"):
            self.notifier.begin_typing(self.chat_id)
        self._last_edit = time.monotonic()
        self._thread = threading.Thread(
            target=self._stream_loop,
            daemon=True,
            name=f"notify-stream-{self.chat_id}",
        )
        self._thread.start()

    def _stream_loop(self) -> None:
        while not self._stop.is_set():
            status = self.turn_controller.turn_status(self.chat_id, wait=0.2)
            if self._stop.is_set():
                break
            if status.get("status") != "running":
                break
            self._update(status)

    def _update(self, status: dict[str, Any]) -> None:
        with self._lock:
            now = time.monotonic()
            if now - self._last_edit < self.min_edit_interval:
                return
            message_text = status.get("message_text", "")
            thought_text = status.get("thought_text", "")
            edited = False
            if (
                message_text
                and message_text != self._last_text
                and self._reply_id is not None
                and hasattr(self.notifier, "edit_message")
            ):
                self.notifier.edit_message(self.chat_id, self._reply_id, message_text)
                self._last_text = message_text
                edited = True
            if (
                thought_text
                and thought_text != self._last_thought
                and self._thought_id is not None
                and hasattr(self.notifier, "edit_message")
            ):
                self.notifier.edit_message(self.chat_id, self._thought_id, thought_text)
                self._last_thought = thought_text
                edited = True
            if edited:
                self._last_edit = now

    def finish(self, chat_result: ChatResult) -> list[Any]:
        if not self.notify or self._reply_id is None:
            if self.notify:
                sent = self.notifier.send(self.chat_id, chat_result.reply or "")
                return [sent] if sent is not None else []
            return []

        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

        if hasattr(self.notifier, "end_typing"):
            self.notifier.end_typing(self.chat_id)

        # If another continuation is already scheduled, keep the placeholder
        # alive so the next _NotifyStream can reuse it.
        if chat_result.continuation:
            return [self._reply_id] if self._reply_id is not None else []

        sent: list[Any] = []
        with self._lock:
            if chat_result.reply:
                if self._reply_id is not None and self.notifier.edit_message(
                    self.chat_id, self._reply_id, chat_result.reply
                ):
                    sent.append(self._reply_id)
                else:
                    fallback = self.notifier.send(self.chat_id, chat_result.reply)
                    if fallback is not None:
                        sent.append(fallback)
            else:
                if self._reply_id is not None and hasattr(self.notifier, "delete_message"):
                    self.notifier.delete_message(self.chat_id, self._reply_id)

            if self._thought_id is not None and hasattr(self.notifier, "edit_message"):
                if self._last_thought:
                    self.notifier.edit_message(self.chat_id, self._thought_id, self._last_thought)
                elif hasattr(self.notifier, "delete_message"):
                    self.notifier.delete_message(self.chat_id, self._thought_id)

        if chat_result.notice:
            notice_id = self.notifier.send(self.chat_id, f"System: {chat_result.notice}")
            if notice_id is not None:
                sent.append(notice_id)

        return sent


class _OutboxHeartbeat:
    """Send periodic liveness notices to the outbox for long non-streaming turns.

    When a turn is driven by the wake queue (or another caller that does not
    have its own streaming transport), the user gets no feedback while the model
    is thinking. This heartbeat pushes a short ``ChatResult`` to the outbox
    every so often so the user knows the agent is still alive and can decide
    whether to wait or send ``/stop``.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        chat_id: str,
        active: ActiveTurn,
        first_beat: float = 30.0,
        beat_interval: float = 90.0,
    ) -> None:
        self.runtime = runtime
        self.chat_id = chat_id
        self.active = active
        self.first_beat = first_beat
        self.beat_interval = beat_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"outbox-heartbeat-{self.chat_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        if self._stop.wait(timeout=self.first_beat):
            return
        while not self._stop.is_set():
            with self.runtime._lock:
                current = self.runtime._active_turns.get(self.chat_id)
                # Read message_text under the same lock that _on_chunk and
                # _on_update use to update it.
                no_visible_text = current is self.active and not self.active.message_text
            if current is not self.active or current is None or current.stopped:
                break

            # Only nudge when the model has not produced any visible reply yet.
            # If it has produced partial text, the user can already see progress
            # in the final message when it arrives.
            if no_visible_text:
                elapsed = time.time() - self.active.start_time
                elapsed_str = _format_elapsed_short(elapsed)
                chat_result = ChatResult(
                    reply=f"⏳ Still thinking... ({elapsed_str})",
                    notice="Send /stop to cancel this turn if you don't want to wait.",
                )
                self.runtime._enqueue_outbox(self.chat_id, chat_result)

            if self._stop.wait(timeout=self.beat_interval):
                break


class TurnController:
    """Per-chat turn and session orchestration."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self.session = TurnSession(self)
        self.rehydrate = TurnRehydrate(self)
        self._dispatch = TurnDispatch(self)

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
        active.message_text = message_text[-_MAX_THOUGHT_TEXT_CHARS:]
        active.thought_text = thought_text[-_MAX_THOUGHT_TEXT_CHARS:]

    def _rehydrate(self, *args: Any, **kwargs: Any) -> Any:
        """Resume a persisted ACP session if possible, otherwise start a new one."""
        return self.rehydrate._rehydrate(*args, **kwargs)

    def _can_resume_record(self, *args: Any, **kwargs: Any) -> Any:
        """Return True if the ACP session for this record can be resumed."""
        return self.session._can_resume_record(*args, **kwargs)

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

            if record is None or model_changed or hard_timeout_before or skills_changed:
                if record and (model_changed or hard_timeout_before or skills_changed):
                    self.runtime._archive_active_session(chat_id, record)
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
                )
                prompt = pctx.prompt
                use_model = pctx.model or use_model
                active = ActiveTurn(chat_id, record.session_id, user_message, time.time())
                self._seed_active_turn(active, wake_event)
                self.runtime._active_turns[chat_id] = active
                is_new = False
                session_number = record.session_number
                old_record = record

        rehydrate_notice: str | None = None

        telegram_config = self.runtime.config.harness.telegram
        notifier_stream = _NotifyStream(
            self,
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
                    updated_at=time.time(),
                ),
            )

        def _on_chunk(text: str) -> None:
            with self.runtime._lock:
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
            with self.runtime._lock:
                a = self.runtime._active_turns.get(chat_id)
                if a:
                    a.thought_text += text
                    if len(a.thought_text) > _MAX_THOUGHT_TEXT_CHARS:
                        a.thought_text = a.thought_text[-_MAX_THOUGHT_TEXT_CHARS:]
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
                    mcp_servers=self.runtime._active_mcp_servers(chat_id) if is_new else None,
                    soft_timeout=self.runtime.config.engine.soft_timeout,
                )
                call_ctx = self.runtime._plugins.before_engine_call(
                    chat_id,
                    EngineCallContext(
                        chat_id=chat_id,
                        request=request,
                        session_id=None if is_new else old_record.session_id,
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
                    if is_new:
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

                ret = self._rehydrate(
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
                reply = result.reply

            # If a prompt came back with a genuinely empty reply,
            # the ACP session is almost certainly stale. Rehydrate once.
            if not reply and not result.partial:
                empty_id = session_id or (old_record.session_id if old_record else "unknown")
                logger.warning(
                    "ACP session %s returned an empty reply; rehydrating for %s",
                    empty_id,
                    chat_id,
                )
                ret = self._rehydrate(
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
                reply = result.reply

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

    @_locked
    def dispatch(
        self,
        chat_id: str,
        context: str | None = None,
    ) -> ChatResult:
        """Register a new dispatch for this chat and return its id."""
        return self._dispatch.dispatch(chat_id, context)

    def continue_turn(
        self,
        dispatch_id: str,
        result: str,
        *,
        notify: bool = True,
    ) -> ChatResult:
        """Resume the ACP session after a background dispatch completes."""
        return self._dispatch.continue_turn(dispatch_id, result, notify=notify)

    def _active_turn_status(self, active: ActiveTurn) -> dict[str, Any]:
        return {
            "chat_id": active.chat_id,
            "status": "running",
            "session_id": active.session_id,
            "user_message": active.user_message,
            "message_text": active.message_text,
            "thought_text": active.thought_text,
            "stopped": active.stopped,
            "start_time": active.start_time,
            "elapsed_seconds": round(time.time() - active.start_time, 1),
        }

    def turn_status(self, chat_id: str, wait: float = 0.0) -> dict[str, Any]:
        """Return the current streaming state of an active turn.

        If ``wait`` is greater than 0, block until the turn state changes or the
        timeout expires.  This is used by transports to long-poll /turn.
        """
        with self.runtime._lock:
            active = self.runtime._active_turns.get(chat_id)
        if active is None:
            return {"chat_id": chat_id, "status": "idle"}
        if wait <= 0:
            return self._active_turn_status(active)

        snapshot_message = active.message_text
        snapshot_thought = active.thought_text
        with active._condition:
            active._condition.wait_for(
                lambda: (
                    active.stopped
                    or active.message_text != snapshot_message
                    or active.thought_text != snapshot_thought
                ),
                timeout=wait,
            )

        with self.runtime._lock:
            active = self.runtime._active_turns.get(chat_id)
        if active is None:
            return {"chat_id": chat_id, "status": "idle"}
        return self._active_turn_status(active)

    def stop(self, chat_id: str) -> ChatResult:
        """Cancel an in-flight ACP turn for a chat and drain pending auto-continues."""
        with self.runtime._lock:
            active = self.runtime._active_turns.get(chat_id)
        if active is None:
            return ChatResult(
                reply="No running turn to stop.",
                notice="The agent is not currently working on a reply for this chat.",
            )

        # ActiveTurn.session_id is only set after the ACP call returns, so it
        # is None while a new-session prompt is in flight. Fall back to the
        # active record and then to whatever session the engine is currently
        # prompting, so /stop can cancel even during the long-lived call.
        record = self.runtime._active_record(chat_id)
        session_id = (
            active.session_id
            or (record.session_id if record is not None else None)
            or self.runtime.engine.active_session_id()
        )
        if session_id is None:
            # Nothing to cancel, but still wake the active turn if it is waiting.
            with active._condition:
                active.stopped = True
                active._condition.notify_all()
            return ChatResult(
                reply="No running turn to stop.",
                notice="The agent is not currently working on a reply for this chat.",
            )

        active.session_id = session_id
        with active._condition:
            active.stopped = True
            active._condition.notify_all()
        self.runtime.engine.cancel(session_id)
        if self.runtime.wake_queue is not None:
            count = self.runtime.wake_queue.cancel(chat_id=chat_id, reason="auto_continue")
            if count:
                logger.info("Cancelled %d pending auto-continue wake(s) for %s", count, chat_id)
        return ChatResult(
            reply="Stopping the current turn...",
            notice="The agent will return a partial summary when it aborts.",
        )

    @_locked
    def restart(self, chat_id: str) -> ChatResult:
        """Kill the ACP child and start a fresh transport."""
        with self.runtime._lock:
            active = self.runtime._active_turns.get(chat_id)

        if active is not None:
            with active._condition:
                active.stopped = True
                active._condition.notify_all()

        if self.runtime.wake_queue is not None:
            count = self.runtime.wake_queue.cancel(chat_id=chat_id, reason="auto_continue")
            if count:
                logger.info("Cancelled %d pending auto-continue wake(s) for %s", count, chat_id)

        try:
            self.runtime.engine.restart()
        except AcpTransportError as exc:
            return ChatResult(
                reply="Could not restart the ACP transport.",
                notice=str(exc),
            )
        except Exception:
            logger.exception("Failed to restart ACP transport for %s", chat_id)
            return ChatResult(
                reply="Could not restart the ACP transport.",
                notice="See logs for details.",
            )

        return ChatResult(
            reply="ACP transport restarted.",
            notice="The next message will start a fresh Devin session.",
        )

    def _finalize_session_activation(self, *args: Any, **kwargs: Any) -> Any:
        """Record the turn, append the record, notify plugins, and return the result."""
        return self.session._finalize_session_activation(*args, **kwargs)

    def _start_fresh_session(self, *args: Any, **kwargs: Any) -> Any:
        """Archive any active session and start a fresh ACP session for the chat."""
        return self.session._start_fresh_session(*args, **kwargs)

    def switch_model(self, *args: Any, **kwargs: Any) -> Any:
        """Switch the model for a chat by starting a fresh Devin session."""
        return self.session.switch_model(*args, **kwargs)

    def new_session(self, *args: Any, **kwargs: Any) -> Any:
        """Start a fresh ACP session for a chat, clearing the active context."""
        return self.session.new_session(*args, **kwargs)

    def resume_session(self, *args: Any, **kwargs: Any) -> Any:
        """Resume an archived session as the active one."""
        return self.session.resume_session(*args, **kwargs)

    def branch_session(self, *args: Any, **kwargs: Any) -> Any:
        """Branch from an archived session, creating a new active session."""
        return self.session.branch_session(*args, **kwargs)
