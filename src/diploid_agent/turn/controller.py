"""TurnController: per-chat turn and session logic."""

from __future__ import annotations

import functools
import logging
import time
from typing import TYPE_CHECKING, Any

from diploid_agent.acp_client import AcpTransportError
from diploid_agent.models import ActiveTurn, ChatResult, WakeEvent
from diploid_agent.turn.dispatch import TurnDispatch
from diploid_agent.turn.process import TurnProcess
from diploid_agent.turn.rehydrate import TurnRehydrate
from diploid_agent.turn.session import TurnSession

if TYPE_CHECKING:
    from diploid_agent.runtime.agent_runtime import AgentRuntime

logger = logging.getLogger(__name__)


def _locked(method):
    """Run a TurnController method under the runtime RLock."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.runtime._lock:
            return method(self, *args, **kwargs)

    return wrapper


class TurnController:
    """Per-chat turn and session orchestration."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self.session = TurnSession(self)
        self.rehydrate = TurnRehydrate(self)
        self._dispatch = TurnDispatch(self)
        self._process = TurnProcess(self)

    def _has_pending_continuation(self, chat_id: str, wake_event: WakeEvent | None = None) -> bool:
        """Return True if another auto-continue wake is pending for this chat."""
        return self._process._has_pending_continuation(chat_id, wake_event)

    def _seed_active_turn(self, active: ActiveTurn, wake_event: WakeEvent | None) -> None:
        """Pre-populate the active turn with content from a previous partial turn."""
        self._process._seed_active_turn(active, wake_event)

    def _rehydrate(self, *args: Any, **kwargs: Any) -> Any:
        """Resume a persisted ACP session if possible, otherwise start a new one."""
        return self.rehydrate._rehydrate(*args, **kwargs)

    def _can_resume_record(self, *args: Any, **kwargs: Any) -> Any:
        """Return True if the ACP session for this record can be resumed."""
        return self.session._can_resume_record(*args, **kwargs)

    def process(self, *args, **kwargs) -> ChatResult:
        """Send a message to the persona session and return the reply."""
        return self._process.process(*args, **kwargs)

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
