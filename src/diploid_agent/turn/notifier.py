"""Notifier helpers for streaming turns and outbox liveness."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from diploid_agent.models import ActiveTurn, ChatResult, WakeEvent

if TYPE_CHECKING:
    from diploid_agent.runtime.agent_runtime import AgentRuntime
    from diploid_agent.turn.controller import TurnController


def _format_elapsed_short(seconds: float) -> str:
    """Return a short, human-readable elapsed time."""
    total = int(seconds)
    mins, secs = divmod(total, 60)
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


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
