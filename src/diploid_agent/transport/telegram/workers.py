"""Turn and delivery workers for the Telegram long-polling transport."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from diploid_agent.models import ChatResult
from diploid_agent.transport.command_handler import _coerce_chat_result
from diploid_agent.transport.interactive import (
    extract_ask_block,
)
from diploid_agent.transport.telegram.formatting import (
    _HEARTBEAT_INTERVAL,
    _REPLY_PLACEHOLDER,
    _THINKING_PREFIX,
    _build_heartbeat_text,
    _format_thought,
)
from diploid_agent.transport.telegram.models import ChatInput

if TYPE_CHECKING:
    from diploid_agent.transport.telegram.poller import TelegramPoller

logger = logging.getLogger("telegram_poll")


class TurnWorker(threading.Thread):
    """Run a single turn, stream partial output to Telegram, and support steering."""

    def __init__(
        self,
        poller: TelegramPoller,
        chat_input: ChatInput,
    ):
        super().__init__(daemon=True, name=f"turn-{chat_input.chat_id}")
        self.poller = poller
        self.chat_input = chat_input
        self.chat_id = chat_input.chat_id
        self._should_stop = threading.Event()
        self._running = threading.Event()

    def steer(self, chat_input: ChatInput) -> None:
        """Queue the new input and ask the worker to cancel the current turn."""
        with self.poller._worker_lock:
            self.poller._pending_inputs.setdefault(self.chat_id, deque()).append(chat_input)
        if self.is_alive() and self._running.is_set():
            self.poller._harness_stop(self.chat_id)

    def stop(self) -> None:
        """Cancel the current turn and stop the worker."""
        self._should_stop.set()
        self.poller._harness_stop(self.chat_id)

    def _take_next_message(self) -> ChatInput | None:
        if self._should_stop.is_set():
            return None
        with self.poller._worker_lock:
            queue = self.poller._pending_inputs.get(self.chat_id)
            if queue:
                return queue.popleft()
            return None

    def _harness_chat(self, chat_input: ChatInput) -> dict[str, Any]:
        if self.poller.runtime is not None:
            try:
                result = self.poller.runtime.process(
                    str(self.chat_id),
                    chat_input.text,
                    reply_to=chat_input.reply_to,
                    reply_to_is_bot=chat_input.reply_to_is_bot,
                    reply_to_message_id=chat_input.reply_to_message_id,
                    notify=False,
                )
                return {
                    "reply": getattr(result, "reply", ""),
                    "notice": getattr(result, "notice", None),
                    "session_number": getattr(result, "session_number", None),
                    "turn_number": getattr(result, "turn_number", None),
                    "session_id": getattr(result, "session_id", None),
                    "dispatch_id": getattr(result, "dispatch_id", None),
                    "continuation": getattr(result, "continuation", False),
                }
            except Exception:
                logger.exception("Runtime process failed")
                return {
                    "reply": "Sorry, the runtime is having trouble. Try again in a moment.",
                    "notice": None,
                }

        if self.poller.harness_url is None:
            return {
                "reply": "No runtime or harness URL configured.",
                "notice": None,
            }

        try:
            payload: dict[str, Any] = {
                "chat_id": str(self.chat_id),
                "message": chat_input.text,
            }
            if chat_input.reply_to:
                payload["reply_to"] = chat_input.reply_to
            if chat_input.reply_to_is_bot is not None:
                payload["reply_to_is_bot"] = chat_input.reply_to_is_bot
            if chat_input.reply_to_message_id is not None:
                payload["reply_to_message_id"] = chat_input.reply_to_message_id
            resp = self.poller.client.post(
                f"{self.poller.harness_url}/chat",
                json=payload,
                timeout=self.poller.reply_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("Harness /chat failed")
            return {
                "reply": "Sorry, the harness is having trouble. Try again in a moment.",
                "notice": None,
            }

    def _harness_turn_status(self, wait: float = 0.0) -> dict[str, Any]:
        if self.poller.runtime is not None:
            try:
                return self.poller.runtime.turn_status(str(self.chat_id), wait=wait)
            except Exception:
                logger.exception("Runtime turn_status failed")
                return {"chat_id": str(self.chat_id), "status": "idle"}

        if self.poller.harness_url is None:
            return {"chat_id": str(self.chat_id), "status": "idle"}

        try:
            resp = self.poller.client.get(
                f"{self.poller.harness_url}/turn/{self.chat_id}",
                params={"wait": wait},
                timeout=max(wait + 30.0, 60.0),
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("Harness /turn failed")
            return {"chat_id": str(self.chat_id), "status": "idle"}

    def _send_placeholder(self, text: str) -> int | None:
        return self.poller._send_message(
            self.chat_id,
            text,
            reply_to_message_id=self.chat_input.message_id,
        )

    def _stream_turn(
        self, chat_future: Any, message_id: int | None, thought_id: int | None
    ) -> dict[str, Any]:
        """Long-poll partial status and edit the placeholder(s) until the turn completes.

        When ``intermediate_messages`` is enabled and the streamed reply pauses,
        the current placeholder is committed as a real message and a fresh
        placeholder is started below it. This makes tool-call gaps readable as
        separate Telegram messages instead of one confusing, edited block.
        """
        last_text_sent = ""
        last_thought = ""
        last_thought_sent = ""
        text = ""
        display_text = ""
        visible = ""
        committed_text = ""
        committed_display = ""
        committed_message_id = None
        last_growth_at = turn_start_at = time.monotonic()
        last_edit_at = turn_start_at
        config = self.poller._live_telegram_config

        def _uncommitted_tail(full: str) -> str:
            if not full:
                return ""
            if committed_display and full.startswith(committed_display):
                return full[len(committed_display) :]
            # The model somehow backtracked; restart the commit baseline.
            return full

        def _should_commit(tail: str, idle: float) -> bool:
            if not config.intermediate_messages:
                return False
            if idle < config.intermediate_idle:
                return False
            if len(tail) < config.intermediate_min_chars:
                return False
            stripped = tail.rstrip()
            if not stripped:
                return False
            return stripped[-1] in ".!?\n"

        while not chat_future.done():
            now = time.monotonic()
            remaining = _HEARTBEAT_INTERVAL - (now - last_edit_at)
            idle = now - last_growth_at
            # If the current uncommitted tail is a candidate for an
            # intermediate-message split, wake at the configured idle deadline
            # (not just at the heartbeat). This prevents two separate answer
            # blocks separated by a tool-call gap from being glued into one
            # Telegram message.
            tail = _uncommitted_tail(display_text)
            if _should_commit(tail, idle):
                # Tail is already idle enough; poll very soon to commit it.
                commit_wait = 0.0
            elif (
                config.intermediate_messages
                and len(tail) >= config.intermediate_min_chars
                and (tail.rstrip() and tail.rstrip()[-1] in ".!?\n")
            ):
                commit_wait = max(0.0, config.intermediate_idle - idle)
            else:
                commit_wait = float("inf")
            # Wake for the earlier of the heartbeat deadline and the commit
            # deadline. A 0.5 s floor prevents a tight busy loop when no
            # placeholder can be edited, while still letting us react quickly.
            wait = min(25.0, max(0.5, min(remaining, commit_wait)))
            status = self._harness_turn_status(wait=wait)
            now = time.monotonic()
            running = status.get("status") == "running"
            if running:
                text = status.get("message_text", "")
                display_text, _ = extract_ask_block(text or "")
                display_text = display_text.strip()
                visible = display_text[:4096]
            else:
                text = ""
                display_text = ""
                visible = ""
            edited = False

            # Start a reply placeholder the moment text starts arriving, even
            # when thought streaming is still active.
            if message_id is None and display_text:
                message_id = self._send_placeholder(_REPLY_PLACEHOLDER)
                if message_id is not None:
                    self.poller._save_placeholder_state(self.chat_id, message_id, thought_id)
                last_text_sent = _REPLY_PLACEHOLDER

            if message_id is not None and display_text:
                # If the visible text changed, the model is still writing.
                if visible != last_text_sent:
                    self.poller._edit_message_text(self.chat_id, message_id, visible)
                    last_text_sent = visible
                    last_growth_at = now
                    edited = True
                else:
                    # No new text: check whether the visible tail we already
                    # showed has been sitting idle long enough to be its own
                    # message. We use the displayed text (not the raw text with
                    # hidden ask blocks) so a trailing ask block does not cause
                    # a duplicate commit of the same visible content.
                    tail = _uncommitted_tail(display_text)
                    if _should_commit(tail, now - last_growth_at):
                        # Freeze the current placeholder as a sent message and
                        # start a fresh one so the rest of the reply can stream
                        # below it.
                        committed_text = text
                        committed_display = visible
                        committed_message_id = message_id
                        last_text_sent = _REPLY_PLACEHOLDER
                        last_growth_at = now
                        message_id = self._send_placeholder(_REPLY_PLACEHOLDER)
                        if message_id is not None:
                            self.poller._save_placeholder_state(
                                self.chat_id, message_id, thought_id
                            )
                        edited = True

            if thought_id is not None:
                thought = status.get("thought_text", "")
                if thought:
                    visible = _format_thought(thought)
                    if visible and visible != last_thought_sent:
                        self.poller._edit_message_text(self.chat_id, thought_id, visible)
                        last_thought_sent = visible
                        edited = True
                    last_thought = thought
            if edited:
                last_edit_at = now
            elif now - last_edit_at >= _HEARTBEAT_INTERVAL:
                # Nothing new from the model; nudge the placeholder so the user
                # knows the harness is still alive.
                elapsed = now - turn_start_at
                if message_id is not None:
                    base = display_text[:4096] if display_text else _REPLY_PLACEHOLDER
                    heartbeat = _build_heartbeat_text(base, elapsed)
                    if heartbeat != last_text_sent:
                        self.poller._edit_message_text(self.chat_id, message_id, heartbeat)
                        last_text_sent = heartbeat
                        edited = True
                if thought_id is not None:
                    base = _format_thought(last_thought) if last_thought else _THINKING_PREFIX
                    heartbeat = _build_heartbeat_text(base, elapsed)
                    if heartbeat != last_thought_sent:
                        self.poller._edit_message_text(self.chat_id, thought_id, heartbeat)
                        last_thought_sent = heartbeat
                        edited = True
                # Reset the timer even if we had no placeholder to update, so a
                # failed sendMessage cannot turn this loop into a tight poll.
                last_edit_at = now

        try:
            result = chat_future.result()
        except Exception:
            logger.exception("Turn failed")
            result = {
                "reply": "Sorry, the harness is having trouble. Try again in a moment.",
                "notice": None,
            }

        continuation = result.get("continuation", False)

        if not continuation:
            # Finalise the thought block (when enabled). The placeholder has already
            # been live-edited with the latest visible text, so we only touch it if
            # the final thought is empty (delete) or short enough to fit in one
            # message without splitting. Long thoughts stay as they are, which avoids
            # flooding Telegram with multi-part edits.
            if thought_id is not None:
                # Once the future completes the harness may have already popped the
                # active turn, so a fresh /turn call can return idle with no
                # thought_text. Use the last captured thought as the fallback.
                final_status = self._harness_turn_status()
                thought = final_status.get("thought_text") or last_thought
                if not thought:
                    self.poller._delete_message(self.chat_id, thought_id)
                else:
                    visible = _format_thought(thought)
                    if visible and visible != last_thought_sent:
                        self.poller._edit_message_text(self.chat_id, thought_id, visible)
                thought_id = None

            # The final placeholder is only created after thinking completes, so it
            # is always below the thought block.
            if message_id is None:
                message_id = self._send_placeholder("...")
                if message_id is not None:
                    self.poller._save_placeholder_state(self.chat_id, message_id, thought_id)

            # Replace the placeholder with the final reply. If we already committed
            # an earlier chunk as its own message, send only the uncommitted suffix
            # so the user does not see the same text twice.
            reply = result.get("reply", "")
            display_reply, _ = extract_ask_block(reply)
            display_reply = display_reply.strip()
            if committed_text and reply.startswith(committed_text):
                # The raw final reply still contains the already-committed text;
                # strip the raw prefix so the suffix (which may include a trailing
                # ask block for the keyboard) is sent below the committed message.
                reply = reply[len(committed_text) :].lstrip("\n")
            elif committed_display and display_reply.startswith(committed_display):
                # The visible prefix was already committed, but the raw reply was
                # transformed (e.g. the ask block was stripped). Send only the
                # visible suffix so the committed message is not duplicated.
                reply = display_reply[len(committed_display) :].lstrip("\n")
            if not reply or not reply.strip():
                # If the turn produced no final text, do not leave the placeholder
                # hanging. Delete it and send the notice (if any) as a fresh message.
                if message_id is not None:
                    self.poller._delete_message(self.chat_id, message_id)
                sent = []
            elif message_id is not None:
                sent = self.poller._send_text(
                    self.chat_id,
                    reply,
                    first_message_id=message_id,
                    reply_to_message_id=self.chat_input.message_id,
                )
            else:
                sent = self.poller._send_text(
                    self.chat_id,
                    reply,
                    reply_to_message_id=self.chat_input.message_id,
                )

            session_number = result.get("session_number")
            turn_number = result.get("turn_number")
            if sent and session_number is not None and turn_number is not None:
                self.poller._register_message_ids(
                    self.chat_id, sent, session_number, turn_number, reply, kind="reply"
                )

            notice = result.get("notice")
            if notice:
                self.poller._send_text(
                    self.chat_id,
                    f"System: {notice}",
                    reply_to_message_id=self.chat_input.message_id,
                )
        else:
            if committed_message_id is not None and committed_message_id != message_id:
                self.poller._delete_message(self.chat_id, committed_message_id)

        return result

    def _run_turn(self, chat_input: ChatInput) -> dict[str, Any]:
        """Run one turn with streaming.

        The thought placeholder is sent first (when enabled). The final reply
        placeholder is only created after the thought stream completes, so the
        final reply always ends up below the thought block.
        """
        thought_id: int | None = None
        message_id: int | None = None
        if self.poller._stream_thoughts_enabled(self.chat_id):
            thought_id = self._send_placeholder("Thinking...")
        else:
            message_id = self._send_placeholder("...")

        self.poller._save_placeholder_state(self.chat_id, message_id, thought_id)

        result: dict[str, Any] = {}
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                chat_future = pool.submit(self._harness_chat, chat_input)
                try:
                    with self.poller._typing_context(self.chat_id):
                        result = self._stream_turn(chat_future, message_id, thought_id)
                except Exception:
                    logger.exception("Streaming failed")
                    if not chat_future.done():
                        chat_future.cancel()
                    raise
        finally:
            if not result.get("continuation"):
                self.poller._remove_placeholder_state(self.chat_id)

        return result

    def run(self) -> None:
        chat_input = self.chat_input
        self._running.set()
        try:
            while chat_input:
                self._run_turn(chat_input)
                chat_input = self._take_next_message()
        finally:
            self._running.clear()
            with self.poller._worker_lock:
                active = self.poller._active_workers.get(self.chat_id)
                if active is self:
                    self.poller._active_workers.pop(self.chat_id, None)
                queue = self.poller._pending_inputs.get(self.chat_id)
                if queue and not self.poller._active_workers.get(self.chat_id):
                    next_input = queue.popleft()
                    worker = TurnWorker(self.poller, next_input)
                    self.poller._active_workers[self.chat_id] = worker
                    worker.start()
            self.poller._close_client()


class DeliveryWorker(threading.Thread):
    """Long-poll the runtime outbox and deliver ChatResults to Telegram."""

    _POLL_WAIT = 5.0

    def __init__(self, poller: TelegramPoller, chat_id: int) -> None:
        super().__init__(daemon=True, name=f"delivery-{chat_id}")
        self.poller = poller
        self.chat_id = chat_id
        self._should_stop = threading.Event()

    def stop(self) -> None:
        self._should_stop.set()

    def _fetch_outbox(self) -> ChatResult | None:
        raw = self.poller.command_handler.call(
            method="outbox_pop",
            chat_id=self.chat_id,
            http_path="/outbox/{chat_id}",
            http_method="GET",
            wait=self._POLL_WAIT,
        )
        if raw is None:
            return None
        if isinstance(raw, ChatResult):
            return raw
        if isinstance(raw, dict):
            if "error" in raw:
                return None
            result = raw.get("result")
            if result is None:
                return None
            if isinstance(result, dict):
                return _coerce_chat_result(result)
            if isinstance(result, ChatResult):
                return result
        return _coerce_chat_result(raw)

    def run(self) -> None:
        while not self._should_stop.is_set() and not self.poller._stop.is_set():
            try:
                chat_result = self._fetch_outbox()
                if chat_result is None:
                    time.sleep(self._POLL_WAIT)
                    continue
                self.poller._deliver_outbox_result(self.chat_id, chat_result)
            except Exception:
                logger.exception("DeliveryWorker error for chat %s", self.chat_id)
                time.sleep(self._POLL_WAIT)
