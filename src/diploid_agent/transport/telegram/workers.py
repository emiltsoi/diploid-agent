"""Turn and delivery workers for the Telegram long-polling transport."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from diploid_agent.models import ChatResult
from diploid_agent.runtime.outbox import _is_telegram_chat_id
from diploid_agent.transport.command_handler import _coerce_chat_result
from diploid_agent.transport.telegram.models import ChatInput
from diploid_agent.transport.telegram.stream_display import StreamDisplay

if TYPE_CHECKING:
    from diploid_agent.transport.telegram.poller import TelegramPoller

logger = logging.getLogger("telegram_poll")

# Minimum wall-clock interval between two /turn status polls. The `wait`
# parameter only asks the harness to hold the request server-side; when the
# response comes back immediately (idle, stopped, error), this floor is the
# only thing preventing a hot poll loop.
_MIN_POLL_INTERVAL = 0.5


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
        display = StreamDisplay(
            poller=self.poller,
            chat_id=self.chat_id,
            reply_to_message_id=self.chat_input.message_id,
            config=self.poller._live_telegram_config,
            message_id=message_id,
            thought_id=thought_id,
        )
        while not chat_future.done() and not self._should_stop.is_set():
            wait = display.next_wait()
            poll_started = time.monotonic()
            status = self._harness_turn_status(wait=wait)
            # `wait` is only a server-side long-poll hint: when the harness
            # returns instantly (idle turn, `stopped` already set, unreachable
            # server), nothing paces this loop and it would spin at network
            # speed. Enforce a real floor on the poll rate.
            poll_elapsed = time.monotonic() - poll_started
            if poll_elapsed < _MIN_POLL_INTERVAL:
                time.sleep(_MIN_POLL_INTERVAL - poll_elapsed)
            display.update(status)

        result = display.await_result(chat_future, stopped=self._should_stop.is_set())
        display.finalize(result)
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
        pool = ThreadPoolExecutor(max_workers=1)
        try:
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
            # A user-stopped turn may leave the /chat HTTP request in flight;
            # do not park the worker thread on pool shutdown waiting for it.
            pool.shutdown(wait=not self._should_stop.is_set())
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
    """Long-poll the runtime outbox and deliver ChatResults to Telegram.

    A worker may be scoped to one chat or, when ``chat_id`` is None, act as a
    global outbox consumer that pulls the next item for *any* chat and starts
    per-chat delivery on demand. The global worker is the default starting with
    this harness; per-chat workers remain for tests and callers that need them.
    """

    _POLL_WAIT = 5.0

    def __init__(self, poller: TelegramPoller, chat_id: int | None = None) -> None:
        name = "delivery-global" if chat_id is None else f"delivery-{chat_id}"
        super().__init__(daemon=True, name=name)
        self.poller = poller
        self.chat_id = chat_id
        self._next_chat_id: int | None = None
        self._should_stop = threading.Event()

    def stop(self) -> None:
        self._should_stop.set()

    def _fetch_outbox(self) -> ChatResult | None:
        """Poll the outbox and return the next ChatResult (or None if empty)."""
        if self.chat_id is not None:
            raw = self.poller.command_handler.call(
                method="outbox_pop",
                chat_id=self.chat_id,
                http_path="/outbox/{chat_id}",
                http_method="GET",
                wait=self._POLL_WAIT,
            )
        else:
            self._next_chat_id = None
            raw = self.poller.command_handler.call(
                method="outbox_pop",
                http_path="/outbox",
                http_method="GET",
                wait=self._POLL_WAIT,
                requires_chat_id=False,
                return_chat_id=True,
            )
            if raw is None:
                return None
            if isinstance(raw, tuple):
                chat_id = raw[0]
                if chat_id is None or not _is_telegram_chat_id(str(chat_id)):
                    logger.debug("Skipping non-Telegram outbox item for chat %s", chat_id)
                    return None
                self._next_chat_id = int(str(chat_id))
                return raw[1]
            if isinstance(raw, dict):
                if "error" in raw:
                    return None
                chat_id = raw.get("chat_id")
                if chat_id is None or not _is_telegram_chat_id(str(chat_id)):
                    logger.debug("Skipping non-Telegram outbox item for chat %s", chat_id)
                    return None
                self._next_chat_id = int(str(chat_id))
                result = raw.get("result")
                if result is None:
                    return None
                if isinstance(result, dict):
                    return _coerce_chat_result(result)
                if isinstance(result, ChatResult):
                    return result
                return _coerce_chat_result(result)

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
                if self.chat_id is not None:
                    self.poller._deliver_outbox_result(self.chat_id, chat_result)
                else:
                    chat_id = self._next_chat_id
                    if chat_id is None:
                        time.sleep(self._POLL_WAIT)
                        continue
                    self.poller._deliver_outbox_result(chat_id, chat_result)
            except Exception:
                logger.exception("DeliveryWorker error for chat %s", self.chat_id)
                time.sleep(self._POLL_WAIT)
