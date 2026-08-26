#!/usr/bin/env python3
"""Telegram long-polling transport.

Polls Telegram Bot API `getUpdates` and forwards each text message to the
configured runtime or harness `/chat` endpoint, then sends the reply back to
the user.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from diploid_agent.config import (
    Config,
    ConfigPersistenceError,
    NotificationsConfig,
    TaskConfig,
    TelegramConfig,
    TimerConfig,
    WakerConfig,
)
from diploid_agent.metrics import MetricsCollector
from diploid_agent.transport.base import (
    OutboundMessage,
    RuntimeAPI,
    Transport,
)
from diploid_agent.transport.telegram_format import (
    _prefix_within_utf16_limit,
    _strip_mdv2,
    format_markdown_v2,
    separate_chunk_indicator_from_fence,
    split_telegram_text,
    utf16_len,
)

# The Telegram token is part of the request URL, so suppress httpx's default
# request logging to avoid leaking it.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("telegram_poll")


_THINKING_PREFIX = "Thinking..."
_THINKING_CONTINUED = "... (thinking continues)"
_HEARTBEAT_INTERVAL = 30.0
_REPLY_PLACEHOLDER = "..."


def _format_thought(thought: str, limit: int = 4096) -> str:
    """Return a Telegram-sized thought block, rolling the tail once it grows past the limit.

    Short thoughts are shown with the normal prefix. Once the limit is exceeded,
    the placeholder switches to a "continues" marker and shows the latest tail so
    the user can always see the most recent reasoning without generating new
    Telegram messages.
    """
    if not thought:
        return ""
    full = f"{_THINKING_PREFIX}\n{thought}"
    if len(full) <= limit:
        return full
    tail_limit = limit - len(_THINKING_CONTINUED) - 1  # -1 for the newline
    if tail_limit <= 0:
        return thought[-limit:]
    return f"{_THINKING_CONTINUED}\n{thought[-tail_limit:]}"


def _format_elapsed(seconds: float) -> str:
    """Return a short, human-readable elapsed time."""
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _build_heartbeat_text(base: str, elapsed: float, limit: int = 4096) -> str:
    """Return ``base`` with a small liveness suffix, truncating if needed."""
    suffix = f"\n\n(still working, {_format_elapsed(elapsed)})"
    if not base:
        return suffix[1:] if len(suffix) > limit else suffix
    total = base + suffix
    if len(total) <= limit:
        return total
    max_base = limit - len(suffix) - 3
    if max_base <= 0:
        return total[:limit]
    return base[:max_base] + "..." + suffix


_TELEGRAM_HELP = """Available commands:

/status - current model, session id, working directory, and context-window usage
/metrics - token usage and latency for this chat
/mcp list | /mcp enable <name> | /mcp disable <name> - manage MCP servers
/skill list | /skill enable <name> | /skill disable <name> | /skill create <name> <markdown> - manage skills
/plugin list | /plugin enable <name> | /plugin disable <name> | /plugin reload <name> - manage state plugins
/state <plugin> <event> [args...] - dispatch a state event to a plugin
/memory - show per-chat memory
/models - list ACP model names
/model <name> - switch this chat to a new model
/new - start a fresh Devin session while keeping chat memory
/stop - cancel the current turn and return a partial reply
/continue - resume the previous turn after a partial reply or timeout
/sessions - list numbered sessions for this chat
/resume <n> - resume session n as the active session
/branch <n> - branch from session n and make it active
/summarize - trigger file-backed summarization
/recall <query> - search memory for relevant context
/promote <fact> - append a fact to persona global memory
/stream_thoughts on|off - toggle the optional real-time thought stream
/config <section> <key>=<value> [key=value...] - update live runtime config (task|waker|timer|notifications|telegram)
/help - show this list"""


@dataclass(frozen=True)
class ChatInput:
    """A normalized user message from Telegram, including any reply-to context."""

    chat_id: int
    message_id: int
    text: str
    reply_to: str | None = None
    reply_to_is_bot: bool | None = None
    reply_to_message_id: int | None = None


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
        self._stop = threading.Event()
        self._next_input: ChatInput | None = None
        self._next_lock = threading.Lock()

    def steer(self, chat_input: ChatInput) -> None:
        """Ask the worker to cancel the current turn and start a new one."""
        with self._next_lock:
            self._next_input = chat_input
        self.poller._harness_stop(self.chat_id)

    def stop(self) -> None:
        """Cancel the current turn and stop the worker."""
        self._stop.set()
        self.poller._harness_stop(self.chat_id)

    def _take_next_message(self) -> ChatInput | None:
        with self._next_lock:
            if self._stop.is_set():
                return None
            chat_input = self._next_input
            self._next_input = None
            return chat_input

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
        committed_text = ""
        last_growth_at = turn_start_at = time.monotonic()
        last_edit_at = turn_start_at
        config = self.poller._live_telegram_config

        def _uncommitted_tail(full: str) -> str:
            if not full:
                return ""
            if committed_text and full.startswith(committed_text):
                return full[len(committed_text) :]
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
            # Clamp wait between 5 s and 25 s. The 5 s floor prevents a tight
            # busy loop if the heartbeat has no placeholder to edit (e.g. the
            # initial sendMessage failed) while still letting us react quickly.
            wait = min(25.0, max(remaining, 5.0))
            status = self._harness_turn_status(wait=wait)
            now = time.monotonic()
            running = status.get("status") == "running"
            if running:
                text = status.get("message_text", "")
            edited = False

            # Start a reply placeholder the moment text starts arriving, even
            # when thought streaming is still active.
            if message_id is None and text:
                message_id = self._send_placeholder(_REPLY_PLACEHOLDER)
                if message_id is not None:
                    self.poller._save_placeholder_state(self.chat_id, message_id, thought_id)
                last_text_sent = _REPLY_PLACEHOLDER

            if message_id is not None and text:
                visible = text[:4096]
                # If the visible text changed, the model is still writing.
                if visible != last_text_sent:
                    self.poller._edit_message_text(self.chat_id, message_id, visible)
                    last_text_sent = visible
                    last_growth_at = now
                    edited = True
                else:
                    # No new text: check whether the tail we already showed
                    # has been sitting idle long enough to be its own message.
                    tail = _uncommitted_tail(text)
                    if _should_commit(tail, now - last_growth_at):
                        # Freeze the current placeholder as a sent message and
                        # start a fresh one so the rest of the reply can stream
                        # below it.
                        committed_text = text
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
                    base = text[:4096] if text else _REPLY_PLACEHOLDER
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

        # Finalise the thought block (when enabled). The placeholder has already
        # been live-edited with the latest visible text, so we only touch it if
        # the final thought is empty (delete) or short enough to fit in one
        # message without splitting. Long thoughts stay as they are, which avoids
        # flooding Telegram with multi-part edits and hitting rate limits.
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
        if committed_text and reply.startswith(committed_text):
            reply = reply[len(committed_text) :].lstrip("\n")
        if not reply:
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
            self.poller._remove_placeholder_state(self.chat_id)

        return result

    def run(self) -> None:
        chat_input = self.chat_input
        try:
            while chat_input:
                self._run_turn(chat_input)
                chat_input = self._take_next_message()
        finally:
            with self.poller._worker_lock:
                self.poller._active_workers.pop(self.chat_id, None)
            self.poller._close_client()


class TelegramPoller:
    def __init__(
        self,
        token: str,
        harness_url: str | None = None,
        poll_interval: float = 2.0,
        *,
        runtime: RuntimeAPI | None = None,
        api_key: str | None = None,
        reply_timeout: float = 300.0,
        stream_thoughts_default: bool = False,
        stream_chunk_interval: float = 2.0,
        intermediate_messages: bool = True,
        intermediate_idle: float = 5.0,
        intermediate_min_chars: int = 20,
        state_dir: Path | None = None,
        reply_preview_chars: int = 240,
        min_telegram_interval: float = 1.0,
        min_edit_message_interval: float = 2.0,
        max_telegram_retries: int = 3,
        max_telegram_backoff: float = 30.0,
        metrics: Any | None = None,
    ):
        self.token = token
        self.metrics = metrics
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.harness_url = harness_url
        self.runtime = runtime
        self._api_key = api_key
        self.poll_interval = poll_interval
        self.reply_timeout = reply_timeout
        self._static_telegram_config = TelegramConfig(
            stream_thoughts=stream_thoughts_default,
            stream_chunk_interval=stream_chunk_interval,
            intermediate_messages=intermediate_messages,
            intermediate_idle=intermediate_idle,
            intermediate_min_chars=intermediate_min_chars,
            min_telegram_interval=min_telegram_interval,
            min_edit_message_interval=min_edit_message_interval,
        )
        self.state_dir = state_dir or Path("sessions") / ".poller-placeholders"
        self.reply_preview_chars = reply_preview_chars
        self._max_telegram_retries = max_telegram_retries
        self._max_telegram_backoff = max_telegram_backoff
        self.offset: int | None = None
        self._local = threading.local()
        self._client_timeout = 35.0
        self._stream_thoughts: dict[int, bool] = {}
        self._active_workers: dict[int, TurnWorker] = {}
        self._worker_lock = threading.RLock()
        self._message_registry_lock = threading.RLock()
        self._rate_limit_lock = threading.RLock()
        self._method_backoff_until: dict[str, float] = {}
        self._chat_last_telegram_api_call: dict[int, float] = {}
        self._stop = threading.Event()

    @property
    def client(self) -> httpx.Client:
        """Return a thread-local httpx.Client so threads do not share one."""
        client = getattr(self._local, "client", None)
        if client is None:
            client = httpx.Client(timeout=self._client_timeout)
            self._local.client = client
        return client

    @property
    def _live_telegram_config(self) -> TelegramConfig:
        """Return the runtime Telegram config if available, otherwise the static one."""
        if self.runtime is not None:
            try:
                config = self.runtime.get_telegram_config()
                if isinstance(config, TelegramConfig):
                    return config
            except (NotImplementedError, AttributeError):
                pass
        return self._static_telegram_config

    def _close_client(self) -> None:
        """Close the current thread's httpx.Client, if one exists."""
        client = getattr(self._local, "client", None)
        if client is not None:
            client.close()
            self._local.client = None

    def _stream_thoughts_enabled(self, chat_id: int) -> bool:
        return self._stream_thoughts.get(chat_id, self._live_telegram_config.stream_thoughts)

    def _throttle_telegram(
        self, method: str, chat_id: int | None, *, throttle: bool = True
    ) -> None:
        """Wait for any active method-specific rate-limit backoff and per-chat min spacing."""
        while True:
            with self._rate_limit_lock:
                now = time.monotonic()
                backoff = max(self._method_backoff_until.get(method, 0.0) - now, 0.0)
                min_interval = self._live_telegram_config.min_telegram_interval
                if method in ("editMessageText", "deleteMessage"):
                    min_interval = self._live_telegram_config.min_edit_message_interval
                chat_wait = 0.0
                if throttle and chat_id is not None:
                    chat_wait = max(
                        self._chat_last_telegram_api_call.get(chat_id, 0.0) + min_interval - now,
                        0.0,
                    )
                wait = max(backoff, chat_wait)
                if wait <= 0.0:
                    return
            time.sleep(wait)

    def _api(
        self,
        method: str,
        *,
        throttle: bool = True,
        **params,
    ) -> dict:
        """Call the Telegram Bot API, respecting rate limits and retries.

        * 429 responses are honoured via the ``retry_after`` parameter (capped
          to ``_max_telegram_backoff``) and retried up to ``_max_telegram_retries``.
        * ``MESSAGE_NOT_MODIFIED`` 400s are treated as a no-op and returned as a
          synthetic ok response so callers do not crash.
        * Other errors are logged with the full Telegram body and re-raised.
        """
        chat_id = params.get("chat_id")
        if self.metrics is not None:
            self.metrics.inc("telegram_api_calls_total", method=method)
        for attempt in range(1, self._max_telegram_retries + 1):
            self._throttle_telegram(method, chat_id, throttle=throttle)
            try:
                # Use POST with form data to avoid leaking the token in URL
                # query strings and to side-step GET URL-length limits for long
                # replies.
                resp = self.client.post(f"{self.base_url}/{method}", data=params)
                resp.raise_for_status()
                with self._rate_limit_lock:
                    if chat_id is not None:
                        self._chat_last_telegram_api_call[chat_id] = time.monotonic()
                data = resp.json()
                if not data.get("ok"):
                    raise RuntimeError(f"Telegram API {method} failed: {data}")
                return data
            except httpx.HTTPStatusError as exc:
                try:
                    body = exc.response.json()
                except (ValueError, TypeError):
                    body = {}
                description = (body.get("description") or "").lower()

                if exc.response.status_code == 429:
                    if self.metrics is not None:
                        self.metrics.inc("telegram_api_rate_limited_total", method=method)
                    params_data = body.get("parameters") or {}
                    retry_after = params_data.get("retry_after")
                    if retry_after is not None:
                        delay = float(retry_after)
                    else:
                        delay = 2 ** (attempt - 1)
                    delay = min(delay, self._max_telegram_backoff)
                    with self._rate_limit_lock:
                        self._method_backoff_until[method] = max(
                            self._method_backoff_until.get(method, 0.0),
                            time.monotonic() + delay,
                        )
                    logger.warning(
                        "Telegram rate limit on %s (attempt %d/%d), backing off %.1fs",
                        method,
                        attempt,
                        self._max_telegram_retries,
                        delay,
                    )
                    if attempt < self._max_telegram_retries:
                        continue
                    logger.error(
                        "Telegram %s still rate-limited after %d attempts", method, attempt
                    )
                    raise

                if exc.response.status_code == 400 and "not modified" in description:
                    logger.debug("Telegram edit not modified for %s; treating as no-op", method)
                    return {"ok": True, "result": {}}

                if 500 <= exc.response.status_code < 600:
                    if self.metrics is not None:
                        self.metrics.inc("telegram_api_5xx_total", method=method)
                    delay = min(2 ** (attempt - 1), self._max_telegram_backoff)
                    with self._rate_limit_lock:
                        self._method_backoff_until[method] = max(
                            self._method_backoff_until.get(method, 0.0),
                            time.monotonic() + delay,
                        )
                    logger.warning(
                        "Telegram %s returned %d (attempt %d/%d), retrying in %.1fs",
                        method,
                        exc.response.status_code,
                        attempt,
                        self._max_telegram_retries,
                        delay,
                    )
                    if attempt < self._max_telegram_retries:
                        continue

                if self.metrics is not None:
                    self.metrics.inc("telegram_api_errors_total", method=method)
                logger.error(
                    "Telegram API %s failed: %s - %s",
                    method,
                    exc.response.status_code,
                    body,
                )
                raise

            except httpx.RequestError as exc:
                delay = min(2 ** (attempt - 1), self._max_telegram_backoff)
                with self._rate_limit_lock:
                    self._method_backoff_until[method] = max(
                        self._method_backoff_until.get(method, 0.0),
                        time.monotonic() + delay,
                    )
                logger.warning(
                    "Telegram %s request error on attempt %d/%d: %s; retrying in %.1fs",
                    method,
                    attempt,
                    self._max_telegram_retries,
                    exc,
                    delay,
                )
                if attempt < self._max_telegram_retries:
                    continue
                logger.exception("Telegram %s request failed after %d attempts", method, attempt)
                raise

    def _send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        parse_mode: str | None = None,
    ) -> int | None:
        """Send a Telegram message and return its message_id."""
        text = text[:4096]
        try:
            params: dict[str, Any] = {"chat_id": chat_id, "text": text}
            if parse_mode:
                params["parse_mode"] = parse_mode
            if reply_to_message_id is not None:
                params["reply_to_message_id"] = reply_to_message_id
            data = self._api("sendMessage", **params)
            logger.info("Reply sent to chat %s", chat_id)
            return data.get("result", {}).get("message_id")
        except httpx.HTTPStatusError as exc:
            body: dict[str, Any] = {}
            try:
                body = exc.response.json()
            except (ValueError, TypeError):
                pass
            description = (body.get("description") or "").lower()
            is_parse_error = (
                exc.response.status_code == 400
                and ("parse" in description or "markdown" in description)
                and parse_mode is not None
            )
            if is_parse_error:
                logger.warning(
                    "Telegram %s parse failed for chat %s, falling back to plain",
                    parse_mode,
                    chat_id,
                )
                plain = _strip_mdv2(text)
                try:
                    return self._send_message(
                        chat_id, plain, reply_to_message_id=reply_to_message_id
                    )
                except Exception:
                    logger.exception("Failed to send plain-text fallback to chat %s", chat_id)
            else:
                logger.exception("Failed to send Telegram message")
            return None
        except Exception:
            logger.exception("Failed to send Telegram message")
            return None

    def _edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> bool:
        """Edit an existing Telegram message in place. Return True on success."""
        if message_id is None:
            return False
        text = text[:4096]
        if not text:
            return False
        try:
            params: dict[str, Any] = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
            }
            if parse_mode:
                params["parse_mode"] = parse_mode
            self._api("editMessageText", **params)
            return True
        except httpx.HTTPStatusError as exc:
            body: dict[str, Any] = {}
            try:
                body = exc.response.json()
            except (ValueError, TypeError):
                pass
            description = (body.get("description") or "").lower()
            is_parse_error = (
                exc.response.status_code == 400
                and ("parse" in description or "markdown" in description)
                and parse_mode is not None
            )
            if is_parse_error:
                logger.warning(
                    "Telegram %s edit parse failed for chat %s, falling back to plain",
                    parse_mode,
                    chat_id,
                )
                plain = _strip_mdv2(text)
                try:
                    self._api(
                        "editMessageText",
                        chat_id=chat_id,
                        message_id=message_id,
                        text=plain,
                    )
                    return True
                except Exception:
                    logger.exception("Failed to edit plain-text fallback in chat %s", chat_id)
            else:
                logger.exception("Failed to edit Telegram message")
            return False
        except Exception:
            logger.exception("Failed to edit Telegram message")
            return False

    def _delete_message(self, chat_id: int, message_id: int) -> None:
        """Delete a Telegram message."""
        try:
            self._api("deleteMessage", chat_id=chat_id, message_id=message_id)
        except Exception:
            logger.exception("Failed to delete Telegram message")

    def _chat_dir(self, chat_id: int) -> Path:
        """Return the per-chat session directory, mirroring the harness layout."""
        safe = str(chat_id).replace("/", "_")
        return self.state_dir.parent / safe

    def _message_registry_path(self, chat_id: int) -> Path:
        """Path where the poller records Telegram message_id -> turn mappings."""
        return self._chat_dir(chat_id) / "telegram_messages.jsonl"

    @staticmethod
    def _make_preview(text: str, max_chars: int) -> tuple[str, int]:
        """Return a short preview and the original length."""
        if max_chars <= 0:
            return "", len(text)
        if len(text) <= max_chars:
            return text, len(text)
        from diploid_agent.memory import _trim_to_section

        preview = _trim_to_section(text, max_chars)
        return preview, len(text)

    def _register_message_ids(
        self,
        chat_id: int,
        message_ids: list[int],
        session_number: int,
        turn_number: int,
        text: str,
        kind: str = "reply",
    ) -> None:
        """Record the Telegram message ids for a completed turn.

        This lets a later reply-to reference resolve to a specific turn instead
        of copying the full message text back into the prompt.
        """
        if not message_ids:
            return
        preview, original_length = self._make_preview(text, self.reply_preview_chars)
        path = self._message_registry_path(chat_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            now = time.time()
            for message_id in message_ids:
                record = {
                    "message_id": message_id,
                    "chat_id": chat_id,
                    "session_number": session_number,
                    "turn_number": turn_number,
                    "preview": preview,
                    "original_length": original_length,
                    "kind": kind,
                    "timestamp": now,
                }
                records.append(json.dumps(record))
            with self._message_registry_lock, open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(records) + "\n")
        except Exception:
            logger.exception("Failed to register message ids for chat %s", chat_id)

    def _placeholder_state_path(self, chat_id: int) -> Path:
        """Path where the active placeholder for a chat is tracked."""
        return self.state_dir / f"{chat_id}.json"

    def _save_placeholder_state(
        self, chat_id: int, message_id: int | None, thought_id: int | None
    ) -> None:
        """Record the message ids of the in-flight placeholder(s)."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            state = {"chat_id": chat_id, "message_id": message_id, "thought_id": thought_id}
            self._placeholder_state_path(chat_id).write_text(json.dumps(state))
        except Exception:
            logger.exception("Failed to save placeholder state for chat %s", chat_id)

    def _remove_placeholder_state(self, chat_id: int) -> None:
        """Remove the placeholder state once the turn has completed."""
        try:
            self._placeholder_state_path(chat_id).unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove placeholder state for chat %s", chat_id)

    def _cleanup_orphaned_placeholders(self) -> None:
        """Delete any placeholder messages left over from a previous process."""
        if not self.state_dir.exists():
            return
        for path in self.state_dir.glob("*.json"):
            try:
                state = json.loads(path.read_text())
                chat_id = state.get("chat_id")
                message_id = state.get("message_id")
                thought_id = state.get("thought_id")
                if message_id is not None and chat_id is not None:
                    # Try to update the orphan to a restart notice; if it fails,
                    # delete it instead.
                    try:
                        self._api(
                            "editMessageText",
                            chat_id=chat_id,
                            message_id=message_id,
                            text="Service restarted. The previous reply was interrupted.",
                        )
                    except Exception:  # noqa: BLE001
                        self._delete_message(chat_id, message_id)
                if thought_id is not None and chat_id is not None:
                    self._delete_message(chat_id, thought_id)
            except Exception:
                logger.exception("Failed to clean up placeholder state %s", path)
            finally:
                with contextlib.suppress(OSError):
                    path.unlink()

    @staticmethod
    def _split_telegram_text(text: str, reserve: int = 16) -> list[str]:
        """Split a long message into Telegram-sized chunks.

        Tries to keep paragraphs together, then lines, then sentences, then words.
        Each chunk will be at most ``4096 - reserve`` characters, so the caller
        can append a marker like `` (1/3)`` without exceeding Telegram's limit.
        """
        if not text:
            return [text]

        max_len = 4096 - reserve
        if len(text) <= 4096:
            return [text]

        def split_paragraphs(src: str) -> list[str]:
            out: list[str] = []
            current = ""
            for p in src.split("\n\n"):
                if not p:
                    continue
                if len(p) > max_len:
                    if current:
                        out.append(current)
                        current = ""
                    out.extend(split_lines(p))
                elif current and len(current) + 2 + len(p) <= max_len:
                    current += "\n\n" + p
                else:
                    if current:
                        out.append(current)
                    current = p
            if current:
                out.append(current)
            return out or [src]

        def split_lines(src: str) -> list[str]:
            out: list[str] = []
            current = ""
            for line in src.split("\n"):
                if not line:
                    continue
                if len(line) > max_len:
                    if current:
                        out.append(current)
                        current = ""
                    out.extend(split_sentences(line))
                elif current and len(current) + 1 + len(line) <= max_len:
                    current += "\n" + line
                else:
                    if current:
                        out.append(current)
                    current = line
            if current:
                out.append(current)
            return out or [src]

        def split_sentences(src: str) -> list[str]:
            out: list[str] = []
            current = ""
            # Split on whitespace after sentence-ending punctuation.
            parts = re.split(r"(?<=[.!?])\s+", src)
            for part in parts:
                if not part:
                    continue
                if len(part) > max_len:
                    if current:
                        out.append(current)
                        current = ""
                    out.extend(split_words(part))
                elif current and len(current) + 1 + len(part) <= max_len:
                    current += " " + part
                else:
                    if current:
                        out.append(current)
                    current = part
            if current:
                out.append(current)
            return out or [src]

        def split_words(src: str) -> list[str]:
            out: list[str] = []
            current = ""
            for word in src.split(" "):
                if not word:
                    continue
                if len(word) > max_len:
                    if current:
                        out.append(current)
                        current = ""
                    for i in range(0, len(word), max_len):
                        out.append(word[i : i + max_len])
                    continue
                if current and len(current) + 1 + len(word) <= max_len:
                    current += " " + word
                else:
                    if current:
                        out.append(current)
                    current = word
            if current:
                out.append(current)
            return out or [src]

        return split_paragraphs(text)

    def _send_text(
        self,
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        """Send or edit a message, splitting it into multiple Telegram messages if needed.

        If ``first_message_id`` is provided, the first chunk edits that message
        in place and the remaining chunks are sent as new messages. Each message
        in a multi-part reply is tagged with ``(X/Y)`` at the end.
        """
        if not text:
            return []

        config = self._live_telegram_config
        if config.message_format == "markdown_v2":
            parse_mode = "MarkdownV2"
            formatted = format_markdown_v2(text)
            len_fn: Any = utf16_len
            reserve = 16
            marker_prefix = " \\("
            marker_suffix = "\\)"
        else:
            parse_mode = None
            formatted = text
            len_fn = len
            reserve = 16
            marker_prefix = " ("
            marker_suffix = ")"

        chunks = split_telegram_text(formatted, max_length=4096, len_fn=len_fn, reserve=reserve)
        total = len(chunks)
        sent: list[int] = []

        for i, chunk in enumerate(chunks, start=1):
            if total > 1:
                marker = f"{marker_prefix}{i}/{total}{marker_suffix}"
                max_chunk = 4096 - (len_fn(marker) if len_fn is not len else len(marker))
                if (len_fn or len)(chunk) > max_chunk:
                    if len_fn is utf16_len:
                        chunk = _prefix_within_utf16_limit(chunk, max_chunk)
                    else:
                        chunk = chunk[:max_chunk]
                content = separate_chunk_indicator_from_fence(chunk + marker)
            else:
                content = chunk

            if i == 1 and first_message_id is not None:
                if self._edit_message_text(
                    chat_id, first_message_id, content, parse_mode=parse_mode
                ):
                    sent.append(first_message_id)
                else:
                    # Edit failed (rate limit, deleted message, etc.). Remove the
                    # stale placeholder and send the chunk as a fresh message so the
                    # user still receives the reply.
                    self._delete_message(chat_id, first_message_id)
                    msg_id = self._send_message(
                        chat_id,
                        content,
                        reply_to_message_id=reply_to_message_id,
                        parse_mode=parse_mode,
                    )
                    if msg_id is None:
                        logger.error("Failed to send first chunk of reply to chat %s", chat_id)
                        break
                    sent.append(msg_id)
            else:
                # Edits cannot change a message's reply target, but any new
                # message in a split reply should keep threading to the user.
                msg_id = self._send_message(
                    chat_id,
                    content,
                    reply_to_message_id=reply_to_message_id,
                    parse_mode=parse_mode,
                )
                if msg_id is None:
                    logger.error(
                        "Failed to send chunk %d/%d of reply to chat %s",
                        i,
                        total,
                        chat_id,
                    )
                    break
                sent.append(msg_id)

        return sent

    @staticmethod
    def _parse_update(update: dict) -> ChatInput | None:
        """Extract a normalized ChatInput from a Telegram update, or None."""
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        message_id = message.get("message_id")

        # Prefer text, then caption; ignore other media.
        text = message.get("text") or message.get("caption", "")

        # Do not reply to messages the bot sent itself.
        if message.get("from", {}).get("is_bot"):
            return None

        if not chat_id or not text:
            return None

        reply_to = message.get("reply_to_message", {})
        reply_to_text: str | None = None
        reply_to_is_bot: bool | None = None
        reply_to_message_id: int | None = None
        if reply_to:
            reply_to_text = reply_to.get("text") or reply_to.get("caption")
            if reply_to_text:
                reply_to_is_bot = reply_to.get("from", {}).get("is_bot")
            reply_to_message_id = reply_to.get("message_id")

        return ChatInput(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_to=reply_to_text,
            reply_to_is_bot=reply_to_is_bot,
            reply_to_message_id=reply_to_message_id,
        )

    def _send_typing(self, chat_id: int) -> None:
        """Tell Telegram the bot is typing."""
        try:
            self._api("sendChatAction", chat_id=chat_id, action="typing")
        except Exception:
            logger.exception("Failed to send typing action")

    def _typing_worker(self, chat_id: int, stop_event: threading.Event) -> None:
        """Send a typing action every few seconds until stopped."""
        while not stop_event.is_set():
            self._send_typing(chat_id)
            stop_event.wait(timeout=4.0)

    @contextlib.contextmanager
    def _typing_context(self, chat_id: int) -> Generator[None, None, None]:
        """Keep the Telegram typing indicator alive during a long harness call."""
        stop = threading.Event()
        thread = threading.Thread(
            target=self._typing_worker,
            args=(chat_id, stop),
            daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2.0)

    def _send_result(
        self,
        chat_id: int,
        result: dict[str, Any],
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send the reply and any system notice, splitting long output if needed."""
        reply = result.get("reply", "")
        self._send_text(chat_id, reply, reply_to_message_id=reply_to_message_id)

        notice = result.get("notice")
        if notice:
            self._send_text(chat_id, f"System: {notice}")

    def _harness_reply(self, chat_id: int, message: str) -> dict[str, Any]:
        try:
            with self._typing_context(chat_id):
                resp = self.client.post(
                    f"{self.harness_url}/chat",
                    json={"chat_id": str(chat_id), "message": message},
                    timeout=self.reply_timeout,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Harness /chat failed")
            return {
                "reply": "Sorry, the harness is having trouble. Try again in a moment.",
                "notice": None,
            }

    def _harness_metrics(self, chat_id: int) -> str:
        try:
            resp = self.client.get(
                f"{self.harness_url}/metrics/{chat_id}",
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            cumulative = data.get("cumulative") or {}
            last_turn = data.get("last_turn")
            if not cumulative:
                return "No metrics for this chat yet."
            lines = [
                f"Turns: {cumulative.get('turns', 0)}",
                (
                    f"Tokens: {cumulative.get('total_tokens', 0)} total "
                    f"({cumulative.get('input_tokens', 0)} in / {cumulative.get('output_tokens', 0)} out)"
                ),
            ]
            if cumulative.get("cached_tokens"):
                lines.append(f"Cached tokens: {cumulative['cached_tokens']}")
            lines.append(f"Latency: {cumulative.get('latency_seconds', 0.0):.2f}s")
            if last_turn:
                lines.append(
                    f"Last turn: #{last_turn.get('turn_number', '?')} "
                    f"({last_turn.get('model', '?')}) — "
                    f"{last_turn.get('total_tokens', 0)} tokens in "
                    f"{last_turn.get('latency_seconds', 0.0):.2f}s"
                )
            return "\n".join(lines)
        except Exception:
            logger.exception("Harness /metrics failed")
            return "Sorry, I could not fetch your chat metrics."

    @staticmethod
    def _parse_config_value(raw: str) -> Any:
        """Parse a config value from a Telegram command argument.

        Tries JSON first so numbers, booleans, null, and lists work; falls back
        to a plain string.
        """
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def _harness_config(self, chat_id: int, arg: str) -> str:
        """Handle /config <section> <key>=<value> [...]."""
        parts = arg.split(None, 1)
        if len(parts) < 2:
            return (
                "Usage: /config <section> <key>=<value> [key=value...]\n"
                "Sections: task, waker, timer, notifications, telegram"
            )

        section, rest = parts
        section = section.lower()
        section_map: dict[str, tuple[type, str]] = {
            "task": (TaskConfig, "update_task_config"),
            "waker": (WakerConfig, "update_waker_config"),
            "timer": (TimerConfig, "update_timer_config"),
            "notifications": (NotificationsConfig, "update_notifications_config"),
            "telegram": (TelegramConfig, "update_telegram_config"),
        }

        if section not in section_map:
            return (
                f"Unknown config section: {section}. Use task|waker|timer|notifications|telegram."
            )

        model_cls, update_method = section_map[section]

        fields: dict[str, Any] = {}
        for pair in rest.split():
            if "=" not in pair:
                return f"Invalid pair (expected key=value): {pair}"
            key, value = pair.split("=", 1)
            fields[key] = self._parse_config_value(value)

        try:
            cfg = model_cls(**fields)
        except (ValueError, TypeError) as exc:
            return f"Invalid {section} config: {exc}"

        if self.runtime is not None:
            try:
                updater = getattr(self.runtime, update_method)
                updater(cfg)
                getter = getattr(self.runtime, f"get_{section}_config")
                current = getter()
                return json.dumps(current.model_dump(), indent=2, default=str)
            except ConfigPersistenceError as exc:
                return f"Updated in memory, but could not persist: {exc}"
            except (AttributeError, ValueError, TypeError) as exc:
                logger.exception("Runtime config update failed")
                return f"Sorry, I could not update {section} config: {exc}"

        if self.harness_url is None:
            return "No runtime or harness URL configured."

        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key

        try:
            resp = self.client.post(
                f"{self.harness_url}/{section}/config",
                json=fields,
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2, sort_keys=False)
        except httpx.HTTPStatusError as exc:
            return f"Harness returned {exc.response.status_code}: {exc.response.text}"
        except (httpx.HTTPError, OSError):
            logger.exception("Harness /%s/config failed", section)
            return f"Sorry, I could not update {section} config via the harness."

    def _harness_mcp_list(self, chat_id: int) -> str:
        try:
            resp = self.client.get(
                f"{self.harness_url}/mcp/{chat_id}",
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /mcp failed")
            return "Sorry, I could not fetch the MCP server list."

    def _harness_mcp_enable(self, chat_id: int, name: str) -> str:
        try:
            resp = self.client.post(
                f"{self.harness_url}/mcp",
                json={"chat_id": str(chat_id), "command": "enable", "name": name},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /mcp enable failed")
            return f"Sorry, I could not enable {name}."

    def _harness_mcp_disable(self, chat_id: int, name: str) -> str:
        try:
            resp = self.client.post(
                f"{self.harness_url}/mcp",
                json={"chat_id": str(chat_id), "command": "disable", "name": name},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /mcp disable failed")
            return f"Sorry, I could not disable {name}."

    def _harness_skill_list(self, chat_id: int) -> str:
        try:
            resp = self.client.get(
                f"{self.harness_url}/skill/{chat_id}",
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /skill failed")
            return "Sorry, I could not fetch the skill list."

    def _harness_skill_enable(self, chat_id: int, name: str) -> str:
        try:
            resp = self.client.post(
                f"{self.harness_url}/skill",
                json={"chat_id": str(chat_id), "command": "enable", "name": name},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /skill enable failed")
            return f"Sorry, I could not enable {name}."

    def _harness_skill_disable(self, chat_id: int, name: str) -> str:
        try:
            resp = self.client.post(
                f"{self.harness_url}/skill",
                json={"chat_id": str(chat_id), "command": "disable", "name": name},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /skill disable failed")
            return f"Sorry, I could not disable {name}."

    def _harness_skill_create(self, chat_id: int, name: str, content: str) -> str:
        try:
            resp = self.client.post(
                f"{self.harness_url}/skill",
                json={
                    "chat_id": str(chat_id),
                    "command": "create",
                    "name": name,
                    "content": content,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /skill create failed")
            return f"Sorry, I could not create {name}."

    def _harness_plugin_list(self, chat_id: int) -> str:
        try:
            if self.runtime is not None:
                plugins = self.runtime.plugin_list(str(chat_id))
                return json.dumps(plugins, default=str, indent=2)
            resp = self.client.get(
                f"{self.harness_url}/plugins/{chat_id}",
                timeout=30.0,
            )
            resp.raise_for_status()
            return json.dumps(resp.json().get("plugins", []), default=str, indent=2)
        except Exception:
            logger.exception("Harness /plugins failed")
            return "Sorry, I could not fetch the plugin list."

    def _harness_plugin_enable(self, chat_id: int, name: str, enabled: bool) -> str:
        try:
            if self.runtime is not None:
                result = self.runtime.plugin_set_enabled(str(chat_id), name, enabled)
                return result.reply
            resp = self.client.post(
                f"{self.harness_url}/plugin/enable",
                json={"chat_id": str(chat_id), "name": name, "enabled": enabled},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /plugin/enable failed")
            return f"Sorry, I could not {'enable' if enabled else 'disable'} {name}."

    def _harness_plugin_reload(self, chat_id: int, name: str) -> str:
        try:
            if self.runtime is not None:
                result = self.runtime.plugin_reload(str(chat_id), name)
                return result.reply
            resp = self.client.post(
                f"{self.harness_url}/plugin/reload",
                json={"chat_id": str(chat_id), "name": name},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /plugin/reload failed")
            return f"Sorry, I could not reload {name}."

    def _harness_status(self, chat_id: int) -> str:
        try:
            resp = self.client.get(
                f"{self.harness_url}/status/{chat_id}",
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("active"):
                return "No active session for this chat yet."
            lines = [
                f"Persona: {data.get('persona')}",
                f"Model: {data.get('model')}",
                f"Session: {data.get('session_id')}",
                f"Working directory: {data.get('cwd')}",
            ]
            ctx = data.get("context_usage") or {}
            context_window = ctx.get("context_window")
            if context_window:
                lines.append(f"Context window: {context_window} tokens")
                last_turn = ctx.get("last_turn") or {}
                if last_turn.get("input_percent") is not None:
                    lines.append(
                        f"Last turn: {last_turn['input_percent']}% input, "
                        f"{last_turn.get('total_percent', 0)}% total "
                        f"({last_turn.get('available_tokens', 0)} available)"
                    )
                cumulative = ctx.get("cumulative") or {}
                if cumulative.get("turns") is not None:
                    lines.append(
                        f"Cumulative: {cumulative.get('turns', 0)} turns, "
                        f"{cumulative.get('total_tokens', 0)} tokens"
                    )
            return "\n".join(lines)
        except Exception:
            logger.exception("Harness /status failed")
            return "Sorry, I could not fetch your chat status."

    def _harness_memory(self, chat_id: int) -> str:
        try:
            resp = self.client.get(
                f"{self.harness_url}/memory/{chat_id}",
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("memory", "")
        except Exception:
            logger.exception("Harness /memory failed")
            return "Sorry, I could not fetch your chat memory."

    def _harness_summarize(self, chat_id: int) -> dict[str, Any]:
        try:
            with self._typing_context(chat_id):
                resp = self.client.post(
                    f"{self.harness_url}/summarize/{chat_id}",
                    timeout=300.0,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Harness /summarize failed")
            return {
                "reply": "Sorry, I could not summarize the conversation.",
                "notice": None,
            }

    def _harness_recall(self, chat_id: int, query: str, tags: list[str] | None = None) -> str:
        try:
            resp = self.client.post(
                f"{self.harness_url}/recall",
                json={"chat_id": str(chat_id), "query": query, "tags": tags or []},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "")
        except Exception:
            logger.exception("Harness /recall failed")
            return "Sorry, I could not recall anything."

    def _harness_promote(self, chat_id: int, fact: str) -> dict[str, Any]:
        try:
            with self._typing_context(chat_id):
                resp = self.client.post(
                    f"{self.harness_url}/promote",
                    json={"chat_id": str(chat_id), "message": fact},
                    timeout=30.0,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Harness /promote failed")
            return {
                "reply": "Sorry, I could not promote that to persona memory.",
                "notice": None,
            }

    def _harness_models(self) -> str:
        try:
            resp = self.client.get(f"{self.harness_url}/models", timeout=30.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            # Telegram message limit is 4096 chars; truncate list if needed.
            text = ", ".join(models)
            if len(text) > 4000:
                text = text[:3997] + "..."
            return f"Available models:\n{text}"
        except Exception:
            logger.exception("Harness /models failed")
            return "Sorry, I could not fetch the model list."

    def _harness_switch_model(self, chat_id: int, model: str) -> dict[str, Any]:
        try:
            with self._typing_context(chat_id):
                resp = self.client.post(
                    f"{self.harness_url}/switch-model",
                    json={"chat_id": str(chat_id), "model": model},
                    timeout=300.0,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Harness /switch-model failed")
            return {
                "reply": f"Sorry, I could not switch to model `{model}`.",
                "notice": None,
            }

    def _harness_new(self, chat_id: int) -> dict[str, Any]:
        try:
            with self._typing_context(chat_id):
                resp = self.client.post(
                    f"{self.harness_url}/new/{chat_id}",
                    timeout=300.0,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Harness /new failed")
            return {
                "reply": "Sorry, I could not start a new session.",
                "notice": None,
            }

    def _harness_stop(self, chat_id: int) -> dict[str, Any]:
        if self.runtime is not None:
            try:
                result = self.runtime.stop(str(chat_id))
                return {
                    "reply": getattr(result, "reply", ""),
                    "notice": getattr(result, "notice", None),
                }
            except Exception:
                logger.exception("Runtime stop failed")
                return {
                    "reply": "Sorry, I could not stop the current turn.",
                    "notice": None,
                }

        try:
            resp = self.client.post(
                f"{self.harness_url}/stop",
                json={"chat_id": str(chat_id)},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("Harness /stop failed")
            return {
                "reply": "Sorry, I could not stop the current turn.",
                "notice": None,
            }

    def _harness_state_event(
        self,
        chat_id: int,
        plugin: str,
        event: str,
        raw_args: str | None,
    ) -> str:
        try:
            payload: dict[str, Any] = {
                "chat_id": str(chat_id),
                "plugin": plugin,
                "event": event,
            }
            if raw_args:
                payload["raw_args"] = raw_args
            resp = self.client.post(
                f"{self.harness_url}/state",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("reply", "Done.")
        except Exception:
            logger.exception("Harness /state failed")
            return "Sorry, I could not dispatch that state event."

    def _harness_sessions(self, chat_id: int) -> str:
        try:
            resp = self.client.get(f"{self.harness_url}/sessions/{chat_id}", timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            lines: list[str] = []
            for session in data.get("sessions", []):
                prefix = "* " if session.get("is_active") else "  "
                lines.append(
                    f"{prefix}{session['number']}. {session.get('label') or 'session'} "
                    f"({session['model']})"
                )
            return "Sessions:\n" + "\n".join(lines)
        except Exception:
            logger.exception("Harness /sessions failed")
            return "Sorry, I could not list sessions."

    def _harness_resume(self, chat_id: int, session_number: int) -> dict[str, Any]:
        try:
            with self._typing_context(chat_id):
                resp = self.client.post(
                    f"{self.harness_url}/resume",
                    json={"chat_id": str(chat_id), "session_number": session_number},
                    timeout=300.0,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Harness /resume failed")
            return {
                "reply": "Sorry, I could not resume that session.",
                "notice": None,
            }

    def _harness_branch(self, chat_id: int, session_number: int) -> dict[str, Any]:
        try:
            with self._typing_context(chat_id):
                resp = self.client.post(
                    f"{self.harness_url}/branch",
                    json={"chat_id": str(chat_id), "session_number": session_number},
                    timeout=300.0,
                )
                resp.raise_for_status()
                return resp.json()
        except Exception:
            logger.exception("Harness /branch failed")
            return {
                "reply": "Sorry, I could not branch that session.",
                "notice": None,
            }

    def _harness_help(self, chat_id: int) -> str:
        return _TELEGRAM_HELP

    def run(self) -> None:
        target = self.runtime or self.harness_url
        logger.info("Starting Telegram poller for %s", target)
        self._stop.clear()
        self._cleanup_orphaned_placeholders()
        while not self._stop.is_set():
            try:
                params: dict[str, int] = {"limit": 100, "timeout": 25}
                if self.offset is not None:
                    params["offset"] = self.offset
                data = self._api("getUpdates", throttle=False, **params)
                for update in data.get("result", []):
                    if self._stop.is_set():
                        break
                    self._handle_update(update)
            except Exception:
                logger.exception("Poller error")
                if self._stop.is_set():
                    break
                time.sleep(self.poll_interval)
                continue

            if self._stop.is_set():
                break
            time.sleep(self.poll_interval)

    def _handle_update(self, update: dict) -> None:
        update_id = update.get("update_id")
        if update_id is not None:
            self.offset = max(self.offset or 0, update_id + 1)

        chat_input = self._parse_update(update)
        if chat_input is None:
            return

        chat_id = chat_input.chat_id
        text = chat_input.text

        # Handle bot commands. Strip the bot's Telegram username and any
        # leading/trailing whitespace so commands like "/metrics@mybot" work.
        command_parts = text.strip().split(None, 1)
        command = command_parts[0].split("@")[0] if command_parts else ""
        arg = command_parts[1].strip() if len(command_parts) > 1 else ""

        if command == "/status":
            reply = self._harness_status(chat_id)
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/metrics":
            reply = self._harness_metrics(chat_id)
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/mcp":
            if not arg or arg == "list":
                reply = self._harness_mcp_list(chat_id)
            else:
                parts = arg.split(None, 1)
                sub = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                if sub == "enable" and name:
                    reply = self._harness_mcp_enable(chat_id, name)
                elif sub == "disable" and name:
                    reply = self._harness_mcp_disable(chat_id, name)
                else:
                    reply = "Usage: /mcp list | /mcp enable <name> | /mcp disable <name>"
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/state":
            if not arg:
                reply = "Usage: /state <plugin> <event> [args...]"
            else:
                parts = arg.split(None, 2)
                plugin = parts[0]
                event = parts[1] if len(parts) > 1 else ""
                raw_args = parts[2] if len(parts) > 2 else None
                reply = self._harness_state_event(chat_id, plugin, event, raw_args)
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/skill":
            if not arg or arg == "list":
                reply = self._harness_skill_list(chat_id)
            else:
                parts = arg.split(None, 2)
                sub = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                content = parts[2] if len(parts) > 2 else ""
                if sub == "enable" and name:
                    reply = self._harness_skill_enable(chat_id, name)
                elif sub == "disable" and name:
                    reply = self._harness_skill_disable(chat_id, name)
                elif sub == "create" and name and content:
                    reply = self._harness_skill_create(chat_id, name, content)
                else:
                    reply = "Usage: /skill list | /skill enable <name> | /skill disable <name> | /skill create <name> <markdown>"
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/plugin":
            if not arg or arg == "list":
                reply = self._harness_plugin_list(chat_id)
            else:
                parts = arg.split(None, 2)
                sub = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                if sub == "enable" and name:
                    reply = self._harness_plugin_enable(chat_id, name, True)
                elif sub == "disable" and name:
                    reply = self._harness_plugin_enable(chat_id, name, False)
                elif sub == "reload" and name:
                    reply = self._harness_plugin_reload(chat_id, name)
                else:
                    reply = "Usage: /plugin list | /plugin enable <name> | /plugin disable <name> | /plugin reload <name>"
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/memory":
            reply = self._harness_memory(chat_id)
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/summarize":
            result = self._harness_summarize(chat_id)
            self._send_result(chat_id, result, reply_to_message_id=chat_input.message_id)
        elif command == "/recall":
            if not arg:
                reply = "Usage: /recall <query>"
                self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
            else:
                reply = self._harness_recall(chat_id, arg)
                self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/promote":
            if not arg:
                reply = "Usage: /promote <fact>"
                self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
            else:
                result = self._harness_promote(chat_id, arg)
                self._send_result(chat_id, result, reply_to_message_id=chat_input.message_id)
        elif command == "/models":
            reply = self._harness_models()
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/model":
            if not arg:
                reply = "Usage: /model <name>"
                self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
            else:
                result = self._harness_switch_model(chat_id, arg)
                self._send_result(chat_id, result, reply_to_message_id=chat_input.message_id)
        elif command == "/new":
            result = self._harness_new(chat_id)
            self._send_result(chat_id, result, reply_to_message_id=chat_input.message_id)
        elif command == "/stop":
            with self._worker_lock:
                worker = self._active_workers.get(chat_id)
            if worker and worker.is_alive():
                worker.stop()
                # Final partial result is sent by the worker.
                return
            result = self._harness_stop(chat_id)
            self._send_result(chat_id, result, reply_to_message_id=chat_input.message_id)
        elif command == "/sessions":
            reply = self._harness_sessions(chat_id)
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/resume":
            if not arg.isdigit():
                reply = "Usage: /resume <number>"
                self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
            else:
                result = self._harness_resume(chat_id, int(arg))
                self._send_result(chat_id, result, reply_to_message_id=chat_input.message_id)
        elif command == "/branch":
            if not arg.isdigit():
                reply = "Usage: /branch <number>"
                self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
            else:
                result = self._harness_branch(chat_id, int(arg))
                self._send_result(chat_id, result, reply_to_message_id=chat_input.message_id)
        elif command == "/help":
            self._send_text(
                chat_id,
                self._harness_help(chat_id),
                reply_to_message_id=chat_input.message_id,
            )
        elif command == "/stream_thoughts":
            if arg.lower() not in ("on", "off"):
                self._send_text(
                    chat_id,
                    "Usage: /stream_thoughts on|off",
                    reply_to_message_id=chat_input.message_id,
                )
            else:
                enabled = arg.lower() == "on"
                self._stream_thoughts[chat_id] = enabled
                state = "enabled" if enabled else "disabled"
                self._send_text(
                    chat_id,
                    f"Thought streaming {state}.",
                    reply_to_message_id=chat_input.message_id,
                )
        elif command == "/config":
            if not arg:
                reply = (
                    "Usage: /config <section> <key>=<value> [key=value...]\n"
                    "Sections: task, waker, timer, notifications, telegram"
                )
            else:
                reply = self._harness_config(chat_id, arg)
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/continue":
            continue_input = ChatInput(
                chat_id=chat_id,
                message_id=chat_input.message_id,
                text="continue",
                reply_to=chat_input.reply_to,
                reply_to_is_bot=chat_input.reply_to_is_bot,
                reply_to_message_id=chat_input.reply_to_message_id,
            )
            logger.info("Message from chat %s: /continue", chat_id)
            with self._worker_lock:
                worker = self._active_workers.get(chat_id)
            if worker and worker.is_alive():
                worker.steer(continue_input)
            else:
                worker = TurnWorker(self, continue_input)
                with self._worker_lock:
                    self._active_workers[chat_id] = worker
                worker.start()
        else:
            logger.info("Message from chat %s: %r", chat_id, text[:80])
            with self._worker_lock:
                worker = self._active_workers.get(chat_id)
            if worker and worker.is_alive():
                worker.steer(chat_input)
            else:
                worker = TurnWorker(self, chat_input)
                with self._worker_lock:
                    self._active_workers[chat_id] = worker
                worker.start()


class TelegramTransport(Transport):
    """A Transport implementation backed by the Telegram long-poller."""

    def __init__(
        self,
        token: str,
        runtime: RuntimeAPI | None = None,
        **poller_kwargs: Any,
    ):
        poller_kwargs.setdefault("runtime", runtime)
        self._poller = TelegramPoller(token, **poller_kwargs)
        self._thread: threading.Thread | None = None

    def start(self, runtime: RuntimeAPI | None = None) -> None:
        if runtime is not None:
            self._poller.runtime = runtime
        if self._poller.harness_url is None and self._poller.runtime is None:
            raise RuntimeError("TelegramTransport requires a runtime or harness_url to start")
        self._poller._stop.clear()
        self._thread = threading.Thread(
            target=self._poller.run,
            daemon=True,
            name="telegram-transport",
        )
        self._thread.start()

    def stop(self) -> None:
        self._poller._stop.set()
        with self._poller._worker_lock:
            workers = list(self._poller._active_workers.values())
        for worker in workers:
            worker.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

    def send(self, message: OutboundMessage) -> list[int]:
        chat_id_value: int | str = message.chat_id
        try:
            chat_id_value = int(message.chat_id)
        except (ValueError, TypeError):
            pass

        sent: list[int] = []
        if message.text:
            sent.extend(
                self._poller._send_text(
                    chat_id_value,
                    message.text,
                    reply_to_message_id=message.reply_to_message_id,
                )
            )
        if message.notice:
            sent.extend(
                self._poller._send_text(
                    chat_id_value,
                    f"System: {message.notice}",
                )
            )
        return sent


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram long-polling ingress")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent.parent / "config" / "harness.yaml",
    )
    parser.add_argument(
        "--harness-url",
        default="http://127.0.0.1:4003",
        help="Base URL of the diploid-agent /chat endpoint",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    token = config.harness.telegram.token or os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error(
            "TELEGRAM_BOT_TOKEN not found in secrets.env, env, or config. "
            "Set it before running the poller."
        )
        return 1

    # The poller must wait longer than the harness's absolute ACP timeout,
    # otherwise it gives up on a turn that is still running.
    reply_timeout = config.engine.timeout + 30.0
    metrics = MetricsCollector(prefix="harness")
    poller = TelegramPoller(
        token=token,
        harness_url=args.harness_url.rstrip("/"),
        poll_interval=args.poll_interval,
        reply_timeout=reply_timeout,
        api_key=config.secrets.harness_api_key,
        stream_thoughts_default=config.harness.telegram.stream_thoughts,
        stream_chunk_interval=config.harness.telegram.stream_chunk_interval,
        intermediate_messages=config.harness.telegram.intermediate_messages,
        intermediate_idle=config.harness.telegram.intermediate_idle,
        intermediate_min_chars=config.harness.telegram.intermediate_min_chars,
        min_telegram_interval=config.harness.telegram.min_telegram_interval,
        min_edit_message_interval=config.harness.telegram.min_edit_message_interval,
        state_dir=config.harness.sessions_root / ".poller-placeholders",
        reply_preview_chars=config.harness.memory.max_bot_reply_quote_chars,
        metrics=metrics,
    )
    poller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
