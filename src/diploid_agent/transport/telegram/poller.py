#!/usr/bin/env python3
"""Telegram long-polling transport.

Polls Telegram Bot API `getUpdates` and forwards each text message to the
configured runtime or harness `/chat` endpoint, then sends the reply back to
the user.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx

from diploid_agent.config import (
    TelegramConfig,
)
from diploid_agent.models import ChatResult
from diploid_agent.transport.base import (
    RuntimeAPI,
)
from diploid_agent.transport.command_handler import CommandHandler
from diploid_agent.transport.interactive import (
    AskBlock,
    build_empty_inline_keyboard,
    build_inline_keyboard,
    build_keyboard_remove,
    extract_ask_block,
    is_ask_cancel_callback,
    parse_ask_callback_index,
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


from diploid_agent.transport.telegram.models import ChatInput
from diploid_agent.transport.telegram.workers import DeliveryWorker, TurnWorker

from .commands import TelegramCommandMixin


class TelegramPoller(TelegramCommandMixin):
    def __init__(
        self,
        token: str,
        harness_url: str | None = None,
        poll_interval: float = 2.0,
        *,
        runtime: RuntimeAPI | None = None,
        api_key: str | None = None,
        reply_timeout: float | None = 300.0,
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
        message_format: str = "plain",
        code_style: str = "inline",
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
            message_format=message_format,
            code_style=code_style,
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
        self._pending_inputs: dict[int, deque[ChatInput]] = {}
        self._delivery_workers: dict[int, DeliveryWorker] = {}
        self._last_user_message_ids: dict[int, int] = {}
        self._send_locks: dict[int, threading.RLock] = {}
        self._worker_lock = threading.RLock()
        self._message_registry_lock = threading.RLock()
        self._rate_limit_lock = threading.RLock()
        self._method_backoff_until: dict[str, float] = {}
        self._chat_last_telegram_api_call: dict[int, float] = {}
        self._stop = threading.Event()
        self.command_handler = CommandHandler(
            runtime=runtime,
            harness_url=harness_url,
            api_key=api_key,
            client_provider=lambda: self.client,
        )

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

    def _ensure_delivery_worker(self, chat_id: int) -> None:
        """Start a DeliveryWorker for this chat if outbox delivery is enabled."""
        with self._worker_lock:
            if chat_id in self._delivery_workers and self._delivery_workers[chat_id].is_alive():
                return
            config = self.command_handler.call(
                method="get_config",
                http_path="/config",
                http_method="GET",
                requires_chat_id=False,
                catch=True,
            )
            if not isinstance(config, dict):
                return
            notifications = config.get("harness", {}).get("notifications", {})
            if not notifications.get("outbox_delivery"):
                return
            worker = DeliveryWorker(self, chat_id)
            self._delivery_workers[chat_id] = worker
            worker.start()

    def _deliver_outbox_result(self, chat_id: int, chat_result: ChatResult) -> None:
        """Deliver an outbox ChatResult to Telegram, registering sent message IDs."""
        reply_to_message_id = chat_result.reply_to_message_id
        if reply_to_message_id is None:
            reply_to_message_id = self._last_user_message_ids.get(chat_id)
        if chat_result.reply:
            sent = self._send_text(
                chat_id,
                chat_result.reply,
                reply_to_message_id=reply_to_message_id,
            )
            if sent and chat_result.session_number is not None and chat_result.turn_number is not None:
                self._register_message_ids(
                    chat_id,
                    sent,
                    chat_result.session_number,
                    chat_result.turn_number,
                    chat_result.reply,
                    kind="outbox",
                )
        if chat_result.notice:
            self._send_text(
                chat_id,
                f"System: {chat_result.notice}",
                reply_to_message_id=reply_to_message_id,
            )

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
        reply_markup: dict[str, Any] | None = None,
    ) -> int | None:
        """Send a Telegram message and return its message_id."""
        text = text[:4096]
        try:
            params: dict[str, Any] = {"chat_id": chat_id, "text": text}
            if parse_mode:
                params["parse_mode"] = parse_mode
            if reply_to_message_id is not None:
                params["reply_to_message_id"] = reply_to_message_id
            if reply_markup is not None:
                params["reply_markup"] = json.dumps(reply_markup)
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
                        chat_id,
                        plain,
                        reply_to_message_id=reply_to_message_id,
                        reply_markup=reply_markup,
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

    def _answer_callback_query(self, callback_query_id: str) -> None:
        """Answer a Telegram callback query so the client stops showing a spinner."""
        try:
            self._api("answerCallbackQuery", callback_query_id=callback_query_id)
        except Exception:
            logger.exception("Failed to answer callback query")

    def _clear_inline_keyboard(self, chat_id: int, message_id: int) -> None:
        """Remove the inline keyboard from an existing message."""
        try:
            self._api(
                "editMessageReplyMarkup",
                chat_id=chat_id,
                message_id=message_id,
                reply_markup=json.dumps(build_empty_inline_keyboard()),
            )
        except Exception:
            logger.exception(
                "Failed to clear inline keyboard in chat %s message %s",
                chat_id,
                message_id,
            )

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

    def _pending_question_path(self, chat_id: int) -> Path:
        """Path where the active question for a chat is tracked."""
        return self.state_dir / f"{chat_id}.ask.json"

    def _save_pending_question(
        self, chat_id: int, ask_block: AskBlock, message_id: int | None
    ) -> None:
        """Persist a pending question so we can map the next button press back to it."""
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "chat_id": chat_id,
                "question": ask_block.question,
                "options": ask_block.options,
                "cancellable": ask_block.cancellable,
                "cancel_label": ask_block.cancel_label,
                "message_id": message_id,
            }
            self._pending_question_path(chat_id).write_text(json.dumps(payload))
        except Exception:
            logger.exception("Failed to save pending question for chat %s", chat_id)

    def _load_pending_question(self, chat_id: int) -> dict[str, Any] | None:
        """Load the pending question for a chat, or None."""
        path = self._pending_question_path(chat_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            options = data.get("options") or []
            if not options:
                return None
            return {
                "question": data.get("question", ""),
                "options": [str(o) for o in options],
                "cancellable": data.get("cancellable", False),
                "cancel_label": data.get("cancel_label", "Cancel"),
                "message_id": data.get("message_id"),
            }
        except (OSError, json.JSONDecodeError):
            return None

    def _remove_pending_question(self, chat_id: int) -> None:
        """Remove the pending question for a chat."""
        try:
            self._pending_question_path(chat_id).unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove pending question for chat %s", chat_id)

    def _maybe_answer_pending_question(
        self, chat_input: ChatInput
    ) -> ChatInput | None:
        """If the user is answering a pending question, rewrite the message.

        If the question is cancellable and the user pressed the cancel button,
        remove the keyboard, drop the question, and return ``None`` so no turn
        is started.
        """
        pending = self._load_pending_question(chat_input.chat_id)

        # Callback queries come from inline keyboards. They have no user-facing
        # message, so a cancel can be completely silent and a valid answer is
        # translated back to the option text before being sent to the harness.
        if chat_input.callback_query_id is not None:
            return self._handle_ask_callback(chat_input, pending)

        if pending is None:
            return chat_input

        if pending.get("cancellable") and chat_input.text == pending.get(
            "cancel_label", "Cancel"
        ):
            self._remove_pending_question(chat_input.chat_id)
            try:
                self._send_message(
                    chat_input.chat_id,
                    "Cancelled.",
                    reply_markup=build_keyboard_remove(),
                )
            except Exception:
                logger.exception(
                    "Failed to send cancel confirmation for chat %s", chat_input.chat_id
                )
            # In a private chat the bot can delete the user's button-press
            # message, so the cancel looks like it was swallowed rather than sent.
            self._delete_message(chat_input.chat_id, chat_input.message_id)
            return None

        if chat_input.text not in pending["options"]:
            self._remove_pending_question(chat_input.chat_id)
            return chat_input

        self._remove_pending_question(chat_input.chat_id)
        answer = (
            f'The user answered the question "{pending["question"]}" '
            f"by selecting: {chat_input.text}"
        )
        return ChatInput(
            chat_id=chat_input.chat_id,
            message_id=chat_input.message_id,
            text=answer,
            reply_to=pending["question"],
            reply_to_is_bot=True,
            reply_to_message_id=pending["message_id"],
        )

    def _handle_ask_callback(
        self,
        chat_input: ChatInput,
        pending: dict[str, Any] | None,
    ) -> ChatInput | None:
        """Handle an inline-keyboard button press for a pending ask block.

        Cancels are silent: the question is edited to ``Cancelled.`` and the
        keyboard is removed. Valid answers are translated back to the option
        text and sent to the harness. Unknown/stale callbacks are ignored.
        """
        data = chat_input.text
        self._answer_callback_query(chat_input.callback_query_id)

        if pending is None:
            # Stale callback with no tracked question. Remove the keyboard so
            # the user cannot press it again.
            self._clear_inline_keyboard(chat_input.chat_id, chat_input.message_id)
            return None

        question_message_id = pending.get("message_id")

        if pending.get("cancellable") and is_ask_cancel_callback(data):
            self._remove_pending_question(chat_input.chat_id)
            if question_message_id:
                self._edit_message_text(
                    chat_input.chat_id, question_message_id, "Cancelled."
                )
                self._clear_inline_keyboard(
                    chat_input.chat_id, question_message_id
                )
            return None

        index = parse_ask_callback_index(data)
        if index is None or index < 0 or index >= len(pending["options"]):
            self._remove_pending_question(chat_input.chat_id)
            if question_message_id:
                self._clear_inline_keyboard(
                    chat_input.chat_id, question_message_id
                )
            return None

        selected = pending["options"][index]
        self._remove_pending_question(chat_input.chat_id)
        if question_message_id:
            self._clear_inline_keyboard(
                chat_input.chat_id, question_message_id
            )
        return ChatInput(
            chat_id=chat_input.chat_id,
            message_id=chat_input.message_id,
            text=f'The user answered the question "{pending["question"]}" '
            f"by selecting: {selected}",
            reply_to=pending["question"],
            reply_to_is_bot=True,
            reply_to_message_id=question_message_id,
            callback_query_id=None,
        )

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

        for path in self.state_dir.glob("*.ask.json"):
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

        Sends for a given chat are serialised with an RLock so a turn and its
        delivery worker do not interleave or duplicate messages.
        """
        if not text:
            return []
        send_lock = self._send_locks.setdefault(chat_id, threading.RLock())
        with send_lock:
            return self._send_text_locked(
                chat_id,
                text,
                first_message_id=first_message_id,
                reply_to_message_id=reply_to_message_id,
            )

    def _send_text_locked(
        self,
        chat_id: int,
        text: str,
        *,
        first_message_id: int | None = None,
        reply_to_message_id: int | None = None,
    ) -> list[int]:
        """Implementation of _send_text; caller must hold the per-chat send lock."""
        ask_block: AskBlock | None = None
        display_text = text

        if text:
            display_text, ask_block = extract_ask_block(text)

        if ask_block is not None and first_message_id is not None:
            self._delete_message(chat_id, first_message_id)
            first_message_id = None

        config = self._live_telegram_config
        if config.message_format == "markdown_v2":
            parse_mode = "MarkdownV2"
            formatted = format_markdown_v2(display_text, code_style=config.code_style)
            len_fn: Any = utf16_len
            reserve = 16
            marker_prefix = " \\("
            marker_suffix = "\\)"
        else:
            parse_mode = None
            formatted = display_text
            len_fn = len
            reserve = 16
            marker_prefix = " ("
            marker_suffix = ")"

        logger.info(
            "_send_text chat=%s message_format=%s parse_mode=%s first_message_id=%s has_ask=%s",
            chat_id,
            config.message_format,
            parse_mode,
            first_message_id,
            ask_block is not None,
        )

        chunks = split_telegram_text(formatted, max_length=4096, len_fn=len_fn, reserve=reserve)
        total = len(chunks)
        sent: list[int] = []

        if first_message_id is not None and parse_mode is not None:
            self._delete_message(chat_id, first_message_id)
            first_message_id = None

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

            reply_markup: dict[str, Any] | None = None
            if ask_block is not None and i == 1 and total == 1:
                cancel = ask_block.cancel_label if ask_block.cancellable else None
                reply_markup = build_inline_keyboard(ask_block.options, cancel=cancel)

            if i == 1 and first_message_id is not None:
                if self._edit_message_text(
                    chat_id, first_message_id, content, parse_mode=parse_mode
                ):
                    sent.append(first_message_id)
                else:
                    self._delete_message(chat_id, first_message_id)
                    msg_id = self._send_message(
                        chat_id,
                        content,
                        reply_to_message_id=reply_to_message_id,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                    )
                    if msg_id is None:
                        logger.error("Failed to send first chunk of reply to chat %s", chat_id)
                        break
                    sent.append(msg_id)
            else:
                msg_id = self._send_message(
                    chat_id,
                    content,
                    reply_to_message_id=reply_to_message_id,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup,
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

            if i == 1 and ask_block is not None:
                self._save_pending_question(chat_id, ask_block, msg_id if sent else None)

            if i == 1 and first_message_id is None and ask_block is not None and total > 1:
                logger.warning(
                    "Question in chat %s was split into %d chunks; dropping keyboard",
                    chat_id,
                    total,
                )

        return sent

    @staticmethod
    def _parse_update(update: dict) -> ChatInput | None:
        """Extract a normalized ChatInput from a Telegram update, or None."""
        callback_query = update.get("callback_query")
        if callback_query:
            cq_id = callback_query.get("id")
            from_user = callback_query.get("from", {})
            if from_user.get("is_bot"):
                return None
            message = callback_query.get("message") or {}
            chat = message.get("chat", {})
            chat_id = chat.get("id")
            message_id = message.get("message_id")
            data = callback_query.get("data")
            if not chat_id or not message_id or data is None:
                return None
            # The inline keyboard is attached to the bot's own question message,
            # so the callback data is the user's answer and the original text is
            # the reply-to context.
            return ChatInput(
                chat_id=chat_id,
                message_id=message_id,
                text=data,
                reply_to=message.get("text") or message.get("caption"),
                reply_to_is_bot=True,
                reply_to_message_id=message_id,
                callback_query_id=cq_id,
            )

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
            callback_query_id=None,
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
        result: ChatResult | dict[str, Any],
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send the reply and any system notice, splitting long output if needed."""
        if isinstance(result, ChatResult):
            reply = result.reply or ""
            notice = result.notice
        else:
            reply = result.get("reply", "")
            notice = result.get("notice")
        self._send_text(chat_id, reply, reply_to_message_id=reply_to_message_id)

        if notice:
            self._send_text(chat_id, f"System: {notice}")

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

        chat_input = self._maybe_answer_pending_question(chat_input)
        if chat_input is None:
            return

        chat_id = chat_input.chat_id
        self._last_user_message_ids[chat_id] = chat_input.message_id
        self._ensure_delivery_worker(chat_id)

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
        elif command == "/restart":
            with self._worker_lock:
                worker = self._active_workers.get(chat_id)
            if worker and worker.is_alive():
                worker.stop()
                # The worker will finish its final partial; the restart confirmation is sent below.
            result = self._harness_restart(chat_id)
            self._send_result(chat_id, result, reply_to_message_id=chat_input.message_id)
        elif command == "/graceful-restart":
            if arg:
                service = arg
            elif self.runtime is not None:
                service = f"{self.runtime.config.persona.name}.service"
            else:
                service = "diploid-agent.service"
            result = self._harness_graceful_restart(chat_id, service)
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
        elif command == "/subagents":
            reply = self._harness_subagent_status(chat_id)
            self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
        elif command == "/subagent":
            if not arg:
                reply = "Usage: /subagent <prompt>"
                self._send_text(chat_id, reply, reply_to_message_id=chat_input.message_id)
            else:
                result = self._harness_subagent(chat_id, arg)
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
            if worker is not None and worker.is_alive() and worker._running.is_set():
                worker.steer(continue_input)
            else:
                worker = TurnWorker(self, continue_input)
                with self._worker_lock:
                    self._active_workers[chat_id] = worker
                worker.start()
        else:
            logger.info("Message from chat %s: %r", chat_id, text[:80])
            with self._worker_lock:
                self._pending_inputs.setdefault(chat_id, deque()).append(chat_input)
                worker = self._active_workers.get(chat_id)
                if worker is None or not worker.is_alive():
                    next_input = self._pending_inputs[chat_id].popleft()
                    worker = TurnWorker(self, next_input)
                    self._active_workers[chat_id] = worker
                    worker.start()



