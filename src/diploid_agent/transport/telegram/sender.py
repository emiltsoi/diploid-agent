"""Telegram sender mixin for the long-polling transport.

This mixin holds the low-level Telegram Bot API send/edit/delete helpers
used by ``TelegramPoller``. It is not intended to be used on its own; it
expects the host class to provide attributes such as ``client``,
``base_url``, ``_live_telegram_config``, ``_method_backoff_until``,
``_chat_last_telegram_api_call``, ``_rate_limit_lock``, ``_send_locks``,
``_max_telegram_retries``, ``_max_telegram_backoff``, and ``metrics``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import httpx

from diploid_agent.transport.interactive import (
    AskBlock,
    build_empty_inline_keyboard,
    build_inline_keyboard,
    extract_ask_block,
)
from diploid_agent.transport.telegram_format import (
    _prefix_within_utf16_limit,
    _strip_mdv2,
    format_markdown_v2,
    separate_chunk_indicator_from_fence,
    split_telegram_text,
    utf16_len,
)

logger = logging.getLogger("telegram_poll")


class TelegramSenderMixin:
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
