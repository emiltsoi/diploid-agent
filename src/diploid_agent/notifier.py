"""Outbound notification delivery."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Notifier(ABC):
    @abstractmethod
    def send(self, chat_id: str, text: str, *, reply_to_message_id: int | None = None) -> Any:
        """Send a notification and return any delivery metadata."""

    @abstractmethod
    def typing(self, chat_id: str) -> Any:
        """Send a typing/action indicator, if the backend supports it."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the notifier can deliver messages."""

    def send_placeholder(
        self, chat_id: str, text: str, *, reply_to_message_id: int | None = None
    ) -> Any:
        """Send a placeholder message and return its delivery metadata.

        Backends that cannot edit messages may return None and the caller
        will fall back to ``send`` for the final result.
        """
        return None

    def edit_message(self, chat_id: str, message_id: Any, text: str) -> bool:
        """Edit an existing message in place, if the backend supports it."""
        return False

    def delete_message(self, chat_id: str, message_id: Any) -> bool:
        """Delete an existing message, if the backend supports it."""
        return False

    def begin_typing(self, chat_id: str) -> None:
        """Start a continuous typing indicator for this chat."""
        self.typing(chat_id)

    def end_typing(self, chat_id: str) -> None:
        """Stop the continuous typing indicator for this chat."""
        return


class NoopNotifier(Notifier):
    def send(self, chat_id: str, text: str, *, reply_to_message_id: int | None = None) -> None:
        return None

    def typing(self, chat_id: str) -> None:
        return None

    def health(self) -> bool:
        return True


class TelegramNotifier(Notifier):
    def __init__(
        self,
        token: str,
        *,
        client: httpx.Client | None = None,
        metrics: Any | None = None,
    ) -> None:
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.client = client or httpx.Client(timeout=30.0)
        self.metrics = metrics
        self._lock = threading.Lock()
        self._typing_threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._typing_lock = threading.RLock()

    def _post(self, url: str, data: dict[str, Any] | None = None) -> httpx.Response:
        with self._lock:
            return self.client.post(url, data=data)

    def send(
        self, chat_id: str, text: str, *, reply_to_message_id: int | None = None
    ) -> int | None:
        text = text[:4096]
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        if self.metrics is not None:
            self.metrics.inc("telegram_send_total")
        max_retries = 3
        max_backoff = 30.0
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._post(f"{self.base_url}/sendMessage", data=payload)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("ok"):
                    raise RuntimeError(f"Telegram API error: {data}")
                return data.get("result", {}).get("message_id")
            except httpx.HTTPStatusError as exc:
                try:
                    body = exc.response.json()
                except (ValueError, TypeError):
                    body = {}
                if exc.response.status_code == 429 and attempt < max_retries:
                    params = body.get("parameters") or {}
                    retry_after = params.get("retry_after")
                    delay = min(
                        float(retry_after) if retry_after is not None else 2 ** (attempt - 1),
                        max_backoff,
                    )
                    logger.warning(
                        "Telegram rate limit on notification to %s (attempt %d/%d), backing off %.1fs",
                        chat_id,
                        attempt,
                        max_retries,
                        delay,
                    )
                    if self.metrics is not None:
                        self.metrics.inc("telegram_rate_limited_total")
                    time.sleep(delay)
                    continue
                if 500 <= exc.response.status_code < 600 and attempt < max_retries:
                    delay = min(2 ** (attempt - 1), max_backoff)
                    logger.warning(
                        "Telegram %s on notification to %s (attempt %d/%d), retrying in %.1fs",
                        exc.response.status_code,
                        chat_id,
                        attempt,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if self.metrics is not None:
                    self.metrics.inc("telegram_send_failures_total")
                logger.exception("Failed to send Telegram notification to %s: %s", chat_id, body)
                return None
            except httpx.RequestError as exc:
                if attempt < max_retries:
                    delay = min(2 ** (attempt - 1), max_backoff)
                    logger.warning(
                        "Telegram notification request error for %s (attempt %d/%d): %s; retrying in %.1fs",
                        chat_id,
                        attempt,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                if self.metrics is not None:
                    self.metrics.inc("telegram_send_failures_total")
                logger.exception(
                    "Failed to send Telegram notification to %s after %d attempts", chat_id, attempt
                )
                return None
            except Exception:
                if self.metrics is not None:
                    self.metrics.inc("telegram_send_failures_total")
                logger.exception("Failed to send Telegram notification to %s", chat_id)
                return None
        if self.metrics is not None:
            self.metrics.inc("telegram_send_failures_total")
        return None

    def typing(self, chat_id: str) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "action": "typing"}
        try:
            resp = self._post(f"{self.base_url}/sendChatAction", data=payload)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"Telegram API error: {data}")
        except Exception:
            logger.exception("Failed to send Telegram typing to %s", chat_id)

    def send_placeholder(
        self, chat_id: str, text: str, *, reply_to_message_id: int | None = None
    ) -> int | None:
        """Send a placeholder message that can be edited later."""
        return self.send(chat_id, text, reply_to_message_id=reply_to_message_id)

    def edit_message(self, chat_id: str, message_id: Any, text: str) -> bool:
        """Edit an existing Telegram message in place."""
        if message_id is None:
            return False
        text = text[:4096]
        if not text:
            return False
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        try:
            resp = self._post(f"{self.base_url}/editMessageText", data=payload)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"Telegram API error: {data}")
            return True
        except Exception:
            logger.exception("Failed to edit Telegram message %s in chat %s", message_id, chat_id)
            return False

    def delete_message(self, chat_id: str, message_id: Any) -> bool:
        """Delete an existing Telegram message."""
        if message_id is None:
            return False
        try:
            resp = self._post(
                f"{self.base_url}/deleteMessage",
                data={"chat_id": chat_id, "message_id": message_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                raise RuntimeError(f"Telegram API error: {data}")
            return True
        except Exception:
            logger.exception("Failed to delete Telegram message %s in chat %s", message_id, chat_id)
            return False

    def begin_typing(self, chat_id: str) -> None:
        """Start a continuous typing indicator for this chat."""
        with self._typing_lock:
            self.end_typing(chat_id)
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._typing_heartbeat,
                args=(chat_id, stop_event),
                daemon=True,
                name=f"notifier-typing-{chat_id}",
            )
            self._typing_threads[chat_id] = (thread, stop_event)
            thread.start()

    def end_typing(self, chat_id: str) -> None:
        """Stop the continuous typing indicator for this chat."""
        with self._typing_lock:
            entry = self._typing_threads.pop(chat_id, None)
        if entry is None:
            return
        thread, stop_event = entry
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=2.0)

    def _typing_heartbeat(self, chat_id: str, stop_event: threading.Event) -> None:
        """Send a typing action every few seconds until stopped."""
        self.typing(chat_id)
        while not stop_event.is_set():
            if stop_event.wait(timeout=4.0):
                break
            self.typing(chat_id)

    def health(self) -> bool:
        try:
            resp = self._post(f"{self.base_url}/getMe")
            resp.raise_for_status()
            data = resp.json()
            return bool(data.get("ok"))
        except Exception:  # noqa: BLE001
            return False


class WebhookNotifier(Notifier):
    def __init__(self, url: str, *, client: httpx.Client | None = None) -> None:
        self.url = url
        self.client = client or httpx.Client()

    def send(
        self, chat_id: str, text: str, *, reply_to_message_id: int | None = None
    ) -> dict | None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        try:
            resp = self.client.post(self.url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            logger.exception("Webhook notification failed for %s", chat_id)
            return None

    def typing(self, chat_id: str) -> None:
        return None

    def health(self) -> bool:
        try:
            resp = self.client.head(self.url, timeout=5.0)
            return resp.status_code < 500
        except Exception:  # noqa: BLE001
            return False
