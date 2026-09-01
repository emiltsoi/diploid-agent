"""Per-chat outbox queue and notification delivery."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from diploid_agent.models import ChatResult
from diploid_agent.notifier import NoopNotifier, Notifier, TelegramNotifier, WebhookNotifier

logger = logging.getLogger(__name__)


def _is_telegram_chat_id(chat_id: str) -> bool:
    """Return True if ``chat_id`` looks like a real Telegram chat id."""
    stripped = chat_id.lstrip("-")
    return stripped.isdigit()


class RuntimeOutbox:
    """Per-chat outbox queue and notification delivery."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self._outbox: deque[tuple[str, ChatResult]] = deque()
        self._outbox_condition = threading.Condition()

    @property
    def _lock(self) -> threading.RLock:
        return self._runtime._lock

    @property
    def config(self) -> Any:
        return self._runtime.config

    @property
    def notifier(self) -> Notifier | None:
        return self._runtime.notifier

    def _create_notifier(self) -> Notifier:
        if self.config.harness.notifications.outbox_delivery:
            # The transport (e.g. Telegram long-poller) will consume the outbox.
            return NoopNotifier()
        if not self.config.harness.notifications.enabled:
            return NoopNotifier()
        if self.config.harness.notifications.webhook_url:
            return WebhookNotifier(self.config.harness.notifications.webhook_url)
        token = self.config.harness.telegram.token
        if token:
            return TelegramNotifier(token, metrics=self._runtime.metrics)
        return NoopNotifier()

    @property
    def _outbox_delivery_enabled(self) -> bool:
        return self.config.harness.notifications.outbox_delivery

    def _enqueue_outbox(self, chat_id: str, chat_result: ChatResult) -> None:
        """Put a final ChatResult in the outbox for the transport to deliver."""
        with self._outbox_condition:
            self._outbox.append((chat_id, chat_result))
            self._outbox_condition.notify_all()

    def _safe_notifier_send(
        self,
        chat_id: str,
        text: str,
        notifier: Notifier | None = None,
    ) -> None:
        """Send a notification, swallowing exceptions and logging them."""
        if not text:
            return
        notifier = notifier or self.notifier
        if notifier is None:
            return
        try:
            notifier.send(chat_id, text)
        except Exception:
            logger.exception("Failed to send notification to %s", chat_id)

    def _deliver_chat_result(self, chat_id: str, chat_result: ChatResult) -> None:
        """Send a final ChatResult through the configured delivery channel."""
        if self._outbox_delivery_enabled:
            self._enqueue_outbox(chat_id, chat_result)
            return
        if self.config.harness.notifications.enabled and chat_result.reply:
            self._safe_notifier_send(chat_id, chat_result.reply)

    def outbox_pop(
        self,
        chat_id: str | None = None,
        wait: float = 0.0,
    ) -> ChatResult | None:
        """Return the next ChatResult for a chat, blocking up to ``wait`` seconds."""
        deadline = time.monotonic() + wait if wait > 0 else 0.0
        with self._outbox_condition:
            while True:
                for i, (cid, result) in enumerate(self._outbox):
                    if chat_id is None or cid == chat_id:
                        popped = self._outbox[i]
                        del self._outbox[i]
                        return popped[1]
                if wait <= 0:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._outbox_condition.wait(timeout=remaining)

    def _create_direct_notifier(self) -> Notifier:
        """Create a notifier that bypasses the outbox if possible."""
        if self.config.harness.notifications.webhook_url:
            return WebhookNotifier(self.config.harness.notifications.webhook_url)
        token = self.config.harness.telegram.token
        if token:
            return TelegramNotifier(token, metrics=self._runtime.metrics)
        return NoopNotifier()

    def _send_restart_notices(self) -> None:
        """Notify recently active chats that the service has restarted.

        The notice is sent through a direct notifier (not the outbox) when
        outbox delivery is enabled, because the transport's DeliveryWorker may
        not be running yet at startup.
        """
        if not self.config.harness.notifications.enabled:
            return

        recent_cutoff = time.time() - 86400.0
        chat_ids: list[str] = []
        with self._lock:
            for chat_id, state in self._runtime._store.items():
                if not _is_telegram_chat_id(chat_id):
                    continue
                if not state.sessions:
                    continue
                latest = max(state.sessions.values(), key=lambda r: r.updated_at)
                if latest.updated_at >= recent_cutoff:
                    chat_ids.append(chat_id)

        if not chat_ids:
            return

        logger.info("Sending restart notice to %d recently active chat(s)", len(chat_ids))
        text = "System: service was restarted. You can resume the conversation at any time."
        notifier = self._create_direct_notifier()
        if isinstance(notifier, NoopNotifier):
            return

        for chat_id in chat_ids:
            self._safe_notifier_send(str(chat_id), text, notifier=notifier)
