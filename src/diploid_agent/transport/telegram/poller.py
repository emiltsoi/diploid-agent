#!/usr/bin/env python3
"""Telegram long-polling transport.

Polls Telegram Bot API `getUpdates` and forwards each text message to the
configured runtime or harness `/chat` endpoint, then sends the reply back to
the user.
"""

from __future__ import annotations

import contextlib
import dataclasses
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

# The Telegram token is part of the request URL, so suppress httpx's default
# request logging to avoid leaking it.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("telegram_poll")

# Characters allowed in a saved attachment filename. Anything else (including
# path separators and dots-only names) is collapsed to "_".
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

# Message keys carrying downloadable files. ``photo`` is a size ladder; every
# other key is a single file object.
_ATTACHMENT_KEYS = (
    "document",
    "audio",
    "voice",
    "video",
    "video_note",
    "sticker",
    "animation",
)


def _extract_attachments(message: dict) -> tuple[TelegramAttachment, ...]:
    """Pull downloadable file descriptors out of a Telegram message."""
    out: list[TelegramAttachment] = []
    photos = message.get("photo")
    if isinstance(photos, list) and photos:
        # The same image at several resolutions; keep the largest variant.
        best = max(
            (p for p in photos if isinstance(p, dict) and p.get("file_id")),
            key=lambda p: p.get("file_size") or (p.get("width", 0) * p.get("height", 0)),
            default=None,
        )
        if best is not None:
            out.append(
                TelegramAttachment(
                    kind="photo",
                    file_id=best["file_id"],
                    file_size=best.get("file_size"),
                )
            )
    for key in _ATTACHMENT_KEYS:
        obj = message.get(key)
        if isinstance(obj, dict) and obj.get("file_id"):
            out.append(
                TelegramAttachment(
                    kind=key,
                    file_id=obj["file_id"],
                    file_name=obj.get("file_name"),
                    mime_type=obj.get("mime_type"),
                    file_size=obj.get("file_size"),
                )
            )
    return tuple(out)


def _safe_filename(name: str, *, fallback: str) -> str:
    """Reduce a Telegram-provided name to a single safe path segment."""
    stem = _UNSAFE_FILENAME.sub("_", Path(name).name).strip("._")
    return (stem or fallback)[:96]


from diploid_agent.transport.telegram.models import ChatInput, TelegramAttachment
from diploid_agent.transport.telegram.voice import TRANSCRIBABLE_KINDS, transcribe
from diploid_agent.transport.telegram.workers import (
    DeliveryWorker,
    TurnWorker,
    WakeDisplayWorker,
)

from .commands import TelegramCommandMixin
from .sender import TelegramSenderMixin
from .state import TelegramStateMixin


class TelegramPoller(TelegramCommandMixin, TelegramSenderMixin, TelegramStateMixin):
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
        sessions_root: Path | None = None,
        reply_preview_chars: int = 240,
        min_telegram_interval: float = 1.0,
        min_edit_message_interval: float = 2.0,
        max_telegram_retries: int = 3,
        max_telegram_backoff: float = 30.0,
        metrics: Any | None = None,
        message_format: str = "plain",
        code_style: str = "inline",
        attachments_enabled: bool = True,
        attachments_max_bytes: int = 20_000_000,
        attachments_dirname: str = "inbox",
        stt_provider: str = "none",
        stt_model: str = "small",
        stt_command: str = "",
        tts_provider: str = "none",
        tts_model_path: str = "",
        tts_command: str = "",
        tts_max_chars: int = 800,
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
            attachments_enabled=attachments_enabled,
            attachments_max_bytes=attachments_max_bytes,
            attachments_dirname=attachments_dirname,
            stt_provider=stt_provider,
            stt_model=stt_model,
            stt_command=stt_command,
            tts_provider=tts_provider,
            tts_model_path=tts_model_path,
            tts_command=tts_command,
            tts_max_chars=tts_max_chars,
        )
        self.state_dir = state_dir or Path("sessions") / ".poller-placeholders"
        # Attachments land in <sessions_root>/<chat_id>/<dirname>/, inside the
        # chat's ACP workspace, so a ``session:`` cron trigger can watch them.
        self.sessions_root = Path(sessions_root) if sessions_root else self.state_dir.parent
        self.reply_preview_chars = reply_preview_chars
        self._max_telegram_retries = max_telegram_retries
        self._max_telegram_backoff = max_telegram_backoff
        self.offset: int | None = None
        self._local = threading.local()
        self._clients: set[httpx.Client] = set()
        self._clients_lock = threading.Lock()
        self._client_timeout = 35.0
        self._stream_thoughts: dict[int, bool] = {}
        self._active_workers: dict[int, TurnWorker] = {}
        self._pending_inputs: dict[int, deque[ChatInput]] = {}
        self._delivery_workers: dict[int, DeliveryWorker] = {}
        self._global_delivery_worker: DeliveryWorker | None = None
        self._wake_displays: dict[int, WakeDisplayWorker] = {}
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

    def _harness_headers(self) -> dict[str, str]:
        """Auth headers for direct harness-URL calls (bypassing CommandHandler)."""
        return {"X-API-Key": self._api_key} if self._api_key else {}

    @property
    def client(self) -> httpx.Client:
        """Return a thread-local httpx.Client so threads do not share one."""
        client = getattr(self._local, "client", None)
        if client is None:
            client = httpx.Client(timeout=self._client_timeout)
            self._local.client = client
            with self._clients_lock:
                self._clients.add(client)
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
            with self._clients_lock:
                self._clients.discard(client)

    def _close_all_clients(self) -> None:
        """Close every httpx.Client created so far, across all threads."""
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            client.close()

    def _stream_thoughts_enabled(self, chat_id: int) -> bool:
        return self._stream_thoughts.get(chat_id, self._live_telegram_config.stream_thoughts)

    def _ensure_delivery_worker(self, chat_id: int) -> None:
        """Start the global DeliveryWorker if outbox delivery is enabled.

        The worker long-polls ``/outbox`` without a chat scope, so queued
        messages (including mesh wake replies) are delivered even when no
        Telegram user message has arrived yet.
        """
        with self._worker_lock:
            if self._global_delivery_worker is not None and self._global_delivery_worker.is_alive():
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
            self._global_delivery_worker = DeliveryWorker(self, chat_id=None)
            self._global_delivery_worker.start()

    def _start_wake_display(self, chat_id: int) -> None:
        """Register a WakeDisplayWorker for a wake-driven turn, if enabled.

        Called by DeliveryWorker when an outbox ``turn_started`` marker
        arrives. Skipped when the flag is off, a display is already live, or
        a TurnWorker owns the chat (a user turn streams its own reply).
        """
        if not self._live_telegram_config.wake_stream:
            return
        with self._worker_lock:
            if chat_id in self._wake_displays or chat_id in self._active_workers:
                return
            worker = WakeDisplayWorker(self, chat_id)
            self._wake_displays[chat_id] = worker
            worker.start()

    def _wake_display_for(self, chat_id: int) -> WakeDisplayWorker | None:
        with self._worker_lock:
            return self._wake_displays.get(chat_id)

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
            if (
                sent
                and chat_result.session_number is not None
                and chat_result.turn_number is not None
            ):
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

        # Prefer text, then caption.
        text = message.get("text") or message.get("caption", "")
        attachments = _extract_attachments(message)

        # Do not reply to messages the bot sent itself.
        if message.get("from", {}).get("is_bot"):
            return None

        if not chat_id or not (text or attachments):
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
            attachments=attachments,
        )

    def _download_attachment(
        self,
        chat_id: int,
        message_id: int,
        attachment: TelegramAttachment,
        max_bytes: int,
    ) -> Path:
        """Fetch one attachment from Telegram into the chat's inbox dir."""
        data = self._api("getFile", file_id=attachment.file_id)
        result = data.get("result") or {}
        file_path = result.get("file_path")
        if not file_path:
            raise RuntimeError(f"getFile returned no file_path for {attachment.file_id}")
        declared = result.get("file_size") or attachment.file_size
        if declared is not None and declared > max_bytes:
            raise RuntimeError(f"attachment too large ({declared} bytes)")

        raw_name = attachment.file_name or Path(file_path).name
        name = f"{message_id}-{_safe_filename(raw_name, fallback='file')}"
        dirname = _safe_filename(self._live_telegram_config.attachments_dirname, fallback="inbox")
        inbox = self.sessions_root / str(chat_id) / dirname
        inbox.mkdir(parents=True, exist_ok=True)
        dest = (inbox / name).resolve()
        if dest.parent != inbox.resolve():
            raise RuntimeError("attachment name escapes inbox")

        url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
        written = 0
        try:
            with self.client.stream("GET", url) as resp:
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_bytes(65536):
                        written += len(chunk)
                        if written > max_bytes:
                            raise RuntimeError(f"attachment exceeds {max_bytes} bytes")
                        f.write(chunk)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        return dest

    def _ingest_attachments(self, chat_input: ChatInput) -> ChatInput:
        """Download a message's attachments and append their paths to the text.

        Runs on the turn worker, off the poll loop. A failed or oversized file
        is annotated rather than dropping the message, so the agent can tell the
        user something was sent but could not be saved.
        """
        if not chat_input.attachments:
            return chat_input
        config = self._live_telegram_config
        if not config.attachments_enabled:
            return chat_input
        lines = []
        for att in chat_input.attachments:
            try:
                dest = self._download_attachment(
                    chat_input.chat_id,
                    chat_input.message_id,
                    att,
                    config.attachments_max_bytes,
                )
            except Exception as exc:
                logger.exception(
                    "Attachment %s (%s) for chat %s failed",
                    att.file_id,
                    att.kind,
                    chat_input.chat_id,
                )
                label = att.file_name or att.kind
                lines.append(f"[attachment could not be saved: {label} ({exc})]")
                continue
            rel = f"{dest.parent.name}/{dest.name}"
            desc = att.kind if att.mime_type is None else f"{att.kind}, {att.mime_type}"
            lines.append(f"[attachment saved: {rel} ({desc})]")
            if att.kind in TRANSCRIBABLE_KINDS:
                transcript = transcribe(dest, config)
                if transcript:
                    lines.append(f'[transcript: "{transcript}"]')
                elif config.stt_provider != "none":
                    lines.append("[transcript unavailable]")
        text = chat_input.text
        for line in lines:
            text = f"{text}\n{line}" if text else line
        return dataclasses.replace(chat_input, text=text)

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
        self._sync_bot_menu()
        try:
            # Wait briefly for the harness to come up, then start the global
            # outbox worker so mesh wakes, subagent completions and other outbox
            # items can be delivered before any new Telegram user message arrives.
            startup_deadline = time.monotonic() + 10.0
            while time.monotonic() < startup_deadline:
                if self._stop.is_set():
                    break
                self._ensure_delivery_worker(0)
                if (
                    self._global_delivery_worker is not None
                    and self._global_delivery_worker.is_alive()
                ):
                    break
                time.sleep(0.5)
            while not self._stop.is_set():
                # Keep trying to start the global outbox worker if it failed
                # during startup (e.g. the harness wasn't ready yet).
                self._ensure_delivery_worker(0)
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
        finally:
            self._close_all_clients()

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

        if self._handle_command(chat_input, command, arg):
            return

        logger.info("Message from chat %s: %r", chat_id, text[:80])
        with self._worker_lock:
            self._pending_inputs.setdefault(chat_id, deque()).append(chat_input)
            worker = self._active_workers.get(chat_id)
            if worker is None or not worker.is_alive():
                next_input = self._pending_inputs[chat_id].popleft()
                worker = TurnWorker(self, next_input)
                self._active_workers[chat_id] = worker
                worker.start()
