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
