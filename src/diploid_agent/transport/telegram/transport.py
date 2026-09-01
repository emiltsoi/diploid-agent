"""Telegram Transport implementation and CLI entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path
from typing import Any

from diploid_agent.config import Config
from diploid_agent.metrics import MetricsCollector
from diploid_agent.transport.base import OutboundMessage, RuntimeAPI, Transport
from diploid_agent.transport.telegram.poller import TelegramPoller

logger = logging.getLogger("telegram_poll")


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
            delivery_workers = list(self._poller._delivery_workers.values())
        for worker in workers:
            worker.stop()
        for worker in delivery_workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=5.0)
        for worker in delivery_workers:
            worker.join(timeout=5.0)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        self._poller._close_client()

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
    # otherwise it gives up on a turn that is still running. If the harness has
    # no hard timeout, the poller also waits indefinitely.
    reply_timeout = config.engine.timeout + 30.0 if config.engine.timeout is not None else None
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
        message_format=config.harness.telegram.message_format,
        code_style=config.harness.telegram.code_style,
        state_dir=config.harness.sessions_root / ".poller-placeholders",
        reply_preview_chars=config.harness.memory.max_bot_reply_quote_chars,
        metrics=metrics,
    )
    poller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
