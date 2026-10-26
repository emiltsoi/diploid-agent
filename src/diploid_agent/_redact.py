"""Redact Telegram bot tokens from log records, process-wide.

httpx exceptions embed the full request URL — including the ``bot<token>``
path segment — in their message, so any ``logger.exception`` on a failed send
writes the token to the log. Per-logger filters can't cover the ``telegram_poll``
and ``diploid_agent.*`` trees plus httpx's own records; a LogRecord factory
scrubs every record at creation regardless of logger, handler, or formatter.
"""

from __future__ import annotations

import logging
import re
import traceback
from collections.abc import Mapping

_BOT_TOKEN = re.compile(r"bot\d{6,}:[A-Za-z0-9_-]{20,}")
_REDACTED = "bot<redacted>"


def _scrub(text: str) -> str:
    return _BOT_TOKEN.sub(_REDACTED, text)


def _scrub_args(args: object) -> object:
    if isinstance(args, str):
        return _scrub(args) if "bot" in args else args
    if isinstance(args, tuple):
        return tuple(_scrub(a) if isinstance(a, str) and "bot" in a else a for a in args)
    if isinstance(args, Mapping):
        return {k: (_scrub(v) if isinstance(v, str) and "bot" in v else v) for k, v in args.items()}
    return args


def install() -> None:
    base = logging.getLogRecordFactory()

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = base(*args, **kwargs)
        if isinstance(record.msg, str) and "bot" in record.msg:
            record.msg = _scrub(record.msg)
        if record.args:
            record.args = _scrub_args(record.args)
        if record.exc_info:
            rendered = "".join(traceback.format_exception(*record.exc_info))
            if "bot" in rendered:
                record.exc_text = _scrub(rendered)
                record.exc_info = None
        return record

    logging.setLogRecordFactory(factory)
