"""Dataclasses used by the ACP client."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections import deque
from collections.abc import Callable
from typing import Any

# Bound the per-prompt update buffer.  `devin acp` session/update payloads can
# carry raw tool output (observed 87-274 KB each) and a prompt may run for
# hours, so an unbounded list accumulates real memory.  The retained tail is
# telemetry -- the live on_update callback sees every update -- so dropping the
# oldest entries is safe.
_PROMPT_UPDATES_MAXLEN = 256


@dataclasses.dataclass
class AcpPromptResult:
    """Result of a single `session/prompt` turn."""

    reply: str
    session_id: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    cancelled: bool = False
    partial: bool = False
    timed_out: bool = False
    updates: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Prompt:
    """In-flight ACP prompt state."""

    session_id: str
    prompt_id: int
    text: str
    future: asyncio.Future[dict[str, Any]]
    cancel_done: asyncio.Future[None]
    soft_timeout: float | None = None
    started_at: float = dataclasses.field(default_factory=time.monotonic)
    chunks: list[str] = dataclasses.field(default_factory=list)
    updates: deque[dict[str, Any]] = dataclasses.field(
        default_factory=lambda: deque(maxlen=_PROMPT_UPDATES_MAXLEN)
    )
    on_chunk: Callable[[str], None] | None = None
    on_update: Callable[[dict[str, Any]], None] | None = None
    cancelled: bool = False
    timed_out: bool = False
