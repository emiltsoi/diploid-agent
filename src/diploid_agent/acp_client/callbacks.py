"""Prompt-callback pump: serializes harness callbacks off the reader thread."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diploid_agent.acp_client.state import AcpClientState

logger = logging.getLogger(__name__)


class AcpCallbackPump:
    """Serializes prompt callbacks off the transport reader thread.

    Prompt callbacks (``on_chunk``/``on_update``) are harness code that can
    block on locks or I/O.  Running them on the ACP loop starves the stdout
    reader: the pipe backs up and the child wedges on write.  Dispatch them
    on a dedicated worker thread instead.  The queue is bounded: a wedged
    callback must never backpressure the reader, so overflow drops work
    instead of blocking.
    """

    def __init__(self, state: AcpClientState) -> None:
        # Shared mutable state; a test fake without ``_state`` is used as the
        # state namespace directly (its ``_x`` attrs stand in for the fields).
        self._state = state
        self._cb_queue: queue.Queue[tuple[Callable[[Any], None], Any] | None] = queue.Queue(
            maxsize=2048
        )
        self._cb_dropped = 0
        self._cb_thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the callback worker thread for this transport generation."""
        self._cb_thread = threading.Thread(
            target=self._cb_worker, name="acp-prompt-cb", daemon=True
        )
        self._cb_thread.start()

    async def stop(self) -> None:
        """Stop the prompt-callback worker after queued callbacks drain.

        Only enqueue the sentinel when a worker actually ran -- a stale
        sentinel would make the next generation's worker exit immediately.
        The queue is bounded, so make room for the sentinel if it is full;
        the join timeout below bounds the wait even if the worker is wedged.
        """
        cb_thread = self._cb_thread
        self._cb_thread = None
        if cb_thread is not None:
            try:
                self._cb_queue.put_nowait(None)
            except queue.Full:
                try:
                    self._cb_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._cb_queue.put_nowait(None)
                except queue.Full:
                    pass
            if cb_thread.is_alive():
                await asyncio.to_thread(cb_thread.join, 2.0)

    def _route_update(self, msg: dict[str, Any]) -> None:
        """Route a `session/update` notification to its in-flight prompt."""
        params = msg.get("params", {})
        session_id = params.get("sessionId")
        update = params.get("update", {})
        if not session_id:
            return

        with self._state._lock:
            self._state._last_progress_at = time.monotonic()

        prompt = self._state._active_prompts.get(session_id)
        if prompt is None:
            # Fallback: if only one prompt is active, route to it.
            if len(self._state._active_prompts) == 1:
                prompt = next(iter(self._state._active_prompts.values()))
            else:
                logger.debug("No prompt for update session %s", session_id)
                return

        prompt.updates.append(update)
        if prompt.on_update:
            self._dispatch_cb(prompt.on_update, update)

        kind = update.get("sessionUpdate")
        if kind in ("agent_message", "agent_message_chunk"):
            for text in self._text_from_content(update.get("content", {})):
                if text:
                    prompt.chunks.append(text)
                    if prompt.on_chunk:
                        self._dispatch_cb(prompt.on_chunk, text)

    @staticmethod
    def _text_from_content(content: Any) -> list[str]:
        """Return all text blocks from an ACP content payload."""
        if isinstance(content, list):
            return [b.get("text", "") for b in content if b.get("type") == "text"]
        if isinstance(content, dict) and content.get("type") == "text":
            return [content.get("text", "")]
        return []

    def _dispatch_cb(self, cb: Callable[[Any], None], arg: Any) -> None:
        """Queue a prompt callback for the worker thread (never on the loop).

        When no worker exists (transport never started, or already closed)
        the callback runs inline -- in that state there is no live reader to
        starve, so invocation is safe and preserves delivery for tests and
        teardown edges.
        """
        if self._cb_thread is None:
            try:
                cb(arg)
            except Exception:
                logger.exception("ACP prompt callback failed")
            return
        try:
            self._cb_queue.put_nowait((cb, arg))
        except queue.Full:
            # A wedged callback worker must never backpressure the reader.
            # Dropped chunks only degrade streamed previews; the reply text
            # is still collected in prompt.chunks.
            self._cb_dropped += 1
            if self._cb_dropped == 1 or self._cb_dropped % 100 == 0:
                logger.warning(
                    "ACP prompt callback queue full; dropped %d callbacks",
                    self._cb_dropped,
                )

    def _cb_worker(self) -> None:
        """Run prompt callbacks sequentially on a dedicated thread.

        ``on_chunk``/``on_update`` call into harness code that acquires
        ``runtime._lock`` and can block on slow plugin or memory work.  Doing
        that on the ACP loop would starve the stdout reader: the pipe fills,
        the child blocks on write, and the turn hangs with no error.
        """
        while True:
            item = self._cb_queue.get()
            if item is None:
                return
            cb, arg = item
            try:
                cb(arg)
            except Exception:
                logger.exception("ACP prompt callback failed")
