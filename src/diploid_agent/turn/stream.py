"""Per-turn ACP stream callbacks.

``TurnStream`` holds the ``on_chunk``/``on_update`` callbacks handed to the
engine for a single turn. It updates the ``ActiveTurn`` buffer (message text,
thought text, tool-call side effects), notifies the turn's condition, and
emits ``on_partial`` plugin hooks. Previously these closures were copied
between ``TurnProcess.process`` and ``TurnDispatch.continue_turn``.
"""

from __future__ import annotations

import time
from typing import Any

from diploid_agent.models import PartialTurn


class TurnStream:
    """Stream callbacks for one in-flight turn on one chat."""

    def __init__(self, runtime: Any, chat_id: str) -> None:
        self._runtime = runtime
        self._chat_id = chat_id

    def _maybe_emit_partial(self) -> None:
        a = self._runtime._active_turns.get(self._chat_id)
        if a is None:
            return
        record = self._runtime._active_record(self._chat_id)
        self._runtime._plugins.on_partial(
            self._chat_id,
            PartialTurn.from_active(a, record),
        )

    def on_chunk(self, text: str) -> None:
        with self._runtime._lock:
            a = self._runtime._active_turns.get(self._chat_id)
            if a:
                a.append_full_text(text)
                a.recompute_message_text()
        if a:
            with a._condition:
                a._condition.notify_all()
        self._maybe_emit_partial()

    def on_update(self, update: dict[str, Any]) -> None:
        session_update = update.get("sessionUpdate")
        if session_update in ("tool_call", "tool_call_update"):
            with self._runtime._lock:
                a = self._runtime._active_turns.get(self._chat_id)
                if a:
                    raw_content = update.get("content") or {}
                    if isinstance(raw_content, list):
                        content = {}
                        for item in raw_content:
                            if isinstance(item, dict):
                                content = item
                                break
                    elif isinstance(raw_content, dict):
                        content = raw_content
                    else:
                        content = {}
                    title = (
                        content.get("title")
                        or content.get("kind")
                        or content.get("toolCallId")
                        or "tool"
                    )
                    status = content.get("status") or "running"
                    now = time.time()
                    a.last_side_effect = f"{title} ({status})"[:160]
                    a.last_side_effect_at = now
                    a.side_effects.append({"title": title, "status": status, "at": now})
            self._maybe_emit_partial()
            return
        if session_update not in ("agent_thought", "agent_thought_chunk"):
            return
        content = update.get("content", {})
        if isinstance(content, list):
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        elif content.get("type") == "text":
            text = content.get("text", "")
        else:
            text = ""
        if not text:
            return
        with self._runtime._lock:
            a = self._runtime._active_turns.get(self._chat_id)
            if a:
                a.append_thought_text(text)
                a.recompute_message_text()
        if a:
            with a._condition:
                a._condition.notify_all()
        self._maybe_emit_partial()
