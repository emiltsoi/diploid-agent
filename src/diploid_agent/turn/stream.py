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
        record = self._runtime.active_record(self._chat_id)
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
                    # ACP carries title/kind/status/toolCallId at the top
                    # level of the update; content is the content-block
                    # array. Check top level first, content as fallback.
                    title = (
                        update.get("title")
                        or update.get("kind")
                        or update.get("toolCallId")
                        or content.get("title")
                        or "tool"
                    )
                    status = update.get("status") or content.get("status") or "running"
                    # Exec titles are terminal ids ("exec:0#hash"); prefer the
                    # real command line from rawInput when the tool provides it.
                    raw_input = update.get("rawInput")
                    if isinstance(raw_input, dict):
                        for key in ("command", "CommandLine", "commandLine", "cmd"):
                            cmd = raw_input.get(key)
                            if isinstance(cmd, str) and cmd.strip():
                                title = f"{update.get('kind') or 'exec'}: {cmd.strip()}"
                                break
                    now = time.time()
                    composed = f"{title} ({status})"[:160]
                    # Notify only when the displayed string actually changes:
                    # progress chunks with the same title+status compose the
                    # same line and would only spam long-poll wakes.
                    changed = composed != a.last_side_effect
                    if changed:
                        a.last_side_effect = composed
                        a.last_side_effect_at = now
                    # Record rawInput keys so breadcrumbs reveal which fields
                    # the agent actually emits when our guesses miss.
                    entry = {"title": title, "status": status, "at": now}
                    if isinstance(raw_input, dict) and raw_input:
                        entry["input"] = {
                            k: (v[:80] if isinstance(v, str) else v)
                            for k, v in list(raw_input.items())[:6]
                        }
                    a.side_effects.append(entry)
            if a and changed:
                with a._condition:
                    a._condition.notify_all()
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
