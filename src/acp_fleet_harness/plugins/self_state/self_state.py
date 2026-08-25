"""Self-state continuity plugin."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.models import SessionRecord
from acp_fleet_harness.plugins.base import StatePlugin
from acp_fleet_harness.plugins.contexts import RecordTurnContext
from acp_fleet_harness.runtime.plugin_runtime import PluginRuntime


class SelfStatePlugin(StatePlugin):
    """Save and resume a first-person self-state note across sessions."""

    _SELF_STATE_RE = re.compile(
        r"<self_state>(.*?)</self_state>", re.IGNORECASE | re.DOTALL
    )
    _HEADER = "## State I am resuming from"
    _INSTRUCTION = (
        "At the end of your reply, append a private `<self_state>` block in first "
        "person, present tense, describing your new state. It will be saved and "
        "hidden from the user. Example: "
        "`<self_state>I am warm and curious.</self_state>`"
    )
    _INSTRUCTION_LEN = len(_INSTRUCTION)
    _HEADER_LEN = len(_HEADER)
    _SEPARATOR_LEN = 4  # two "\n\n" separators

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Any,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._state_path: Path = self._chat_dir() / self.config.state_file

    def _chat_dir(self) -> Path:
        return self.sessions_root / self.chat_id.replace("/", "_")

    def _load_state(self) -> str:
        try:
            return self._state_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    def _save_state(self, text: str) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(text, encoding="utf-8")

    def _extract_self_state(self, reply: str) -> tuple[str, str | None]:
        matches = list(self._SELF_STATE_RE.finditer(reply))
        if not matches:
            return reply, None
        note = matches[-1].group(1).strip()
        stripped = self._SELF_STATE_RE.sub("", reply).rstrip()
        return stripped, note

    def _fallback_note(self, record: SessionRecord | None, reply: str) -> str:
        if record is not None and record.last_stop_reason != "completed":
            return "I was in the middle of replying when I stopped. I want to continue."
        snippet = reply.strip()
        if len(snippet) <= 100:
            note = f"I just replied to the user: {snippet}"
        else:
            note = "I just replied to the user."
        return note[:200]

    def before_record_turn(self, context: RecordTurnContext) -> RecordTurnContext:
        stripped, note = self._extract_self_state(context.reply)
        if note is not None:
            self._save_state(note)
            context.reply = stripped
        else:
            self._save_state(self._fallback_note(context.record, context.reply))
        return context

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        note = self._load_state()
        if not note:
            return None
        block = f"{self._HEADER}\n\n{note}\n\n{self._INSTRUCTION}"
        if max_chars is not None and len(block) > max_chars:
            note_budget = (
                max_chars
                - self._HEADER_LEN
                - self._INSTRUCTION_LEN
                - self._SEPARATOR_LEN
            )
            if note_budget > 0:
                block = (
                    f"{self._HEADER}\n\n{note[:note_budget]}\n\n{self._INSTRUCTION}"
                )
            else:
                block = block[:max_chars]
        return block
