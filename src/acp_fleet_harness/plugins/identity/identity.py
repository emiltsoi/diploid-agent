"""Identity/self-narrative plugin: per-chat overlay on top of persona files."""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.plugins.base import StatePlugin, TurnInfo
from acp_fleet_harness.runtime.plugin_runtime import PluginRuntime


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class IdentityPlugin(StatePlugin):
    """Keeps a chat-scoped self-narrative in chat_SELF.md with edit history."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._state_path: Path | None = self.state_path()

    def state_path(self) -> Path | None:  # type: ignore[override]
        if not self.config.state_file:
            return None
        chat_dir = self.sessions_root / self.chat_id.replace("/", "_")
        return chat_dir / self.config.state_file

    def _history_path(self) -> Path | None:
        if self._state_path is None:
            return None
        return self._state_path.parent / "chat_SELF_history.jsonl"

    def _read_text(self) -> str:
        if self._state_path is None or not self._state_path.exists():
            return ""
        return self._state_path.read_text(encoding="utf-8")

    def _read_content(self) -> str | None:
        text = self._read_text()
        if not text:
            return None
        parts = text.split("\n\n", 1)
        if parts[0].startswith("<!-- identity-file:"):
            content = parts[1] if len(parts) > 1 else ""
        else:
            content = text
        content = content.strip()
        return content if content else None

    def _write_file(self, reason: str, content: str) -> str:
        if self._state_path is None:
            return ""
        header = (
            "<!-- identity-file: chat_SELF.md -->\n"
            "<!-- provenance: chat-scoped self-narrative; "
            "persona SOUL.md and MEMORY.md stay canonical -->\n"
            f"<!-- last_edit: {reason} @ {_now_iso()} -->"
        )
        full = f"{header}\n\n{content}"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(full, encoding="utf-8")
        return full

    def _append_history(self, old_text: str, new_text: str, reason: str) -> None:
        path = self._history_path()
        if path is None:
            return
        record = {
            "old_text": old_text,
            "new_text": new_text,
            "reason": reason,
            "at": time.time(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _update(self, content: str, reason: str = "manual") -> str:
        old_text = self._read_text()
        new_text = self._write_file(reason, content)
        self._append_history(old_text, new_text, reason)
        return f"Self-narrative updated: {content}"

    def _clear(self, reason: str = "manual") -> str:
        old_text = self._read_text()
        new_text = self._write_file(reason, "")
        self._append_history(old_text, new_text, reason)
        return "Self-narrative cleared."

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        content = self._read_content()
        if content is None:
            return None
        block = f"## Who I am right now\n\n{content}"
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block

    def _history(self, limit: int = 10) -> list[dict[str, Any]]:
        path = self._history_path()
        if path is None or not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records[-limit:]

    def event(
        self,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> str:
        if event == "update":
            content = params.get("content")
            if content is None and raw_args:
                content = raw_args.strip()
            if not content:
                return "Usage: /state identity update <content> [reason <reason>]"
            reason = params.get("reason") or "manual"
            return self._update(str(content), reason)

        if event == "clear":
            reason = params.get("reason") or "manual"
            return self._clear(reason)

        if event == "state":
            return self.prompt_block() or "No self-narrative set."

        if event == "history":
            return json.dumps(self._history(params.get("limit", 10)), default=str, indent=2)

        return "Usage: /state identity update <content> [reason <reason>] | clear | state"

    def on_turn_end(self, turn: TurnInfo) -> None:
        if not self.config.config.get("auto_rewrite"):
            return
        pattern = re.compile(r"```identity\s*\n(.*?)\n```", re.DOTALL)
        for match in pattern.finditer(turn.reply):
            content = match.group(1).strip()
            if content:
                self._update(content, "model-auto")
