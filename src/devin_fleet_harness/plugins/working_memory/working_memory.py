"""Working-memory plugin: per-chat scratchpad for intent, plan, and notes."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from devin_fleet_harness.config import PluginConfig
from devin_fleet_harness.plugins.base import StatePlugin
from devin_fleet_harness.runtime.plugin_runtime import PluginRuntime

DEFAULT_STATE: dict[str, Any] = {
    "intent": "",
    "plan": [],
    "open_questions": [],
    "notes": [],
}


class WorkingMemoryPlugin(StatePlugin):
    """Keeps a small, persistent working-memory scratchpad for each chat."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self._state: dict[str, Any] = self._load_state()

    def state_path(self) -> Path | None:
        if not self.config.state_file:
            return None
        chat_dir = self.sessions_root / self.chat_id.replace("/", "_")
        return chat_dir / self.config.state_file

    def _load_state(self) -> dict[str, Any]:
        path = self.state_path()
        state = copy.deepcopy(DEFAULT_STATE)
        if path is None or not path.exists():
            return state
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                state.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass
        return state

    def _save_state(self) -> None:
        path = self.state_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state, indent=2, default=str))

    def _set_intent(self, value: str | None) -> str:
        if not value:
            return "Usage: /state working_memory update intent <value>"
        self._state["intent"] = value
        self._save_state()
        return f"Intent set to: {value}"

    def _append_to(self, field: str, value: str | None) -> str:
        if not value:
            return f"Usage: /state working_memory update {field} <value>"
        if field not in DEFAULT_STATE:
            return f"Unknown field: {field}"
        self._state.setdefault(field, []).append(value)
        self._save_state()
        return f"Added to {field}: {value}"

    def _append_to_plan(self, value: str | None) -> str:
        return self._append_to("plan", value)

    def _append_open_question(self, value: str | None) -> str:
        return self._append_to("open_questions", value)

    def _append_note(self, value: str | None) -> str:
        return self._append_to("notes", value)

    def _clear(self, field: str | None = None) -> str:
        if field is None:
            self._state = copy.deepcopy(DEFAULT_STATE)
            self._save_state()
            return "Cleared working memory."
        if field not in DEFAULT_STATE:
            return f"Unknown field: {field}"
        self._state[field] = copy.deepcopy(DEFAULT_STATE[field])
        self._save_state()
        return f"Cleared working memory {field}."

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        intent = self._state.get("intent", "")
        plan = self._state.get("plan") or []
        questions = self._state.get("open_questions") or []
        notes = self._state.get("notes") or []

        if not intent and not plan and not questions and not notes:
            return None

        lines = ["## Working memory"]

        if intent:
            lines.append(f"- Intent: {intent}")

        if plan:
            lines.append("- Plan:")
            for item in plan:
                lines.append(f"  - {item}")

        if questions:
            lines.append("- Open questions:")
            for item in questions:
                lines.append(f"  - {item}")

        if notes:
            lines.append("- Notes:")
            for item in notes:
                lines.append(f"  - {item}")

        block = "\n".join(lines)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block

    def event(
        self,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> str:
        if event == "update":
            field = params.get("field")
            value = params.get("value")
            if value is None or field is None:
                field, value = _parse_field_value(raw_args or "")
            if not field:
                return "Usage: /state working_memory update <field> <value>"
            if field == "intent":
                return self._set_intent(value)
            if field in ("plan", "open_questions", "notes"):
                return self._append_to(field, value)
            return f"Unknown field: {field}"

        if event == "clear":
            field = params.get("field")
            if field is None and raw_args:
                field = raw_args.strip() or None
            return self._clear(field)

        if event == "state":
            return self.prompt_block() or "Working memory is empty."

        return "Usage: /state working_memory update <field> <value> | clear [field] | state"


def _parse_field_value(raw_args: str) -> tuple[str | None, str | None]:
    """Parse a leading field and the rest of the line as a value."""
    text = raw_args.strip()
    if not text:
        return None, None
    match = re.match(r'^\s*(?:"([^"]*)"|(\S+))(?:\s+(.*))?$', text, re.DOTALL)
    if not match:
        return None, None
    field = match.group(1) or match.group(2)
    value = (match.group(3) or "").strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return field, value
