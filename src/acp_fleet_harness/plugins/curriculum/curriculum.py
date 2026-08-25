"""Curriculum plugin: simple language-learning state."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.plugins.base import StatePlugin
from acp_fleet_harness.runtime.plugin_runtime import PluginRuntime


class CurriculumPlugin(StatePlugin):
    """Tracks a small vocabulary and target language for a chat."""

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
        if path is None or not path.exists():
            return {"target_language": None, "unit": None, "vocabulary": []}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {"target_language": None, "unit": None, "vocabulary": []}

    def _save_state(self) -> None:
        path = self.state_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._state, indent=2, default=str))

    def _add_word(self, word: str, translation: str) -> str:
        if not word or not translation:
            return "Usage: /state curriculum add_word <word> <translation>"
        self._state.setdefault("vocabulary", []).append({"word": word, "translation": translation})
        self._save_state()
        return f"Added '{word}' -> '{translation}' to vocabulary."

    def _set_target_language(self, language: str) -> str:
        if not language:
            return "Usage: /state curriculum set_target_language <language>"
        self._state["target_language"] = language
        self._save_state()
        return f"Target language set to {language}."

    def _set_unit(self, unit: str) -> str:
        if not unit:
            return "Usage: /state curriculum set_unit <unit>"
        self._state["unit"] = unit
        self._save_state()
        return f"Unit set to {unit}."

    def event(
        self,
        *,
        event: str | None = None,
        raw_args: str | None = None,
        **params: Any,
    ) -> str:
        if event == "add_word":
            word = params.get("word") or ""
            translation = params.get("translation") or ""
            if raw_args and not (word and translation):
                parts = _split_unquoted(raw_args)
                if len(parts) >= 2:
                    word, translation = parts[0], parts[1]
            return self._add_word(word, translation)

        if event == "set_target_language":
            language = params.get("language") or raw_args or ""
            return self._set_target_language(language)

        if event == "set_unit":
            unit = params.get("unit") or raw_args or ""
            return self._set_unit(unit)

        return (
            "Usage: /state curriculum "
            "add_word <word> <translation> | "
            "set_target_language <language> | "
            "set_unit <unit>"
        )

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        lines = ["## Curriculum"]

        target = self._state.get("target_language")
        if target:
            lines.append(f"- Target language: {target}")

        unit = self._state.get("unit")
        if unit:
            lines.append(f"- Current unit: {unit}")

        vocab = self._state.get("vocabulary") or []
        if vocab:
            lines.append(f"- Vocabulary ({len(vocab)} words):")
            for entry in vocab[-10:]:
                lines.append(f"  - {entry['word']}: {entry['translation']}")

        if len(lines) == 1:
            return None

        block = "\n".join(lines)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block


def _split_unquoted(text: str) -> list[str]:
    """Split on whitespace, respecting double-quoted strings."""
    parts = re.findall(r'"([^"]*)"|(\S+)', text)
    return [quoted or unquoted for quoted, unquoted in parts]
