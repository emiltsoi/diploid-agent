"""No-code state plugin that renders a prompt from a JSON file template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from devin_fleet_harness.config import PluginConfig
from devin_fleet_harness.plugins.base import StatePlugin
from devin_fleet_harness.runtime.plugin_runtime import PluginRuntime


class JsonStatePlugin(StatePlugin):
    """A plugin that loads `state_file` as JSON and renders `prompt_template`."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)

    def _load_state(self) -> dict[str, Any]:
        path = self.state_path()
        if path is None or not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        template = self.config.prompt_template
        if not template:
            return None
        state = self._load_state()
        block = template.format(**state)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block or None
