"""No-code state plugin that renders a prompt from a JSON file template."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from diploid_agent.config import PluginConfig
from diploid_agent.plugins.base import StatePlugin
from diploid_agent.runtime.plugin_runtime import PluginRuntime


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

    def _state_mtime(self) -> float | None:
        path = self.state_path()
        if path is None or not path.exists():
            return None
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def prompt_block_changed(self, since: float | None = None) -> bool | None:
        mtime = self._state_mtime()
        if mtime is None:
            # No state file yet; let the harness decide by calling prompt_block.
            return None
        if since is None:
            return True
        return mtime > since

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        template = self.config.prompt_template
        if not template:
            return None
        state = self._load_state()
        block = template.format(**state)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block or None
