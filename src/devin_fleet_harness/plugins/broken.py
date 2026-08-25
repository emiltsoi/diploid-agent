"""No-op fallback plugin for load failures."""

from __future__ import annotations

from devin_fleet_harness.plugins.base import StatePlugin


class FailedPlugin(StatePlugin):
    """A placeholder plugin used when a configured plugin fails to load."""

    def __init__(self, config, chat_id, sessions_root, runtime=None, error=None):
        super().__init__(config, chat_id, sessions_root, runtime=runtime)
        self.error = error

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        return None
