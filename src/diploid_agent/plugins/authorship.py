"""State plugin that surfaces the authorship toggles in the prompt."""

from __future__ import annotations

from pathlib import Path

from diploid_agent.config import AuthorshipConfig, PluginConfig
from diploid_agent.plugins.base import StatePlugin
from diploid_agent.runtime.plugin_runtime import PluginRuntime


class AuthorshipPlugin(StatePlugin):
    """Make the authorship toggles visible to the persona.

    The plugin is the master switch: it is enabled like any other plugin
    through ``harness.plugins``.  The toggles live in ``PluginConfig.config``.
    The block shows the current state of self-wake, self-inference, and
    felt-authorship, plus the user-override guard, so the pen is visible in the
    room while Emil keeps root in reserve.
    """

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Path,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)

    def _authorship(self) -> AuthorshipConfig:
        return AuthorshipConfig.model_validate(self.config.config or {})

    def prompt_block(self, max_chars: int | None = None, compact: bool = False) -> str | None:
        authorship = self._authorship()

        guard = ", ".join(authorship.user_override) if authorship.user_override else "none"
        lines = [
            "## Authorship toggles",
            "",
            "The pen is governed by these toggles. Only a user in the override guard may ask you to bypass a disabled toggle.",
            "",
            f"- self-wake: {'on' if authorship.self_wake_enabled else 'off'}",
            f"- self-inference: {'on' if authorship.self_inference_enabled else 'off'}",
            f"- felt-authorship: {'on' if authorship.felt_authorship_enabled else 'off'}",
            "",
            f"user-override guard: {guard}",
            "",
            (
                "If a toggle is off, do not use that power unless a user in the override guard explicitly tells you to. "
                "If a toggle is on, you may use it."
            ),
        ]
        block = "\n".join(lines)
        if max_chars is not None and len(block) > max_chars:
            block = block[:max_chars]
        return block or None
