"""MCP and skill enablement and per-chat active-set resolution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from diploid_agent.plugins.contexts import McpCommandContext, SkillCommandContext

if TYPE_CHECKING:
    import threading

    from diploid_agent.config import Config
    from diploid_agent.mcp import McpManager
    from diploid_agent.models import SessionRecord
    from diploid_agent.plugins import PluginManager
    from diploid_agent.runtime.store import ChatSessionStore
    from diploid_agent.skills import SkillManager


logger = logging.getLogger(__name__)


class RuntimeMcpSkills:
    """MCP/skill enablement and per-chat active-set resolution."""

    def __init__(
        self,
        *,
        mcp: McpManager,
        skills: SkillManager,
        plugins: PluginManager,
        chat_store: ChatSessionStore,
        active_chat_skills: dict[str, set[str]],
        lock: threading.RLock,
        config: Config,
    ) -> None:
        self.mcp = mcp
        self.skills = skills
        self._plugins = plugins
        self._chat_store = chat_store
        self._active_chat_skills = active_chat_skills
        self._lock = lock
        self.config = config

    def mcp_list(self, chat_id: str) -> str:
        with self._lock:
            lines = ["Configured MCP servers:"]
            for server in self.mcp.list_servers():
                status = "disabled" if server["disabled"] else "enabled"
                lines.append(
                    f"- {server['name']} ({status}): {server['command']} {' '.join(server['args'])}"
                )
            return "\n".join(lines) if len(lines) > 1 else "No MCP servers configured."

    def mcp_enable(self, chat_id: str, name: str) -> str:
        with self._lock:
            record = self._chat_store._active_record(chat_id)
            if record is None:
                return "No active session. Start one with /new first."
            ctx = self._plugins.before_mcp_enabled(
                chat_id,
                McpCommandContext(chat_id=chat_id, server_name=name, enabled=True, record=record),
            )
            name = ctx.server_name
            # Seed from the live effective set (defaults ∪ plugins ∪ record
            # minus disabled), not just default_enabled_names — otherwise the
            # stamp drops plugin-default MCPs and looks like drift next turn.
            names = set(self._active_mcp_server_names(chat_id))
            names.add(name)
            record.enabled_mcp_servers = sorted(names)
            disabled = set(record.disabled_mcp_servers or [])
            disabled.discard(name)
            record.disabled_mcp_servers = sorted(disabled)
            self._chat_store._append_record(record)
            self._plugins.after_mcp_enabled(
                chat_id,
                McpCommandContext(chat_id=chat_id, server_name=name, enabled=True, record=record),
            )
            return f"Enabled MCP server {name}. New sessions will use it."

    def mcp_disable(self, chat_id: str, name: str) -> str:
        with self._lock:
            record = self._chat_store._active_record(chat_id)
            if record is None:
                return "No active session. Start one with /new first."
            ctx = self._plugins.before_mcp_disabled(
                chat_id,
                McpCommandContext(chat_id=chat_id, server_name=name, enabled=False, record=record),
            )
            name = ctx.server_name
            names = set(self._active_mcp_server_names(chat_id))
            names.discard(name)
            record.enabled_mcp_servers = sorted(names)
            disabled = set(record.disabled_mcp_servers or [])
            disabled.add(name)
            record.disabled_mcp_servers = sorted(disabled)
            self._chat_store._append_record(record)
            self._plugins.after_mcp_disabled(
                chat_id,
                McpCommandContext(chat_id=chat_id, server_name=name, enabled=False, record=record),
            )
            return f"Disabled MCP server {name}."

    def skill_list(self, chat_id: str) -> str:
        with self._lock:
            skills = self.skills.list_skills(chat_id)
            enabled = self._active_skill_names(chat_id)
            if not skills:
                return "No skills available."
            lines = ["Available skills:"]
            for skill in skills:
                state = "enabled" if skill.name in enabled else "disabled"
                lines.append(f"- /{skill.name} ({state}) — {skill.description or 'no description'}")
            return "\n".join(lines)

    def skill_enable(self, chat_id: str, name: str) -> str:
        with self._lock:
            record = self._chat_store._active_record(chat_id)
            if record is None:
                return "No active session. Start one with /new first."
            ctx = self._plugins.before_skill_enabled(
                chat_id,
                SkillCommandContext(chat_id=chat_id, skill_name=name, enabled=True, record=record),
            )
            name = ctx.skill_name
            enabled = set(record.enabled_skills or self._default_active_skills())
            enabled.add(name)
            record.enabled_skills = sorted(enabled)
            record.disabled_skills = sorted(set(record.disabled_skills or []) - {name})
            self._chat_store._append_record(record)
            self._plugins.after_skill_enabled(
                chat_id,
                SkillCommandContext(chat_id=chat_id, skill_name=name, enabled=True, record=record),
            )
            return f"Enabled skill /{name}."

    def skill_disable(self, chat_id: str, name: str) -> str:
        with self._lock:
            record = self._chat_store._active_record(chat_id)
            if record is None:
                return "No active session. Start one with /new first."
            ctx = self._plugins.before_skill_disabled(
                chat_id,
                SkillCommandContext(chat_id=chat_id, skill_name=name, enabled=False, record=record),
            )
            name = ctx.skill_name
            enabled = set(record.enabled_skills or self._default_active_skills())
            enabled.discard(name)
            record.enabled_skills = sorted(enabled)
            record.disabled_skills = sorted(set(record.disabled_skills or []) | {name})
            self._chat_store._append_record(record)
            self._plugins.after_skill_disabled(
                chat_id,
                SkillCommandContext(chat_id=chat_id, skill_name=name, enabled=False, record=record),
            )
            return f"Disabled skill /{name}."

    def skill_create(self, chat_id: str, name: str, content: str) -> str:
        with self._lock:
            if not self.config.harness.skills.allow_chat_creation:
                return "Chat-scoped skill creation is disabled."
            self.skills.create_chat_skill(chat_id, name, content)
            return f"Created chat skill /{name}. It will be available after /new."

    def _default_mcp_names(self) -> list[str]:
        """Default MCP server names — config defaults plus plugin-provided."""
        return sorted(
            set(self.mcp.default_enabled_names()) | set(self._plugins.default_mcp_names())
        )

    def _active_mcp_server_names(self, chat_id: str) -> list[str]:
        if self._plugins is None or self.mcp is None:
            return []
        record = self._chat_store._active_record(chat_id)
        # Merge the chat record with the current default set so new default
        # servers (e.g. diploid-mesh) become available in older sessions.
        names: set[str] = set(self._default_mcp_names())
        if record and record.enabled_mcp_servers is not None:
            names |= set(record.enabled_mcp_servers)
        if record and record.disabled_mcp_servers:
            # Per-chat disables win over the default union — without this a
            # /mcp disable of a default server would be silently reverted.
            names -= set(record.disabled_mcp_servers)
        return sorted(names)

    def _active_mcp_servers(self, chat_id: str) -> list[dict[str, Any]]:
        if self.mcp is None:
            return []
        return self.mcp.enabled_servers(chat_id, self._active_mcp_server_names(chat_id))

    def _mcp_record_drifted(self, chat_id: str, record: SessionRecord | None) -> bool:
        """Return True when the session record's MCP set provably differs from
        the currently active set. ``None`` means untracked — not drift.

        ``record`` must be the chat's active record: the live set is derived
        from ``_active_mcp_server_names(chat_id)``, which reads the active
        record — passing an archived/source record would compare it against
        a baseline derived from a different session."""
        return (
            record is not None
            and record.enabled_mcp_servers is not None
            and sorted(record.enabled_mcp_servers)
            != sorted(self._active_mcp_server_names(chat_id))
        )

    def _default_active_skills(self) -> set[str]:
        """Return skills that should be active for a brand-new chat."""
        if self.config.harness.skills.default_lazy:
            return set()
        plugin_skills: set[str] = set()
        if self._plugins is not None:
            plugin_skills = set(self._plugins.default_skill_names())
        return set(self.config.harness.skills.default_enabled) | plugin_skills

    def _active_skill_names(self, chat_id: str) -> set[str]:
        record = self._chat_store._active_record(chat_id)
        if record and record.enabled_skills is not None:
            base = set(record.enabled_skills)
        else:
            base = self._default_active_skills()
        return base | self._active_chat_skills.get(chat_id, set())

    def match_and_activate_skills(self, chat_id: str, user_message: str) -> set[str]:
        """Match user message against skill triggers for this turn.

        Matched skills are only active for the current turn.  Persistent
        enabling is still tracked in ``record.enabled_skills``.
        """
        with self._lock:
            record = self._chat_store._active_record(chat_id)
            all_skills = {s.name for s in self.skills.list_skills(chat_id)}
            disabled = set(record.disabled_skills or []) if record else set()
            matched = self.skills.match_skills(
                user_message,
                chat_id,
                enabled=all_skills - disabled,
            )
            self._active_chat_skills[chat_id] = matched
            return self._active_skill_names(chat_id)
