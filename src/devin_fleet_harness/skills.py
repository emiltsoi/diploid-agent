"""Skill discovery and syncing for Devin ACP sessions."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    source: str  # e.g. "persona/example/review" or "chat/review"
    path: Path
    description: str | None = None
    argument_hint: str | None = None
    model: str | None = None
    subagent: bool = False
    agent: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    permissions: dict[str, Any] = field(default_factory=dict)
    triggers: list[str] = field(default_factory=lambda: ["user", "model"])
    content: str = ""

    def to_request(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "allowed_tools": self.allowed_tools,
            "permissions": self.permissions,
            "triggers": self.triggers,
            "content": self.content,
        }


class SkillManager:
    """Discover and sync skills into a chat workspace."""

    def __init__(
        self,
        personas_root: Path,
        shared_root: Path,
        chat_cwd_root: Path | None = None,
    ):
        self.personas_root = Path(personas_root)
        self.shared_root = Path(shared_root)
        self.chat_cwd_root = Path(chat_cwd_root) if chat_cwd_root else None

    def _chat_skill_root(self, chat_id: str) -> Path | None:
        if self.chat_cwd_root is None:
            return None
        return self.chat_cwd_root / chat_id / ".devin" / "skills"

    def _skill_dirs(self, chat_id: str | None) -> list[Path]:
        dirs: list[Path] = []
        if chat_id and self.chat_cwd_root:
            chat_root = self._chat_skill_root(chat_id)
            if chat_root and chat_root.exists():
                dirs.append(chat_root)
        if (self.shared_root / "skills").exists():
            dirs.append(self.shared_root / "skills")
        if self.personas_root.exists():
            for persona_dir in self.personas_root.iterdir():
                if persona_dir.is_dir() and (persona_dir / "skills").exists():
                    dirs.append(persona_dir / "skills")
        return dirs

    @staticmethod
    def _load_skill(path: Path, source: str) -> Skill | None:
        if not path.is_dir():
            return None
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            return None
        text = skill_md.read_text(encoding="utf-8")
        if not text.startswith("---"):
            return None
        parts = re.split(r"\n---\n", text.removeprefix("---"), maxsplit=1)
        if len(parts) != 2:
            return None
        try:
            front = yaml.safe_load(parts[0]) or {}
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(front, dict):
            return None
        return Skill(
            name=front.get("name", path.name),
            source=source,
            path=skill_md,
            description=front.get("description"),
            argument_hint=front.get("argument-hint"),
            model=front.get("model"),
            subagent=bool(front.get("subagent", False)),
            agent=front.get("agent"),
            allowed_tools=front.get("allowed-tools", []),
            permissions=front.get("permissions", {}),
            triggers=front.get("triggers", ["user", "model"]),
            content=parts[1].strip(),
        )

    def list_skills(self, chat_id: str | None = None) -> list[Skill]:
        """Return all available skills, chat-scoped first."""
        seen: set[str] = set()
        skills: list[Skill] = []
        for root in self._skill_dirs(chat_id):
            source = "shared"
            if chat_id and self.chat_cwd_root and self._chat_skill_root(chat_id) == root:
                source = f"chat/{chat_id}"
            elif self.personas_root in root.parents:
                source = f"persona/{root.parent.name}"
            for path in root.iterdir():
                if not path.is_dir():
                    continue
                skill = self._load_skill(path, source)
                if skill and skill.name not in seen:
                    seen.add(skill.name)
                    skills.append(skill)
        return skills

    def skill(self, name: str, chat_id: str | None = None) -> Skill | None:
        for skill in self.list_skills(chat_id):
            if skill.name == name:
                return skill
        return None

    def sync_to_chat(self, chat_id: str, cwd: Path, enabled: set[str] | None = None) -> None:
        """Copy enabled skills into chat cwd/.devin/skills so devin acp discovers them."""
        if enabled is None:
            enabled = {s.name for s in self.list_skills(chat_id)}
        target = cwd / ".devin" / "skills"
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        for name in enabled:
            skill = self.skill(name, chat_id)
            if not skill:
                continue
            skill_target = target / name
            skill_target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skill.path, skill_target / "SKILL.md")

    def create_chat_skill(self, chat_id: str, name: str, content: str) -> Path:
        if self.chat_cwd_root is None:
            raise RuntimeError("chat_cwd_root not set")
        target = self._chat_skill_root(chat_id) / name
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath("SKILL.md").write_text(content, encoding="utf-8")
        return target

    def match_skills(
        self,
        user_message: str,
        chat_id: str | None = None,
        *,
        enabled: set[str] | None = None,
    ) -> set[str]:
        """Return skill names whose trigger phrases appear in the user message.

        If `enabled` is provided, only consider those names; otherwise consider
        all available skills.
        """
        text = user_message.lower()
        candidates = self.list_skills(chat_id)
        if enabled is not None:
            enabled_lower = {n.lower() for n in enabled}
            candidates = [s for s in candidates if s.name.lower() in enabled_lower]

        matched: set[str] = set()
        for skill in candidates:
            slash = f"/{skill.name.lower()}"
            if slash in text:
                matched.add(skill.name)
                continue
            for trigger in skill.triggers:
                if trigger in ("user", "model"):
                    continue
                if trigger.lower() in text:
                    matched.add(skill.name)
                    break
        return matched

    def skill_index_text(
        self,
        chat_id: str | None = None,
        active: set[str] | None = None,
    ) -> str | None:
        """Build a compact, prompt-friendly index of available skills."""
        skills = self.list_skills(chat_id)
        if not skills:
            return None

        active = active or set()
        lines: list[str] = ["## Available skills", ""]
        for skill in skills:
            slash = f"/{skill.name}"
            marker = " (active)" if skill.name in active else ""
            lines.append(f"- **{skill.name}**{marker}: {skill.description or 'no description'}")
            if skill.argument_hint:
                lines.append(f"  - argument: `{skill.argument_hint}`")
            if skill.triggers and not all(t in ("user", "model") for t in skill.triggers):
                lines.append(f"  - triggers: {', '.join(skill.triggers)}")
            if skill.name in active:
                lines.append(f"  - invoke: say `{slash}` or a trigger phrase")

        return "\n".join(lines)

    def active_skills_text(self, active: set[str], chat_id: str | None = None) -> str | None:
        """Return the full SKILL.md content of the currently active skills."""
        if not active:
            return None

        parts: list[str] = []
        for name in sorted(active):
            skill = self.skill(name, chat_id)
            if not skill:
                continue
            header = [f"## Skill: {skill.name}"]
            if skill.description:
                header.append(f"**Description:** {skill.description}")
            if skill.allowed_tools:
                header.append(f"**Allowed tools:** {', '.join(skill.allowed_tools)}")
            if skill.argument_hint:
                header.append(f"**Argument hint:** `{skill.argument_hint}`")
            parts.append("\n".join(header) + "\n\n" + skill.content)

        if not parts:
            return None
        return "\n\n".join(parts)
