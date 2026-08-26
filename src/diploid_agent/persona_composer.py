"""Compose the identity prompt for a persona from on-disk files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from diploid_agent.config import PersonaConfig

DEFAULT_IDENTITY_FILES = [
    "SOUL.md",
    "AGENTS.md",
]


@dataclass
class PersonaPrompt:
    """Result of composing a persona identity prompt.

    The `text` field contains the identity sections (SOUL, AGENTS, and fleet
    shared context). The memory fields are retained for backward compatibility
    while the `MemoryManager` loads and caps the persona `MEMORY.md`.
    """

    text: str
    memory_text: str = ""
    memory_truncated: bool = False
    memory_path: Path | None = None
    limit: int = 0
    loaded: int = 0
    total: int = 0


def _trim_to_section(text: str, limit: int) -> str:
    """Return the first `limit` characters, rounded down to a section break."""
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    # Try to end on a blank line; otherwise end on the last newline.
    last_blank = candidate.rfind("\n\n")
    if last_blank > 0:
        return text[:last_blank]
    last_newline = candidate.rfind("\n")
    if last_newline > 0:
        return text[:last_newline]
    return candidate


def compose_persona(config: PersonaConfig) -> PersonaPrompt:
    """Return the persona identity as a structured prompt.

    The returned `PersonaPrompt.text` contains only the identity files
    (`SOUL.md`, `AGENTS.md`, and the fleet `shared/AGENTS.md`). The persona
    `MEMORY.md` content is no longer loaded here; it is loaded and capped by
    `MemoryManager` during prompt assembly.
    """
    parts: list[str] = []
    root = config.profile_root

    for filename in DEFAULT_IDENTITY_FILES:
        path = root / filename
        if path.exists():
            parts.append(f"## {path.stem}\n\n{path.read_text()}")

    fleet_root = config.fleet_root or root.parent
    shared = (fleet_root / "shared" / "AGENTS.md") if fleet_root else None
    if shared and shared.exists():
        parts.append(f"## Fleet shared AGENTS\n\n{shared.read_text()}")

    if not parts:
        raise FileNotFoundError(f"No persona files found under {root} for persona '{config.name}'")

    identity_text = "\n\n".join(section.strip() for section in parts)

    return PersonaPrompt(text=identity_text)


def identity_anchor(config: PersonaConfig) -> str:
    """A short identity anchor for follow-up messages."""
    return f"You are {config.name}. Follow your AGENTS.md and MEMORY.md."
