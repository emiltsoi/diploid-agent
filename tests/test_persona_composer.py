"""Tests for persona prompt composition."""

from pathlib import Path

from diploid_agent.config import PersonaConfig
from diploid_agent.persona_composer import compose_persona, identity_anchor


def test_compose_persona_from_fixture() -> None:
    root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = PersonaConfig(
        name="test-pilot",
        profile_root=root,
        fleet_root=root.parent,
    )
    prompt = compose_persona(config)
    assert "I am **Test Pilot**" in prompt.text
    assert "Test Pilot" in prompt.text
    assert prompt.memory_text == ""
    assert prompt.memory_path is None


def test_identity_anchor() -> None:
    config = PersonaConfig(
        name="test-pilot",
        profile_root=Path("/tmp"),
        fleet_root=Path("/tmp"),
    )
    anchor = identity_anchor(config)
    assert "test-pilot" in anchor
    assert "Your full identity is in:" in anchor
    assert "Follow them." in anchor


def test_identity_anchor_lists_existing_files(tmp_path: Path) -> None:
    root = tmp_path / "persona"
    root.mkdir()
    (root / "SOUL.md").write_text("# SOUL")
    (root / "AGENTS.md").write_text("# AGENTS")
    (root / "MEMORY.md").write_text("# MEMORY")
    fleet = tmp_path / "fleet"
    (fleet / "shared").mkdir(parents=True)
    (fleet / "shared" / "AGENTS.md").write_text("# Shared")

    config = PersonaConfig(
        name="test-pilot",
        profile_root=root,
        fleet_root=fleet,
    )
    anchor = identity_anchor(config)

    assert str(root / "SOUL.md") in anchor
    assert str(root / "AGENTS.md") in anchor
    assert str(root / "MEMORY.md") in anchor
    assert str(fleet / "shared" / "AGENTS.md") in anchor
