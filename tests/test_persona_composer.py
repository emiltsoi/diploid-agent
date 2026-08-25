"""Tests for persona prompt composition."""

from pathlib import Path

from acp_fleet_harness.config import PersonaConfig
from acp_fleet_harness.persona_composer import compose_persona, identity_anchor


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
    assert "test-pilot" in identity_anchor(config)
