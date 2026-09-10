"""Tests for the authorship toggle plugin."""

from pathlib import Path

from diploid_agent.config import AuthorshipConfig, PluginConfig
from diploid_agent.plugins.authorship import AuthorshipPlugin


def _plugin(config: dict | None = None) -> AuthorshipPlugin:
    cfg = PluginConfig(
        name="authorship",
        module="diploid_agent.plugins.authorship",
        enabled=True,
        prompt_slot="authorship",
        config=config or {},
    )
    return AuthorshipPlugin(cfg, "chat1", Path("sessions"), runtime=None)


def test_defaults_show_all_off() -> None:
    block = _plugin().prompt_block()
    assert block is not None
    assert "## Authorship toggles" in block
    assert "self-wake: off" in block
    assert "self-inference: off" in block
    assert "felt-authorship: off" in block
    assert "user-override guard: none" in block


def test_prompt_block_shows_toggles_and_guard() -> None:
    block = _plugin(
        {
            "self_wake_enabled": True,
            "self_inference_enabled": False,
            "felt_authorship_enabled": False,
            "user_override": ["emil"],
        }
    ).prompt_block()
    assert block is not None
    assert "self-wake: on" in block
    assert "self-inference: off" in block
    assert "felt-authorship: off" in block
    assert "user-override guard: emil" in block


def test_config_validated() -> None:
    plugin = _plugin({"user_override": ["emil"]})
    assert plugin._authorship() == AuthorshipConfig(user_override=["emil"])
