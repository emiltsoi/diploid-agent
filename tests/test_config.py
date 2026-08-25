"""Tests for configuration loading."""

from pathlib import Path

import yaml

from acp_fleet_harness.config import Config, PersonaConfig


def test_config_loads_windsurf_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    secrets_path = tmp_path / "secrets.env"
    secrets_path.write_text("WINDSURF_API_KEY='test-key'\n")
    config_path.write_text(
        yaml.safe_dump(
            {
                "devin": {"bin": "/bin/devin"},
                "persona": {
                    "name": "test-pilot",
                    "profile_root": "/tmp/profile",
                },
            }
        )
    )
    config = Config.load(config_path, secrets_path)
    assert config.secrets
    assert config.secrets.windsurf_api_key == "test-key"


def test_skill_config_parses() -> None:
    from acp_fleet_harness.config import Config

    data = {
        "persona": {"name": "test-pilot", "profile_root": "/tmp"},
        "harness": {
            "skills": {
                "shared_root": "/tmp/shared",
                "default_enabled": ["review"],
            }
        },
    }
    config = Config(**data)
    assert config.harness.skills.default_enabled == ["review"]
    assert config.harness.skills.shared_root == Path("/tmp/shared")


def test_mcp_config_parses() -> None:
    from acp_fleet_harness.config import Config

    data = {
        "persona": {
            "name": "test-pilot",
            "profile_root": "/tmp",
        },
        "harness": {
            "mcp": {
                "servers": [
                    {
                        "name": "github",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": [],
                    }
                ],
                "default_enabled": ["github"],
            }
        },
    }
    config = Config(**data)
    assert len(config.harness.mcp.servers) == 1
    assert config.harness.mcp.servers[0].name == "github"
    assert config.harness.mcp.default_enabled == ["github"]


def test_config_loads_acp_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    secrets_path = tmp_path / "secrets.env"
    secrets_path.write_text("TELEGRAM_BOT_TOKEN=tok\n")
    config_path.write_text(
        yaml.safe_dump(
            {
                "devin": {"bin": "/bin/devin", "model": "swe-1-7"},
                "persona": {
                    "name": "test-pilot",
                    "profile_root": "/tmp/profile",
                    "fleet_root": "/tmp/fleet",
                },
                "harness": {"sessions_root": "/tmp/sessions"},
            }
        )
    )
    config = Config.load(config_path, secrets_path)
    assert config.devin.model == "swe-1-7"
    assert config.devin.bin == "/bin/devin"
    assert config.persona.name == "test-pilot"
    assert config.harness.telegram.token == "tok"


def test_persona_config_expands_paths() -> None:
    p = PersonaConfig(
        name="test",
        profile_root=Path("~/tmp/profile"),
        fleet_root=Path("~/tmp/fleet"),
    )
    assert p.profile_root == Path.home() / "tmp" / "profile"
    assert p.fleet_root == Path.home() / "tmp" / "fleet"


def test_persona_config_fleet_root_is_optional(tmp_path: Path) -> None:
    p = PersonaConfig(
        name="test",
        profile_root=tmp_path / "profile",
    )
    assert p.fleet_root is None
