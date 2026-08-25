"""Tests for the engine factory."""

from acp_fleet_harness.config import Config, EngineConfig, HarnessConfig, PersonaConfig
from acp_fleet_harness.engine import AcpEngine, DevinAcpEngine, build_engine
from acp_fleet_harness.engine.fake import FakeAgentEngine


def test_build_devin_engine() -> None:
    config = EngineConfig(provider="devin", bin="/bin/echo")
    engine = build_engine(config, api_key="test-key")
    assert isinstance(engine, AcpEngine)
    assert isinstance(engine, DevinAcpEngine)


def test_build_generic_engine() -> None:
    config = EngineConfig(provider="generic", bin="/bin/echo")
    engine = build_engine(config, api_key="test-key")
    assert isinstance(engine, AcpEngine)


def test_build_fake_engine() -> None:
    config = EngineConfig(provider="fake")
    engine = build_engine(config)
    assert isinstance(engine, FakeAgentEngine)


def test_config_devin_alias_still_works() -> None:
    config = Config(
        devin=EngineConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test",
            profile_root=HarnessConfig().sessions_root,
        ),
    )
    assert config.engine.bin == "/bin/echo"
    assert config.engine.model == "swe-1-7"
