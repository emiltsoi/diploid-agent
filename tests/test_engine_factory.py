"""Tests for the engine factory."""

from diploid_agent.config import Config, EngineConfig, HarnessConfig, PersonaConfig
from diploid_agent.engine import AcpEngine, build_engine
from diploid_agent.engine.fake import FakeAgentEngine


def test_build_diploid_engine() -> None:
    config = EngineConfig(provider="diploid", bin="/bin/echo")
    engine = build_engine(config, api_key="test-key")
    assert isinstance(engine, AcpEngine)


def test_build_generic_engine() -> None:
    config = EngineConfig(provider="generic", bin="/bin/echo")
    engine = build_engine(config, api_key="test-key")
    assert isinstance(engine, AcpEngine)


def test_build_fake_engine() -> None:
    config = EngineConfig(provider="fake")
    engine = build_engine(config)
    assert isinstance(engine, FakeAgentEngine)


def test_config_diploid_alias_still_works() -> None:
    config = Config(
        diploid=EngineConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test",
            profile_root=HarnessConfig().sessions_root,
        ),
    )
    assert config.engine.bin == "/bin/echo"
    assert config.engine.model == "swe-1-7"
