"""Tests for WakeContext forwarding through ContextBuilder."""

from pathlib import Path

from diploid_agent.config import (
    Config,
    DiploidConfig,
    HarnessConfig,
    PersonaConfig,
    PluginConfig,
)
from diploid_agent.context import ContextBuilder
from diploid_agent.engine.fake import FakeAgentEngine
from diploid_agent.memory import MemoryManager
from diploid_agent.models import WakeEvent
from diploid_agent.plugins.base import StatePlugin, WakeContext
from diploid_agent.plugins.manager import PluginManager


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_config(tmp_path: Path) -> Config:
    return Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=_fixture_root(),
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
    )


class WakeSpyPlugin(StatePlugin):
    def __init__(self):
        self.context: WakeContext | None = None
        super().__init__(
            PluginConfig(name="spy", enabled=True),
            "chat-1",
            Path("."),
        )

    def on_waking(self, context: WakeContext) -> None:
        self.context = context


def test_wake_context_receives_event_and_other_instance_flag(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    spy = WakeSpyPlugin()
    mgr = PluginManager(
        plugins=[PluginConfig(name="spy", enabled=True, module=None)],
        sessions_root=tmp_path,
        instance_id="i-1",
        instance_started_at=0.0,
    )
    mgr._instances["chat-1"] = {"spy": spy}

    engine = FakeAgentEngine()

    def memory_factory(chat_id: str) -> MemoryManager:
        return MemoryManager(
            config=cfg.harness.memory,
            persona=cfg.persona,
            sessions_root=tmp_path / "sessions",
            chat_id=chat_id,
            devin_client=engine,
        )

    builder = ContextBuilder(cfg, mgr, memory_factory)
    event = WakeEvent(
        id="w1",
        chat_id="chat-1",
        reason="timer",
        priority=1,
        scheduled_at=0.0,
        created_at=0.0,
    )
    builder.build_first(
        "chat-1",
        "hi",
        None,
        wake_event=event,
        other_instance_running=True,
    )
    assert spy.context is not None
    assert spy.context.wake_event is event
    assert spy.context.other_instance_running is True


def test_wake_context_receives_event_and_other_instance_flag_in_follow_up(
    tmp_path: Path,
) -> None:
    cfg = _make_config(tmp_path)
    spy = WakeSpyPlugin()
    mgr = PluginManager(
        plugins=[PluginConfig(name="spy", enabled=True, module=None)],
        sessions_root=tmp_path,
        instance_id="i-1",
        instance_started_at=0.0,
    )
    mgr._instances["chat-1"] = {"spy": spy}

    engine = FakeAgentEngine()

    def memory_factory(chat_id: str) -> MemoryManager:
        return MemoryManager(
            config=cfg.harness.memory,
            persona=cfg.persona,
            sessions_root=tmp_path / "sessions",
            chat_id=chat_id,
            devin_client=engine,
        )

    builder = ContextBuilder(cfg, mgr, memory_factory)
    event = WakeEvent(
        id="w2",
        chat_id="chat-1",
        reason="mesh",
        priority=1,
        scheduled_at=0.0,
        created_at=0.0,
    )
    builder.build_follow_up(
        "chat-1",
        "hi",
        None,
        wake_event=event,
        other_instance_running=True,
    )
    assert spy.context is not None
    assert spy.context.wake_event is event
    assert spy.context.other_instance_running is True
