"""Tests for AgentEngine implementations."""

from pathlib import Path

from devin_fleet_harness.engine import TurnRequest
from devin_fleet_harness.engine.fake import FakeAgentEngine


def test_fake_engine_prompt_creates_session() -> None:
    engine = FakeAgentEngine()
    req = TurnRequest(prompt="hello", cwd=Path("."))
    result = engine.prompt(req)
    assert result.reply == "ok"
    assert result.session_id is not None


def test_fake_engine_prompt_continues_session() -> None:
    engine = FakeAgentEngine()
    result = engine.prompt(TurnRequest(prompt="hello", cwd=Path(".")), session_id="sess-1")
    assert result.session_id == "sess-1"


def test_fake_engine_records_calls() -> None:
    engine = FakeAgentEngine(replies=["first", "second"])
    engine.prompt(TurnRequest(prompt="a", cwd=Path(".")))
    engine.cancel("sess-1")
    assert len(engine.call_log) == 2
    assert engine.call_log[0][0] == "prompt"
    assert engine.call_log[1][0] == "cancel"


def test_fake_engine_list_models() -> None:
    engine = FakeAgentEngine(models=["m1", "m2"])
    assert engine.list_models() == ["m1", "m2"]
