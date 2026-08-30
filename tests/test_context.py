"""Tests for the ContextBuilder prompt-assembly module."""

import time
from pathlib import Path

from diploid_agent.config import Config, DiploidConfig, HarnessConfig, PersonaConfig
from diploid_agent.context import ContextBuilder
from diploid_agent.dispatch import DispatchStore
from diploid_agent.engine.fake import FakeAgentEngine
from diploid_agent.memory import MemoryItem, MemoryManager
from diploid_agent.models import SessionRecord
from diploid_agent.plugins import PluginManager
from diploid_agent.skills import SkillManager


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


def _make_builder(tmp_path: Path) -> ContextBuilder:
    config = _make_config(tmp_path)
    sessions_root = tmp_path / "sessions"
    plugin_manager = PluginManager(
        plugins=list(config.harness.plugins),
        sessions_root=sessions_root,
        instance_id="test-instance",
        instance_started_at=time.time(),
        dispatch_store=DispatchStore(tmp_path / "dispatch.jsonl"),
    )
    engine = FakeAgentEngine()

    def memory_factory(chat_id: str) -> MemoryManager:
        return MemoryManager(
            config=config.harness.memory,
            persona=config.persona,
            sessions_root=sessions_root,
            chat_id=chat_id,
            devin_client=engine,
        )

    return ContextBuilder(config, plugin_manager, memory_factory)


def test_build_first_prompt_for_new_chat(tmp_path: Path) -> None:
    """A first-turn prompt contains identity, user message, and empty memory flags."""
    builder = _make_builder(tmp_path)

    pctx = builder.build_first("chat-1", "Hello, can you help me?", record=None)

    assert "Hello, can you help me?" in pctx.prompt
    assert pctx.notice is None
    assert pctx.memory_flags == {
        "persona_memory_exceeded": False,
        "chat_memory_exceeded": False,
    }
    assert pctx.slots.get("user") == ["Hello, can you help me?"]
    # The fixture has a SOUL.md and AGENTS.md identity file.
    assert "## SOUL" in pctx.prompt or "## AGENTS" in pctx.prompt


def test_build_follow_up_for_existing_chat(tmp_path: Path) -> None:
    """A follow-up prompt re-injects the full identity and the user message."""
    builder = _make_builder(tmp_path)

    pctx = builder.build_follow_up("chat-1", "What about the second item?", record=None)

    assert "What about the second item?" in pctx.prompt
    assert "I am **Test Pilot**" in pctx.prompt
    assert pctx.notice is None
    assert pctx.memory_flags == {
        "persona_memory_exceeded": False,
        "chat_memory_exceeded": False,
    }
    assert "identity" in pctx.slots
    assert "user" in pctx.slots


def test_build_first_prompt_trims_reply_quote_and_injects_continuation(tmp_path: Path) -> None:
    """A long reply-to quote is trimmed, and the continuation anchor is injected."""
    builder = _make_builder(tmp_path)

    long_quote = "word " * 600  # 3000 chars, above the 2048 default cap.
    anchor = "Please continue from the previous partial result."

    pctx = builder.build_first(
        "chat-1",
        "Please expand.",
        record=None,
        reply_to=long_quote,
        reply_to_is_bot=False,
        continuation_anchor=anchor,
    )

    assert "[In reply to your earlier message:]" in pctx.prompt
    assert "[Your new message:]" in pctx.prompt
    assert "Please expand." in pctx.prompt
    assert "[..." in pctx.prompt and "truncated" in pctx.prompt
    assert pctx.prompt.count("word ") < 500
    assert anchor in pctx.prompt
    assert "continuation" in pctx.slots and anchor in pctx.slots["continuation"]


def test_continuation_anchor_for_stopped_turn(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path)
    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="s1",
        model="m1",
        persona="test",
        cwd="/tmp",
        created_at=0.0,
        updated_at=0.0,
        last_stop_reason="stopped",
    )

    anchor = builder.continuation_anchor(record, "Continue")
    assert anchor is not None
    assert "stopped by the user" in anchor
    assert "Continue" in anchor


def test_continuation_anchor_for_cancelled_turn(tmp_path: Path) -> None:
    builder = _make_builder(tmp_path)
    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="s1",
        model="m1",
        persona="test",
        cwd="/tmp",
        created_at=0.0,
        updated_at=0.0,
        last_stop_reason="cancelled",
    )

    anchor = builder.continuation_anchor(record, "Continue")
    assert anchor is not None
    assert "interrupted" in anchor


def test_build_first_rehydration_notice_is_in_prompt_not_user_notice(tmp_path: Path) -> None:
    """A rehydrated first turn injects a system notice into the prompt only."""
    builder = _make_builder(tmp_path)
    pctx = builder.build_first(
        "chat-1",
        "hello",
        record=None,
        rehydrated=True,
    )
    assert "rehydrated" in pctx.prompt.lower()
    assert "rehydrated" not in (pctx.notice or "").lower()


def test_skill_context_is_compact_index_no_full_content(tmp_path: Path) -> None:
    """Only the skill index is injected; full SKILL.md content is not."""
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True)
    skill_dir = skills_root / "model-review"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: model-review\n---\n\nRun model review.\n", encoding="utf-8"
    )

    manager = SkillManager(
        personas_root=tmp_path / "personas",
        shared_root=tmp_path,
        chat_cwd_root=tmp_path,
    )

    builder = _make_builder(tmp_path)
    builder.skill_manager = manager
    builder.active_skill_names = lambda chat_id: {"model-review"}

    pctx = builder.build_first("chat-1", "Run a model review.", record=None)

    assert "## Available skills" in pctx.prompt
    assert "model-review" in pctx.prompt
    assert "(active)" in pctx.prompt
    assert "Run model review." not in pctx.prompt


def test_build_first_includes_chat_memory_block(tmp_path: Path) -> None:
    """A first-turn prompt loads the on-disk chat memory block."""
    builder = _make_builder(tmp_path)
    mgr = builder.memory_factory("chat-1")
    fb = mgr._file_backend
    assert fb is not None
    fb.retain([MemoryItem(content="We agreed on Postgres.", tags=["memory"])])
    pctx = builder.build_first("chat-1", "hello", record=None)
    assert "## Chat memory (on disk)" in pctx.prompt
    assert "Postgres" in pctx.prompt


def test_build_follow_up_includes_chat_memory_anchor(tmp_path: Path) -> None:
    """A follow-up prompt loads the full chat memory."""
    builder = _make_builder(tmp_path)
    mgr = builder.memory_factory("chat-1")
    fb = mgr._file_backend
    assert fb is not None
    fb.retain([MemoryItem(content="We agreed on Postgres.", tags=["memory"])])
    pctx = builder.build_follow_up("chat-1", "what next?", record=None)
    assert "## Chat memory (on disk)" in pctx.prompt
    assert "Postgres" in pctx.prompt
