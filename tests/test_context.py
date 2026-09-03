"""Tests for the ContextBuilder prompt-assembly module."""

import json
import os
import time
from pathlib import Path

from diploid_agent.config import Config, DiploidConfig, HarnessConfig, PersonaConfig, PluginConfig
from diploid_agent.context import ContextBuilder
from diploid_agent.dispatch import Dispatch, DispatchStatus, DispatchStore
from diploid_agent.engine.fake import FakeAgentEngine
from diploid_agent.memory import MemoryItem, MemoryManager, RecallResult
from diploid_agent.models import SessionRecord
from diploid_agent.plugins import PluginManager
from diploid_agent.skills import SkillManager


def _fixture_root() -> Path:
    return Path(__file__).parent / "fixtures" / "test-pilot"


def _make_builder_with_plugins(
    tmp_path: Path, plugin_configs: list[PluginConfig]
) -> ContextBuilder:
    """Return a ContextBuilder with additional plugin configs."""
    config = _make_config(tmp_path)
    if plugin_configs:
        config.harness.plugins.extend(plugin_configs)
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


def _make_builder_with_profile_root(tmp_path: Path, profile_root: Path) -> ContextBuilder:
    """Return a ContextBuilder that uses a custom persona profile root."""
    config = Config(
        diploid=DiploidConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=profile_root,
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
    )
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
    # The `persona_state` slot has been split into three dedicated slots.
    assert "body" in pctx.slots
    assert "self_state" in pctx.slots
    assert "mesh" in pctx.slots
    assert "persona_state" not in pctx.slots
    # The fixture has a SOUL.md and AGENTS.md identity file.
    assert "## SOUL" in pctx.prompt or "## AGENTS" in pctx.prompt


def test_build_follow_up_for_existing_chat(tmp_path: Path) -> None:
    """A follow-up prompt uses a short, linked identity anchor and the user message."""
    builder = _make_builder(tmp_path)

    pctx = builder.build_follow_up("chat-1", "What about the second item?", record=None)

    assert "What about the second item?" in pctx.prompt
    # The follow-up uses the linked identity anchor, not the full SOUL/AGENTS text.
    assert "You are test-pilot" in pctx.prompt
    assert str(_fixture_root() / "SOUL.md") in pctx.prompt
    assert str(_fixture_root() / "AGENTS.md") in pctx.prompt
    assert "I am **Test Pilot**" not in pctx.prompt
    assert pctx.notice is None
    assert pctx.memory_flags == {
        "persona_memory_exceeded": False,
        "chat_memory_exceeded": False,
    }
    assert "identity" in pctx.slots
    assert "user" in pctx.slots
    # The `persona_state` slot has been split into three dedicated slots.
    assert "body" in pctx.slots
    assert "self_state" in pctx.slots
    assert "mesh" in pctx.slots
    assert "persona_state" not in pctx.slots


def _fake_recall_result() -> RecallResult:
    return RecallResult(
        text="RECALL",
        truncated=False,
        memory_path=None,
        limit=100,
        loaded=6,
        total=6,
    )


def test_build_follow_up_skips_recall_when_disabled(tmp_path: Path, monkeypatch) -> None:
    """Hindsight recall is skipped on follow-ups when recall_on_follow_up is False."""
    builder = _make_builder(tmp_path)
    calls: list[str] = []

    def fake_recall(_self, user_message: str, model: str | None = None) -> RecallResult:
        calls.append(user_message)
        return _fake_recall_result()

    monkeypatch.setattr(MemoryManager, "recall_context", fake_recall)
    pctx = builder.build_follow_up("chat-1", "What about the second item?", record=None)

    assert not calls
    assert "RECALL" not in pctx.prompt
    assert pctx.slots.get("recall") == []


def test_build_follow_up_includes_recall_when_enabled(tmp_path: Path, monkeypatch) -> None:
    """Hindsight recall is run on follow-ups when recall_on_follow_up is True."""
    builder = _make_builder(tmp_path)
    builder.config.harness.memory.recall_on_follow_up = True
    calls: list[str] = []

    def fake_recall(_self, user_message: str, model: str | None = None) -> RecallResult:
        calls.append(user_message)
        return _fake_recall_result()

    monkeypatch.setattr(MemoryManager, "recall_context", fake_recall)
    pctx = builder.build_follow_up("chat-1", "What about the second item?", record=None)

    assert calls
    assert "RECALL" in pctx.prompt
    assert pctx.slots.get("recall") == ["## Chat memory\n\nRECALL"]


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
    """A follow-up prompt loads the on-disk chat memory."""
    builder = _make_builder(tmp_path)
    mgr = builder.memory_factory("chat-1")
    fb = mgr._file_backend
    assert fb is not None
    fb.retain([MemoryItem(content="We agreed on Postgres.", tags=["memory"])])
    pctx = builder.build_follow_up("chat-1", "what next?", record=None)
    assert "## Chat memory (on disk)" in pctx.prompt
    assert "Postgres" in pctx.prompt


def test_build_follow_up_skips_unchanged_chat_memory(tmp_path: Path) -> None:
    """A follow-up prompt skips the on-disk chat memory block if it has not changed."""
    builder = _make_builder(tmp_path)
    mgr = builder.memory_factory("chat-1")
    fb = mgr._file_backend
    assert fb is not None
    fb.retain([MemoryItem(content="We agreed on Postgres.", tags=["memory"])])

    pctx_first = builder.build_first("chat-1", "hello", record=None)
    assert "Postgres" in pctx_first.prompt

    pctx_follow = builder.build_follow_up("chat-1", "what next?", record=None)
    assert "## Chat memory (on disk)" not in pctx_follow.prompt
    assert "Postgres" not in pctx_follow.prompt


def test_build_follow_up_includes_changed_chat_memory(tmp_path: Path) -> None:
    """A follow-up prompt re-injects on-disk chat memory when it changes."""
    builder = _make_builder(tmp_path)
    mgr = builder.memory_factory("chat-1")
    fb = mgr._file_backend
    assert fb is not None
    fb.retain([MemoryItem(content="We agreed on Postgres.", tags=["memory"])])

    pctx_first = builder.build_first("chat-1", "hello", record=None)
    assert "Postgres" in pctx_first.prompt

    fb.retain([MemoryItem(content="We agreed on SQLite.", tags=["memory"])])
    chat_path = mgr.chat_memory_path
    assert chat_path is not None
    last = builder._last_file_mtimes["chat-1"][str(chat_path)]
    os.utime(chat_path, (last + 1, last + 1))

    pctx_follow = builder.build_follow_up("chat-1", "what next?", record=None)
    assert "SQLite" in pctx_follow.prompt
    assert "Postgres" in pctx_follow.prompt


def test_build_follow_up_skips_unchanged_json_state_plugin(tmp_path: Path) -> None:
    """A follow-up prompt skips an unchanged JsonStatePlugin block."""
    cfg = PluginConfig(
        name="mood",
        enabled=True,
        state_file="mood.json",
        prompt_slot="body",
        prompt_template="Mood: {mood}",
    )
    builder = _make_builder_with_plugins(tmp_path, [cfg])
    chat_dir = tmp_path / "sessions" / "chat-1"
    chat_dir.mkdir(parents=True, exist_ok=True)
    (chat_dir / "mood.json").write_text(json.dumps({"mood": "calm"}))

    pctx_first = builder.build_first("chat-1", "hello", record=None)
    assert "Mood: calm" in pctx_first.prompt

    pctx_follow = builder.build_follow_up("chat-1", "how are you?", record=None)
    assert "Mood: calm" not in pctx_follow.prompt


def test_build_follow_up_includes_changed_json_state_plugin(tmp_path: Path) -> None:
    """A follow-up prompt re-injects a JsonStatePlugin block when its state file changes."""
    cfg = PluginConfig(
        name="mood",
        enabled=True,
        state_file="mood.json",
        prompt_slot="body",
        prompt_template="Mood: {mood}",
    )
    builder = _make_builder_with_plugins(tmp_path, [cfg])
    chat_dir = tmp_path / "sessions" / "chat-1"
    chat_dir.mkdir(parents=True, exist_ok=True)
    (chat_dir / "mood.json").write_text(json.dumps({"mood": "calm"}))

    pctx_first = builder.build_first("chat-1", "hello", record=None)
    assert "Mood: calm" in pctx_first.prompt

    (chat_dir / "mood.json").write_text(json.dumps({"mood": "happy"}))
    mood_path = chat_dir / "mood.json"
    last = mood_path.stat().st_mtime
    os.utime(mood_path, (last + 1, last + 1))

    pctx_follow = builder.build_follow_up("chat-1", "how are you?", record=None)
    assert "Mood: happy" in pctx_follow.prompt


def test_build_follow_up_skips_unchanged_persona_memory(tmp_path: Path) -> None:
    """A follow-up prompt skips the persona MEMORY.md if it has not changed."""
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    (profile_root / "SOUL.md").write_text("# SOUL")
    (profile_root / "AGENTS.md").write_text("# AGENTS")
    (profile_root / "MEMORY.md").write_text("We value kindness.")

    builder = _make_builder_with_profile_root(tmp_path, profile_root)
    pctx_first = builder.build_first("chat-1", "hello", record=None)
    assert "We value kindness." in pctx_first.prompt

    pctx_follow = builder.build_follow_up("chat-1", "how are you?", record=None)
    assert "We value kindness." not in pctx_follow.prompt


def test_build_dispatch_continuation_completed(tmp_path: Path) -> None:
    """A completed dispatch produces a structured result block."""
    builder = _make_builder(tmp_path)
    dispatch = Dispatch(
        id="dispatch-abc123",
        chat_id="chat-1",
        session_id="session-1",
        status=DispatchStatus.COMPLETED,
        result="full result",
        summary="Short summary",
        started_at=1000.0,
        finished_at=1030.0,
        full_result_path="/tmp/sessions/chat-1/subagent-results/subagent-dispatch-abc123.md",
    )
    text = builder.build_dispatch_continuation(dispatch)
    assert "## Subagent result" in text
    assert "- **status:** completed" in text
    assert "- **duration:** 30s" in text
    assert "- **summary:** Short summary" in text
    assert (
        "- **full_result_path:** /tmp/sessions/chat-1/subagent-results/subagent-dispatch-abc123.md"
        in text
    )
    assert "Please continue and present the result to the user." in text
    assert "stopped because" not in text


def test_build_dispatch_continuation_timeout(tmp_path: Path) -> None:
    """A timed-out dispatch reports timeout and partial summary."""
    builder = _make_builder(tmp_path)
    dispatch = Dispatch(
        id="dispatch-abc123",
        chat_id="chat-1",
        session_id="session-1",
        status=DispatchStatus.TIMEOUT,
        result="partial result",
        summary="Partial summary",
        started_at=1000.0,
        finished_at=1100.0,
        full_result_path="/tmp/sessions/chat-1/subagent-results/subagent-dispatch-abc123.md",
        stop_reason="timeout",
    )
    text = builder.build_dispatch_continuation(dispatch)
    assert "- **status:** timeout" in text
    assert "- **duration:** 1m 40s" in text
    assert "The subagent stopped because it ran out of time." in text
    assert "The summary below is partial." in text


def test_build_dispatch_continuation_cancelled(tmp_path: Path) -> None:
    """A cancelled dispatch reports cancellation."""
    builder = _make_builder(tmp_path)
    dispatch = Dispatch(
        id="dispatch-abc123",
        chat_id="chat-1",
        session_id="session-1",
        status=DispatchStatus.CANCELLED,
        result="partial result",
        summary="Partial summary",
        started_at=1000.0,
        finished_at=1005.0,
        full_result_path="/tmp/sessions/chat-1/subagent-results/subagent-dispatch-abc123.md",
    )
    text = builder.build_dispatch_continuation(dispatch)
    assert "- **status:** cancelled" in text
    assert "The subagent stopped because it was cancelled." in text


def test_build_dispatch_continuation_failed(tmp_path: Path) -> None:
    """A failed dispatch reports failed status without a partial warning."""
    builder = _make_builder(tmp_path)
    dispatch = Dispatch(
        id="dispatch-abc123",
        chat_id="chat-1",
        session_id="session-1",
        status=DispatchStatus.PENDING,
        result="error output",
        summary="Error summary",
        started_at=1000.0,
        finished_at=1020.0,
        stop_reason="failed",
    )
    text = builder.build_dispatch_continuation(dispatch)
    assert "- **status:** failed" in text
    assert "stopped because" not in text


def test_build_dispatch_continuation_running_fallback_path(tmp_path: Path) -> None:
    """A still-running dispatch reports running and falls back to a computed path."""
    builder = _make_builder(tmp_path)
    dispatch = Dispatch(
        id="dispatch-abc123",
        chat_id="chat-1",
        session_id="session-1",
        status=DispatchStatus.PENDING,
        started_at=time.time(),
    )
    text = builder.build_dispatch_continuation(dispatch)
    assert "- **status:** running" in text
    assert "subagent-results/subagent-dispatch-abc123.md" in text


def test_build_follow_up_reinjects_full_soul_on_rehydrated(tmp_path: Path) -> None:
    """A rehydrated follow-up loads full persona memory and chat memory."""
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    (profile_root / "SOUL.md").write_text("# SOUL")
    (profile_root / "AGENTS.md").write_text("# AGENTS")
    (profile_root / "MEMORY.md").write_text("We value kindness.")

    builder = _make_builder_with_profile_root(tmp_path, profile_root)
    mgr = builder.memory_factory("chat-1")
    fb = mgr._file_backend
    assert fb is not None
    fb.retain([MemoryItem(content="We agreed on Postgres.", tags=["memory"])])

    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="session-1",
        model="swe-1-7",
        persona="test-pilot",
        cwd=str(tmp_path),
        created_at=time.time(),
        updated_at=time.time(),
        turn_number=5,
    )

    pctx = builder.build_follow_up("chat-1", "how are you?", record=record, rehydrated=True)
    assert "We value kindness." in pctx.prompt
    assert "## Chat memory (on disk)" in pctx.prompt


def test_build_follow_up_small_soul_under_context_pressure(tmp_path: Path) -> None:
    """Under context pressure, cheap soul slots are forced into the prompt."""
    cfg = PluginConfig(
        name="mood",
        enabled=True,
        state_file="mood.json",
        prompt_slot="body",
        prompt_template="Mood: {mood}",
    )
    builder = _make_builder_with_plugins(tmp_path, [cfg])
    chat_dir = tmp_path / "sessions" / "chat-1"
    chat_dir.mkdir(parents=True, exist_ok=True)
    (chat_dir / "mood.json").write_text(json.dumps({"mood": "calm"}))

    pctx_first = builder.build_first("chat-1", "hello", record=None)
    assert "Mood: calm" in pctx_first.prompt

    # High input ratio, but cumulative low enough to stay below full threshold.
    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="session-1",
        model="swe-1-7",
        persona="test-pilot",
        cwd=str(tmp_path),
        created_at=time.time(),
        updated_at=time.time(),
        turn_number=5,
        cumulative_metrics={"total_tokens": 100},
        last_turn_metrics={"input_tokens": 700},
    )
    builder.config.engine.context_window = 1000
    # Disable proactive fresh so this test stays in the small-soul regime.
    builder.config.harness.proactive_new_session_threshold = 5.0

    pctx_follow = builder.build_follow_up("chat-1", "how are you?", record=record)
    assert "Mood: calm" in pctx_follow.prompt
    assert "Context pressure is high" in pctx_follow.prompt
    assert not pctx_follow.force_new_session


def test_build_follow_up_proactive_fresh_session(
    tmp_path: Path,
) -> None:
    """Proactive prompt sizing triggers a compact fresh session before overflow."""
    builder = _make_builder(tmp_path)
    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="session-1",
        model="swe-1-7",
        persona="test-pilot",
        cwd=str(tmp_path),
        created_at=time.time(),
        updated_at=time.time(),
        turn_number=5,
        cumulative_metrics={"total_tokens": 500},
        last_turn_metrics={"input_tokens": 100},
    )
    builder.config.engine.context_window = 1000

    pctx = builder.build_follow_up("chat-1", "how are you?", record=record)
    assert "Fresh ACP session for context pressure" in pctx.prompt
    assert pctx.force_new_session


def test_build_follow_up_fresh_soul_at_high_pressure(
    tmp_path: Path,
) -> None:
    """At very high context pressure, a compact fresh soul is loaded and a new ACP session is requested."""
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    (profile_root / "SOUL.md").write_text("# SOUL")
    (profile_root / "AGENTS.md").write_text("# AGENTS")
    (profile_root / "MEMORY.md").write_text("We value kindness.")

    builder = _make_builder_with_profile_root(tmp_path, profile_root)
    mgr = builder.memory_factory("chat-1")
    fb = mgr._file_backend
    assert fb is not None
    fb.retain([MemoryItem(content="We agreed on Postgres.", tags=["memory"])])

    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="session-1",
        model="swe-1-7",
        persona="test-pilot",
        cwd=str(tmp_path),
        created_at=time.time(),
        updated_at=time.time(),
        turn_number=5,
        cumulative_metrics={"total_tokens": 950},
        last_turn_metrics={"input_tokens": 100},
    )
    builder.config.engine.context_window = 1000

    pctx = builder.build_follow_up("chat-1", "how are you?", record=record)
    assert "We value kindness." in pctx.prompt
    assert "## Chat memory (on disk)" in pctx.prompt
    assert pctx.force_new_session


def test_chars_per_token_prefers_live_calibration(tmp_path: Path) -> None:
    """_chars_per_token calibrates from last-turn prompt_chars / input_tokens."""
    builder = _make_builder(tmp_path)
    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="session-1",
        model="swe-1-7",
        persona="test-pilot",
        cwd=str(tmp_path),
        created_at=time.time(),
        updated_at=time.time(),
        turn_number=1,
        last_turn_metrics={"input_tokens": 1000, "prompt_chars": 3500},
    )
    assert builder._chars_per_token("swe-1-7", record) == 3.5

    # Below the minimum prompt length, the hand table is used instead.
    record.last_turn_metrics = {"input_tokens": 1000, "prompt_chars": 50}
    assert builder._chars_per_token("swe-1-7", record) == 3.5

    # Unknown model with calibration data uses the calibrated ratio.
    record.last_turn_metrics = {"input_tokens": 1000, "prompt_chars": 4000}
    assert builder._chars_per_token("custom-model-v1", record) == 4.0

    # Calibration disabled falls back to the table/fallback.
    builder.config.harness.proactive_calibration_enabled = False
    record.last_turn_metrics = {"input_tokens": 1000, "prompt_chars": 4000}
    assert builder._chars_per_token("swe-1-7", record) == 3.5


def test_proactive_sizing_uses_calibrated_chars_per_token(tmp_path: Path) -> None:
    """_estimate_next_prompt_tokens uses the live-calibrated ratio."""
    builder = _make_builder(tmp_path)
    record = SessionRecord(
        chat_id="chat-1",
        session_number=1,
        session_id="session-1",
        model="swe-1-7",
        persona="test-pilot",
        cwd=str(tmp_path),
        created_at=time.time(),
        updated_at=time.time(),
        turn_number=1,
        last_turn_metrics={"input_tokens": 1000, "prompt_chars": 4000, "output_tokens": 0},
    )
    estimate = builder._estimate_next_prompt_tokens("chat-1", record, "hello")
    assert estimate["chars_per_token"] == 4.0
    assert estimate["short_term"] == int(
        builder.config.harness.memory.max_short_term_chars / 4.0
    )
    assert estimate["last_total"] == 1000


def test_prompt_chars_recorded_in_turn_metrics(tmp_path: Path) -> None:
    """Turn metrics include the length of the prompt sent to the engine."""
    import threading
    from types import SimpleNamespace

    from diploid_agent.config import Config, DiploidConfig, PersonaConfig
    from diploid_agent.metrics import MetricsCollector
    from diploid_agent.runtime.metrics import RuntimeMetrics

    runtime = SimpleNamespace(
        config=Config(
            diploid=DiploidConfig(bin="/bin/echo"),
            persona=PersonaConfig(name="test", profile_root=tmp_path),
        ),
        metrics=MetricsCollector(),
        _lock=threading.RLock(),
    )
    metrics = RuntimeMetrics(runtime)
    result = metrics._record_turn_metrics(
        "chat-1", 1, "swe-1-7", {"input_tokens": 100}, 0.5, prompt_chars=350
    )
    assert result["prompt_chars"] == 350
