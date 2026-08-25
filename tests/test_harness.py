"""Tests for the conversational harness."""

import threading
import time
from pathlib import Path
from typing import Any

from acp_fleet_harness.acp_client import AcpPromptResult
from acp_fleet_harness.config import (
    Config,
    DevinConfig,
    HarnessConfig,
    McpConfig,
    McpServerConfig,
    PersonaConfig,
    Secrets,
)
from acp_fleet_harness.harness import ConversationHarness


def _make_config(tmp_path: Path, fixture_root: Path) -> Config:
    return Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=fixture_root,
            fleet_root=tmp_path / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )


def test_new_session_creates_fresh_session_and_keeps_memory(monkeypatch, tmp_path: Path) -> None:
    """`/new` drops the old record and starts a new ACP session."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    call_log: list[tuple[str, ...]] = []

    def fake_create_session(
        prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs
    ):
        call_log.append(("create", prompt[:20], str(cwd), model))
        return AcpPromptResult(reply=f"Ready. — {model or 'default'}", session_id="session-initial")

    def fake_send_message(
        session_id: str, prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs
    ):
        call_log.append(("send", session_id, prompt[:20], model))
        return AcpPromptResult(reply="Follow-up. — DP")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    # First turn establishes a session.
    result1 = harness.process("chat-1", "hello")
    assert result1.session_id == "session-initial"
    assert result1.turn_number == 1
    assert harness.status("chat-1")["active"]

    # /new starts a different session.
    result2 = harness.new_session("chat-1")
    assert result2.session_id == "session-initial"  # fake always returns the same id
    assert result2.turn_number == 1
    assert "New session started" in result2.reply

    # create_session should have been called twice (first turn + /new)
    assert [c[0] for c in call_log].count("create") == 2


def test_list_sessions_and_new_session_archive(monkeypatch, tmp_path: Path) -> None:
    """`/new` archives the previous session and `/sessions` lists both."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    monkeypatch.setattr(
        harness.client,
        "create_session",
        lambda prompt, *, cwd=None, model=None, **kwargs: AcpPromptResult(
            reply="Ready.", session_id="session-1"
        ),
    )
    monkeypatch.setattr(
        harness.client,
        "send_message",
        lambda session_id, prompt, *, cwd=None, model=None, **kwargs: AcpPromptResult(
            reply="Follow-up."
        ),
    )

    harness.process("chat-3", "hello")
    harness.process("chat-3", "world")
    harness.new_session("chat-3")

    listing = harness.list_sessions("chat-3")
    assert listing["active"] == 2
    assert len(listing["sessions"]) == 2
    assert listing["sessions"][0]["number"] == 1
    assert listing["sessions"][0]["is_active"] is False
    assert listing["sessions"][1]["number"] == 2
    assert listing["sessions"][1]["is_active"] is True


def test_new_session_keeps_durable_chat_ledger(monkeypatch, tmp_path: Path) -> None:
    """The chat transcript and memory files survive /new and stay append-only."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    monkeypatch.setattr(
        harness.client,
        "create_session",
        lambda prompt, *, cwd=None, model=None, **kwargs: AcpPromptResult(
            reply="Ready.", session_id="session-1"
        ),
    )
    monkeypatch.setattr(
        harness.client,
        "send_message",
        lambda session_id, prompt, *, cwd=None, model=None, **kwargs: AcpPromptResult(
            reply="Follow-up."
        ),
    )

    harness.process("chat-ledger", "hello")
    harness.process("chat-ledger", "world")
    chat_dir = tmp_path / "sessions" / "chat-ledger"
    transcript = chat_dir / "chat_transcript.jsonl"
    assert transcript.exists()
    lines_before = transcript.read_text().splitlines()
    assert len(lines_before) == 4  # 2 user + 2 assistant

    harness.new_session("chat-ledger")
    harness.process("chat-ledger", "again")

    lines_after = transcript.read_text().splitlines()
    assert len(lines_after) == 8
    contents = "\n".join(lines_after)
    assert "hello" in contents
    assert "world" in contents
    assert "again" in contents


def test_resume_session_rehydrates(monkeypatch, tmp_path: Path) -> None:
    """`/resume` copies an archived session and starts a fresh ACP session."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Ready.", session_id=f"session-{model}")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(
        harness.client, "send_message", lambda *a, **kwargs: AcpPromptResult(reply="Follow-up.")
    )
    monkeypatch.setattr(harness.client, "session_alive", lambda *a, **k: False)

    harness.process("chat-4", "hello")
    harness.new_session("chat-4")

    result = harness.resume_session("chat-4", 1)
    assert result.session_id == "session-swe-1-7"
    assert harness.list_sessions("chat-4")["active"] == 1


def test_branch_session_creates_new_active(monkeypatch, tmp_path: Path) -> None:
    """`/branch` creates a new active session copied from an archived one."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Ready.", session_id=f"session-{model}")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(
        harness.client, "send_message", lambda *a, **kwargs: AcpPromptResult(reply="Follow-up.")
    )

    harness.process("chat-5", "hello")
    harness.new_session("chat-5")

    result = harness.branch_session("chat-5", 1)
    assert result.session_id == "session-swe-1-7"

    listing = harness.list_sessions("chat-5")
    active = listing["active"]
    assert active == 3
    assert listing["sessions"][-1]["parent"] == 1


def test_auto_recovery_on_stale_session(monkeypatch, tmp_path: Path) -> None:
    """A stale ACP session in process() is rehydrated automatically."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    create_count = [0]

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        create_count[0] += 1
        return AcpPromptResult(reply="Ready.", session_id=f"session-{create_count[0]}")

    def fake_send_message(session_id, prompt, *, cwd=None, model=None, **kwargs):
        raise RuntimeError("ACP session/prompt failed: Session not found")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    result1 = harness.process("chat-6", "hello")
    assert result1.session_number == 1
    assert result1.session_id == "session-1"

    result2 = harness.process("chat-6", "follow-up")
    assert result2.session_number == 1
    assert result2.session_id == "session-2"
    assert create_count[0] == 2


def test_model_switch_starts_new_session(monkeypatch, tmp_path: Path) -> None:
    """`/model <name>` forces a new session when the model changes."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    monkeypatch.setattr(
        harness.client,
        "create_session",
        lambda prompt, *, cwd=None, model=None, **kwargs: AcpPromptResult(
            reply=f"Ready. — {model}", session_id=f"session-{model}"
        ),
    )

    harness.process("chat-2", "hello")
    result = harness.switch_model("chat-2", "glm-5-2")
    assert "glm-5-2" in result.reply
    assert result.session_id == "session-glm-5-2"


def test_partial_turn_returns_partial_notice(monkeypatch, tmp_path: Path) -> None:
    """A cancelled/partial ACP result is surfaced to the user."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Partial.", session_id="s-1", partial=True, cancelled=True)

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    result = harness.process("chat-partial", "hello")
    assert "Partial" in result.reply
    assert result.notice and "stopped" in result.notice.lower()


def test_stop_calls_cancel_and_returns_message(monkeypatch, tmp_path: Path) -> None:
    """stop() forwards a cancel request to the ACP client."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)
    cancel_log: list[str] = []

    def fake_cancel(session_id: str) -> None:
        cancel_log.append(session_id)

    monkeypatch.setattr(harness.client, "cancel", fake_cancel)

    from acp_fleet_harness.models import ActiveTurn

    harness._active_turns["chat-stop"] = ActiveTurn("chat-stop", "s-1", "hello", time.time())

    result = harness.stop("chat-stop")
    assert "stopping" in result.reply.lower()
    assert "s-1" in cancel_log

    result2 = harness.stop("chat-no-active")
    assert "no running turn" in result2.reply.lower()


def test_process_includes_reply_to_quote_and_label(monkeypatch, tmp_path: Path) -> None:
    """A reply-to message is injected into the prompt with a clear label."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    prompts: list[str] = []

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        prompts.append(prompt)
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(
        harness.client,
        "send_message",
        lambda *a, **kwargs: AcpPromptResult(reply="Follow-up."),
    )

    result = harness.process(
        "chat-reply",
        "Can you explain this?",
        reply_to="This is the earlier bot message.",
        reply_to_is_bot=True,
    )
    assert result.reply == "Ready."
    prompt = prompts[0]
    assert "[In reply to the assistant's earlier message:]" in prompt
    assert "This is the earlier bot message." in prompt
    assert "[Your new message:]" in prompt
    assert "Can you explain this?" in prompt


def test_process_trims_long_reply_to_quote(monkeypatch, tmp_path: Path) -> None:
    """A reply-to quote longer than max_reply_quote_chars is trimmed."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    prompts: list[str] = []

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        prompts.append(prompt)
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    long_quote = "word " * 600  # 3000 characters, well above the 2048 default cap.
    result = harness.process("chat-long-reply", "Explain.", reply_to=long_quote)
    assert result.reply == "Ready."
    prompt = prompts[0]
    assert "[..." in prompt and "truncated" in prompt
    # The repeated quote was trimmed; there should be far fewer "word " occurrences
    # than in the original 600-repetition string.
    assert prompt.count("word ") < 500


def test_system_notice_uses_recall_numbers_not_disk_size(monkeypatch, tmp_path: Path) -> None:
    """_build_system_notice must not mix RecallResult.loaded with chat_status.total."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    from acp_fleet_harness.memory import RecallResult
    from acp_fleet_harness.persona_composer import PersonaPrompt

    path = tmp_path / "chat-MEMORY.md"
    recall = RecallResult(
        text="Recent conversation:...",
        truncated=True,
        memory_path=path,
        limit=8192,
        loaded=17539,
        total=17541,
    )
    chat_status = {
        "path": path,
        "limit": 8192,
        "total": 1855,
        "exceeded": False,
    }
    persona = PersonaPrompt(
        text="",
        memory_text="",
        memory_truncated=False,
        memory_path=None,
        limit=0,
        loaded=0,
        total=0,
    )

    notice = harness._build_system_notice(persona, recall, chat_status)
    assert notice is not None
    # The notice should use recall's loaded/total, not the raw file size.
    assert "17539 of 17541" in notice
    assert "1855" not in notice


def test_chat_memory_exceeded_flag_tracks_file_not_prompt(monkeypatch, tmp_path: Path) -> None:
    """The session record's chat_memory_exceeded flag should reflect the file, not recall truncation."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    config.engine.model = "swe-1-7"
    harness = ConversationHarness(config)

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(
        harness.client,
        "send_message",
        lambda *a, **kwargs: AcpPromptResult(reply="Follow-up."),
    )

    # Add enough long turns that the short-term transcript makes the prompt large.
    # Also make a keyword match from MEMORY.md so recall.truncated becomes true.
    mgr = harness._memory_manager("chat-flag")
    for i in range(5):
        mgr.record_turn(
            f"hello keyword turn {i} " * 100,
            f"assistant reply {i} " * 100,
            model="swe-1-7",
            turn_number=i + 1,
        )
    # The MEMORY.md file is still small; only recall is truncated.
    prompt, _, flags = harness._build_first_prompt("chat-flag", "hello keyword")
    assert (
        "Chat memory (short-term transcript + recalled content)" in prompt
        or not prompt.startswith("##")
    )
    # The record flag should be False because the on-disk file is not over cap.
    assert flags["chat_memory_exceeded"] is False

    # Now write enough to MEMORY.md to exceed the cap.
    memory_path = Path(mgr.chat_memory_path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text("x" * 9000)
    _, _, flags = harness._build_first_prompt("chat-flag", "hello keyword")
    assert flags["chat_memory_exceeded"] is True


def test_system_notice_uses_chat_status_when_file_exceeds(tmp_path: Path) -> None:
    """If the on-disk memory file exceeds the cap but recall is not truncated, use chat_status."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    from acp_fleet_harness.memory import RecallResult
    from acp_fleet_harness.persona_composer import PersonaPrompt

    path = tmp_path / "chat-MEMORY.md"
    recall = RecallResult(
        text="Recent conversation:...",
        truncated=False,
        memory_path=path,
        limit=8192,
        loaded=200,
        total=200,
    )
    chat_status = {
        "path": path,
        "limit": 8192,
        "total": 18555,
        "exceeded": True,
    }
    persona = PersonaPrompt(
        text="",
        memory_text="",
        memory_truncated=False,
        memory_path=None,
        limit=0,
        loaded=0,
        total=0,
    )

    notice = harness._build_system_notice(persona, recall, chat_status)
    assert notice is not None
    assert "8192 of 18555" in notice
    assert "(limit: 8192)" in notice


def test_process_uses_telegram_message_registry_for_bot_reply(monkeypatch, tmp_path: Path) -> None:
    """A known reply_to_message_id resolves to the registry preview, not the Telegram quote."""
    import json

    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    prompts: list[str] = []

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        prompts.append(prompt)
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    registry_path = harness._telegram_message_registry_path("chat-reg")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "message_id": 42,
        "chat_id": "chat-reg",
        "session_number": 1,
        "turn_number": 2,
        "preview": "Short preview.",
        "original_length": 100,
        "kind": "reply",
        "timestamp": time.time(),
    }
    registry_path.write_text(json.dumps(entry) + "\n")

    result = harness.process(
        "chat-reg",
        "Explain this.",
        reply_to="The full original bot message that should not be quoted.",
        reply_to_is_bot=True,
        reply_to_message_id=42,
    )
    assert result.reply == "Ready."
    prompt = prompts[0]
    assert "[In reply to the assistant's earlier message (session 1, turn 2):]" in prompt
    assert "Short preview." in prompt
    assert "The full original bot message that should not be quoted." not in prompt
    assert "[Your new message:]" in prompt
    assert "Explain this." in prompt


def test_process_caps_long_bot_reply_quote(monkeypatch, tmp_path: Path) -> None:
    """A reply to a long bot message is capped at max_bot_reply_quote_chars."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    prompts: list[str] = []

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        prompts.append(prompt)
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    long_reply = "word " * 100  # 500 characters
    result = harness.process(
        "chat-bot-long",
        "Expand.",
        reply_to=long_reply,
        reply_to_is_bot=True,
    )
    assert result.reply == "Ready."
    prompt = prompts[0]
    assert "[In reply to the assistant's earlier message:]" in prompt
    assert "[..." in prompt and "truncated" in prompt
    assert prompt.count("word ") < 60


def test_process_streams_partial_message_and_thought(monkeypatch, tmp_path: Path) -> None:
    """process() updates active_turns while the ACP call is in flight."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    call_event = threading.Event()
    done_event = threading.Event()
    captured_callbacks: list[tuple[str, Any]] = []

    def fake_create_session(
        prompt: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        on_chunk: Any | None = None,
        on_update: Any | None = None,
        **kwargs: Any,
    ) -> AcpPromptResult:
        # Simulate ACP streaming two message chunks and a thought chunk.
        for text in ["Hello, ", "world!"]:
            if on_chunk:
                on_chunk(text)
                captured_callbacks.append(("chunk", text))
        if on_update:
            on_update(
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "thinking..."},
                }
            )
            captured_callbacks.append(("thought", "thinking..."))
        call_event.set()
        done_event.wait(timeout=5.0)
        return AcpPromptResult(reply="Hello, world!", session_id="session-stream")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(
        harness.client,
        "send_message",
        lambda *a, **kwargs: AcpPromptResult(reply="Follow-up."),
    )

    result_holder: dict[str, Any] = {}

    def run_process() -> None:
        result_holder["result"] = harness.process("chat-stream", "hi")

    worker = threading.Thread(target=run_process)
    worker.start()

    # Wait until the fake ACP call has invoked callbacks.
    call_event.wait(timeout=5.0)

    # While the ACP call is still in flight, /turn can see partial state.
    status = harness.turn_status("chat-stream")
    assert status["status"] == "running"
    assert status["message_text"] == "Hello, world!"
    assert status["thought_text"] == "thinking..."

    # Let the fake ACP call complete and the worker finish.
    done_event.set()
    worker.join(timeout=5.0)

    result = result_holder["result"]
    assert result.reply == "Hello, world!"
    assert result.session_id == "session-stream"

    # Both chunks were passed to on_chunk and the thought to on_update.
    assert ("chunk", "Hello, ") in captured_callbacks
    assert ("chunk", "world!") in captured_callbacks
    assert ("thought", "thinking...") in captured_callbacks


def test_process_records_turn_metrics(monkeypatch, tmp_path: Path) -> None:
    """process() returns latency and ACP usage in ChatResult.metrics."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(
        prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs: Any
    ):
        return AcpPromptResult(
            reply="Ready.",
            session_id="session-1",
            usage={
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
                "cachedReadTokens": 25,
            },
        )

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    result = harness.process("chat-metrics", "hello")
    assert result.reply == "Ready."
    assert result.metrics is not None
    assert result.metrics["input_tokens"] == 100
    assert result.metrics["output_tokens"] == 50
    assert result.metrics["total_tokens"] == 150
    assert result.metrics["cached_tokens"] == 25
    assert result.metrics["model"] == "swe-1-7"
    assert result.metrics["turn_number"] == 1
    assert result.metrics["latency_seconds"] >= 0

    status = harness.status("chat-metrics")
    assert status["last_turn_metrics"] == result.metrics
    assert status["cumulative_metrics"]["turns"] == 1
    assert status["cumulative_metrics"]["total_tokens"] == 150


def test_metrics_exposed_in_prompt_when_enabled(monkeypatch, tmp_path: Path) -> None:
    """When expose_in_prompt is true, cumulative usage is injected into the prompt."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = Config(
        devin=DevinConfig(bin="/bin/echo", model="swe-1-7"),
        persona=PersonaConfig(
            name="test-pilot",
            profile_root=fixture_root,
            fleet_root=tmp_path / "fleet",
        ),
        harness=HarnessConfig(
            sessions_root=tmp_path / "sessions",
            session_store_path=tmp_path / "sessions.jsonl",
            memory={"backend": "file"},  # type: ignore[arg-type]
            metrics={"expose_in_prompt": True},  # type: ignore[arg-type]
        ),
        secrets=Secrets(WINDSURF_API_KEY="test-key"),
    )
    harness = ConversationHarness(config)

    prompts: list[str] = []

    def fake_create_session(
        prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs: Any
    ):
        prompts.append(prompt)
        return AcpPromptResult(
            reply="Ready.",
            session_id="session-1",
            usage={"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        )

    def fake_send_message(
        session_id: str,
        prompt: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AcpPromptResult:
        prompts.append(prompt)
        return AcpPromptResult(
            reply="Follow-up.",
            usage={"inputTokens": 20, "outputTokens": 10, "totalTokens": 30},
        )

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    # First turn: no prior metrics, so no usage context yet.
    harness.process("chat-prompt-metrics", "hello")
    first_prompt = prompts[0]
    assert "## Cumulative usage" not in first_prompt

    # Follow-up: compact metrics line appended.
    harness.process("chat-prompt-metrics", "again")
    follow_prompt = prompts[1]
    assert "[Cumulative usage:" in follow_prompt
    assert "15 tokens" in follow_prompt

    # New session after model switch: full usage block.
    harness.process("chat-prompt-metrics", "new model", model="glm-5-2")
    new_prompt = prompts[2]
    assert "## Cumulative usage" in new_prompt
    assert "45 total token(s)" in new_prompt


def test_get_metrics_aggregates_per_chat_and_globally(monkeypatch, tmp_path: Path) -> None:
    """get_metrics returns per-chat and global cumulative metrics."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(
        prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs: Any
    ):
        return AcpPromptResult(
            reply="Ready.",
            session_id="session-1",
            usage={"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
        )

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    harness.process("chat-a", "hello")
    harness.process("chat-b", "hello")

    chat_a = harness.get_metrics("chat-a")
    assert chat_a["cumulative"]["turns"] == 1
    assert chat_a["cumulative"]["total_tokens"] == 150

    global_metrics = harness.get_metrics()
    assert global_metrics["global"]["turns"] == 2
    assert global_metrics["global"]["total_tokens"] == 300


def test_new_session_uses_enabled_mcp_servers(monkeypatch, tmp_path: Path) -> None:
    """New sessions pass the enabled mcpServers list into create_session."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    config.harness.mcp = McpConfig(
        servers=[
            McpServerConfig(
                name="github",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env=[],
            ),
        ],
        default_enabled=["github"],
    )
    harness = ConversationHarness(config)

    captured: dict[str, Any] = {}

    def fake_create_session(
        prompt: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AcpPromptResult:
        captured["mcp_servers"] = mcp_servers
        return AcpPromptResult(reply="ok", session_id="session-1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    harness.new_session("chat-123")
    assert captured["mcp_servers"] == [
        {
            "name": "github",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": [],
        }
    ]


def test_mcp_enable_disable_persists(monkeypatch, tmp_path: Path) -> None:
    """Enabled/disabled MCP servers are stored on the active record."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    config.harness.mcp = McpConfig(
        servers=[
            McpServerConfig(name="github", command="npx", args=["-y"], env=[]),
            McpServerConfig(name="local", command="/bin/echo", args=["x"], env=[]),
        ],
        default_enabled=["github"],
    )
    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, *, mcp_servers=None, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="session-1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    harness.new_session("chat-abc")
    assert harness.status("chat-abc")["enabled_mcp_servers"] == ["github"]

    harness.mcp_enable("chat-abc", "local")
    assert harness.status("chat-abc")["enabled_mcp_servers"] == ["github", "local"]

    harness.mcp_disable("chat-abc", "github")
    assert harness.status("chat-abc")["enabled_mcp_servers"] == ["local"]


def test_new_session_syncs_skills(monkeypatch, tmp_path: Path) -> None:
    """New sessions sync enabled skills into the chat working directory."""
    from acp_fleet_harness.config import SkillsConfig

    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    shared = tmp_path / "personas" / "shared"
    shared.mkdir(parents=True)
    skill_dir = shared / "skills" / "review"
    skill_dir.mkdir(parents=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "---\nname: review\n---\n\nReview changes.\n",
        encoding="utf-8",
    )
    config.harness.skills = SkillsConfig(shared_root=shared, default_enabled=["review"])

    harness = ConversationHarness(config)

    def fake_create_session(prompt: str, *, mcp_servers=None, **kwargs: Any) -> AcpPromptResult:
        return AcpPromptResult(reply="ok", session_id="session-1")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    harness.new_session("chat-123")
    assert (harness._chat_dir("chat-123") / ".devin" / "skills" / "review" / "SKILL.md").exists()


def test_soft_timeout_invites_continue(monkeypatch, tmp_path: Path) -> None:
    """A cancelled/partial ACP result prompts the user to Continue."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(
        prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs: Any
    ):
        return AcpPromptResult(
            reply="Partial.",
            session_id="s-1",
            partial=True,
            cancelled=False,
            stop_reason="cancelled",
            timed_out=True,
        )

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    result = harness.process("chat-partial", "hello")
    assert "Partial" in result.reply
    assert result.notice and "Continue" in result.notice


def test_continue_uses_same_session_after_soft_timeout(monkeypatch, tmp_path: Path) -> None:
    """After a soft timeout, 'Continue' reuses the same ACP session and adds an anchor."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(
        prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs: Any
    ):
        return AcpPromptResult(
            reply="Partial.",
            session_id="s-1",
            partial=True,
            cancelled=False,
            stop_reason="cancelled",
            timed_out=True,
        )

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    result1 = harness.process("chat-continue", "hello")
    assert result1.session_number == 1
    assert result1.session_id == "s-1"
    assert harness._active_record("chat-continue").last_stop_reason == "cancelled"

    sent: list[tuple[str, str]] = []

    def fake_send_message(
        session_id: str,
        prompt: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AcpPromptResult:
        sent.append((session_id, prompt))
        return AcpPromptResult(reply="Continued.", session_id=session_id)

    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    result2 = harness.process("chat-continue", "Continue")
    assert result2.session_number == 1
    assert result2.session_id == "s-1"
    assert sent[0][0] == "s-1"
    assert "interrupted" in sent[0][1]


def test_user_stop_does_not_auto_continue(monkeypatch, tmp_path: Path) -> None:
    """A /stop marks the turn as 'stopped' so auto_continue does not restart it."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(
        prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs: Any
    ):
        active = harness._active_turns.get("chat-stop")
        if active is not None:
            active.stopped = True
        return AcpPromptResult(
            reply="Partial.",
            session_id="s-1",
            partial=True,
            cancelled=True,
            stop_reason="cancelled",
        )

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    result1 = harness.process("chat-stop", "hello")
    assert result1.session_number == 1
    assert result1.session_id == "s-1"
    assert harness._active_record("chat-stop").last_stop_reason == "stopped"


def test_hard_timeout_rehydrates_and_restarts_transport(monkeypatch, tmp_path: Path) -> None:
    """A hard timeout forces a new session; a stuck transport is restarted once."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    def fake_create_session(
        prompt: str, *, cwd: Path | None = None, model: str | None = None, **kwargs: Any
    ):
        return AcpPromptResult(
            reply="",
            session_id="s-1",
            partial=True,
            stop_reason="timeout",
            timed_out=True,
        )

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)

    result1 = harness.process("chat-hard", "hello")
    assert result1.session_number == 1
    assert result1.reply == ""
    assert "Continue" in (result1.notice or "")
    assert harness._active_record("chat-hard").last_stop_reason == "timeout"

    restarts: list[None] = []
    monkeypatch.setattr(harness.client, "restart_transport", lambda: restarts.append(None))

    call_count = [0]

    def fake_create_session2(
        prompt: str,
        *,
        cwd: Path | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AcpPromptResult:
        call_count[0] += 1
        if call_count[0] == 1:
            raise TimeoutError("stuck")
        return AcpPromptResult(reply="Resumed.", session_id="s-2")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session2)

    result2 = harness.process("chat-hard", "Continue")
    assert result2.session_number == 2
    assert result2.session_id == "s-2"
    assert "Resumed." in result2.reply
    assert len(restarts) == 1
    assert call_count[0] == 2


def test_continuation_triggers_match_punctuation(monkeypatch, tmp_path: Path) -> None:
    """Continuation triggers ignore surrounding punctuation and whitespace."""
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    assert harness.is_continuation_message("Continue.") is True
    assert harness.is_continuation_message("  Go On!  ") is True
    assert harness.is_continuation_message("proceed") is True
    assert harness.is_continuation_message("what to continue") is False


def test_harness_has_dispatch_store_and_notifier(tmp_path: Path) -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)
    assert harness.dispatch_store is not None
    assert harness.notifier is not None


def test_harness_continue_turn_after_dispatch(monkeypatch, tmp_path: Path) -> None:
    from acp_fleet_harness.acp_client import AcpPromptResult

    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    call_log: list[tuple[str, ...]] = []

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        call_log.append(("create", prompt[:50], str(cwd), model))
        return AcpPromptResult(reply="Ready.", session_id="session-1")

    def fake_send_message(session_id, prompt, *, cwd=None, model=None, **kwargs):
        call_log.append(("send", session_id, prompt, model))
        return AcpPromptResult(reply="Done.")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    # First user turn.
    harness.process("chat-1", "Please dispatch a task")

    # Create dispatch.
    chat_result = harness.dispatch("chat-1")

    # Simulate external worker completing the dispatch.
    result = harness.continue_turn(
        chat_result.dispatch_id, "Background task completed with result."
    )

    assert result.reply == "Done."
    assert any(c[0] == "send" for c in call_log)
    send_call = next(c for c in call_log if c[0] == "send")
    assert "Background task completed" in send_call[2]
    assert "Continue" in send_call[2]

    completed = harness.dispatch_store.get(chat_result.dispatch_id)
    assert completed is not None
    assert completed.status.value == "completed"


def test_harness_continue_turn_rehydrates_stale_session(monkeypatch, tmp_path: Path) -> None:
    from acp_fleet_harness.acp_client import AcpPromptResult

    fixture_root = Path(__file__).parent / "fixtures" / "test-pilot"
    config = _make_config(tmp_path, fixture_root)
    harness = ConversationHarness(config)

    call_log: list[tuple[str, ...]] = []

    def fake_create_session(prompt, *, cwd=None, model=None, **kwargs):
        call_log.append(("create",))
        return AcpPromptResult(reply="Rehydrated.", session_id="session-2")

    def fake_send_message(session_id, prompt, *, cwd=None, model=None, **kwargs):
        call_log.append(("send",))
        raise RuntimeError("ACP session/prompt failed: Session not found")

    monkeypatch.setattr(harness.client, "create_session", fake_create_session)
    monkeypatch.setattr(harness.client, "send_message", fake_send_message)

    harness.process("chat-1", "Please dispatch")
    chat_result = harness.dispatch("chat-1")
    result = harness.continue_turn(chat_result.dispatch_id, "result")

    assert result.reply == "Rehydrated."
    assert ("create",) in call_log
