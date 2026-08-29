"""Tests for memory backends."""

import json
from pathlib import Path
from typing import Any

from diploid_agent.memory import (
    FileMemoryBackend,
    HindsightMemoryBackend,
    MemoryItem,
    MemoryManager,
    RecallResult,
)


def test_file_backend_append_and_load_transcript(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path, "chat-1")
    backend.append_transcript("hello", "hi there")
    entries = backend.load_transcript()
    assert len(entries) == 2
    assert entries[0] == {"role": "user", "content": "hello"}
    assert entries[1] == {"role": "assistant", "content": "hi there"}


def test_file_backend_retain_appends_to_memory(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path, "chat-1")
    item = MemoryItem(content="a promoted fact", tags=["memory"])
    backend.retain([item])
    assert backend._memory_path.exists()
    assert "a promoted fact" in backend._memory_path.read_text()


def test_file_backend_recall_matches_keyword(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path, "chat-1")
    backend.append_transcript("I like Python.", "Python is great.")
    item = MemoryItem(content="Python is the favorite language", tags=["memory"])
    backend.retain([item])
    result = backend.recall("Python")
    assert "Python" in result


def test_file_backend_recall_returns_empty_when_no_match(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path, "chat-1")
    backend.append_transcript("hello", "hi")
    result = backend.recall("Python")
    assert result == ""


def test_hindsight_spool_when_unhealthy(tmp_path: Path) -> None:
    backend = HindsightMemoryBackend(
        base_url="http://127.0.0.1:65535",
        bank="test",
        chat_id="chat-1",
        sessions_root=tmp_path,
        spool_path=tmp_path / "spool.jsonl",
    )
    item = MemoryItem(content="fact")
    backend.retain([item])
    assert (tmp_path / "spool.jsonl").exists()
    lines = (tmp_path / "spool.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert "fact" in lines[0]


def test_hindsight_spool_flush_checks_success_and_keeps_invalid_lines(
    tmp_path: Path, monkeypatch
) -> None:
    spool_path = tmp_path / "spool.jsonl"
    spool_path.write_text(
        json.dumps({"content": "good1", "document_id": "d1"})
        + "\n"
        + "not-json\n"
        + json.dumps({"content": "good2", "document_id": "d2"})
        + "\n"
    )
    backend = HindsightMemoryBackend(
        base_url="http://127.0.0.1:1",
        bank="test",
        chat_id="chat-1",
        sessions_root=tmp_path,
        spool_path=spool_path,
    )
    monkeypatch.setattr(backend, "health", lambda: True)
    posted: list[list[dict]] = []

    class OKResp:
        status_code = 200

        def json(self) -> dict:
            return {"success": True}

    def fake_post(url: str, *, json: dict, **kwargs: Any) -> OKResp:
        posted.append(json["items"])
        return OKResp()

    monkeypatch.setattr(backend._client, "post", fake_post)
    backend._flush_spool()

    remaining = spool_path.read_text().splitlines()
    assert len(remaining) == 1
    assert "not-json" in remaining[0]
    assert len(posted) == 1
    assert len(posted[0]) == 2


def test_hindsight_spool_flush_does_not_flush_on_failure(tmp_path: Path, monkeypatch) -> None:
    spool_path = tmp_path / "spool.jsonl"
    spool_path.write_text(json.dumps({"content": "good1", "document_id": "d1"}) + "\n")
    backend = HindsightMemoryBackend(
        base_url="http://127.0.0.1:1",
        bank="test",
        chat_id="chat-1",
        sessions_root=tmp_path,
        spool_path=spool_path,
    )
    monkeypatch.setattr(backend, "health", lambda: True)

    class FailResp:
        status_code = 200

        def json(self) -> dict:
            return {"success": False}

    monkeypatch.setattr(backend._client, "post", lambda *args, **kwargs: FailResp())
    backend._flush_spool()

    remaining = spool_path.read_text().splitlines()
    assert len(remaining) == 1
    assert "good1" in remaining[0]


def test_hindsight_spool_dead_letters_4xx(tmp_path: Path, monkeypatch) -> None:
    spool_path = tmp_path / "spool.jsonl"
    spool_path.write_text(json.dumps({"content": "bad", "document_id": "d1"}) + "\n")
    backend = HindsightMemoryBackend(
        base_url="http://127.0.0.1:1",
        bank="test",
        chat_id="chat-1",
        sessions_root=tmp_path,
        spool_path=spool_path,
    )
    monkeypatch.setattr(backend, "health", lambda: True)

    class BadRequestResp:
        status_code = 422
        text = "Unprocessable Entity"

    monkeypatch.setattr(backend._client, "post", lambda *args, **kwargs: BadRequestResp())
    backend._flush_spool()

    remaining = spool_path.read_text().splitlines()
    assert len(remaining) == 0
    dead_letter_path = spool_path.with_name("hindsight-dead-letter.jsonl")
    assert dead_letter_path.exists()
    entries = [json.loads(line) for line in dead_letter_path.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["reason"] == "422"
    assert entries[0]["item"]["document_id"] == "d1"


def test_hindsight_retain_rejects_empty_content(tmp_path: Path, monkeypatch) -> None:
    spool_path = tmp_path / "spool.jsonl"
    backend = HindsightMemoryBackend(
        base_url="http://127.0.0.1:1",
        bank="test",
        chat_id="chat-1",
        sessions_root=tmp_path,
        spool_path=spool_path,
    )
    monkeypatch.setattr(backend, "health", lambda: True)
    posted: list[list[dict]] = []

    class OKResp:
        status_code = 200

        def json(self) -> dict:
            return {"success": True}

    def fake_post(url: str, *, json: dict, **kwargs: Any) -> OKResp:
        posted.append(json["items"])
        return OKResp()

    monkeypatch.setattr(backend._client, "post", fake_post)
    empty = MemoryItem(content="", document_id="empty-1")
    good = MemoryItem(content="good", document_id="good-1")
    backend.retain([empty, good])

    dead_letter_path = spool_path.with_name("hindsight-dead-letter.jsonl")
    assert dead_letter_path.exists()
    entries = [json.loads(line) for line in dead_letter_path.read_text().splitlines()]
    assert len(entries) == 1
    assert "empty content" in entries[0]["reason"]
    assert len(posted) == 1
    assert posted[0][0]["document_id"] == "good-1"


def test_record_turn_document_id_is_unique_per_session(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        pass

    persona = PersonaConfig(name="test-persona", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(
        backend="hindsight",
        hindsight={
            "base_url": "http://127.0.0.1:1",
            "bank": "test",
            "spool_path": tmp_path / "spool.jsonl",
        },
    )
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    manager.record_turn("hi", "hello", model="m1", turn_number=1, session_number=1)
    manager.record_turn("hi", "hello", model="m1", turn_number=1, session_number=2)

    lines = (tmp_path / "spool.jsonl").read_text().splitlines()
    assert len(lines) == 2
    items = [json.loads(line) for line in lines]
    assert items[0]["document_id"] != items[1]["document_id"]
    assert "session:1" in items[0]["tags"]
    assert "session:2" in items[1]["tags"]
    assert "persona:test-persona" in items[0]["tags"]


def test_promote_to_persona_indexes_in_hindsight(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        pass

    persona = PersonaConfig(name="test-persona", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    spool_path = tmp_path / "spool.jsonl"
    config = MemoryConfig(
        backend="hindsight",
        hindsight={
            "base_url": "http://127.0.0.1:1",
            "bank": "test",
            "spool_path": spool_path,
        },
    )
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    manager.promote_to_persona("I like tea.")

    assert (persona.profile_root / "MEMORY.md").read_text().strip() == "- I like tea."
    lines = spool_path.read_text().splitlines()
    assert len(lines) == 1
    item = json.loads(lines[0])
    assert item["content"] == "I like tea."
    assert "persona" in item["tags"]
    assert "promoted" in item["tags"]
    assert f"persona:{persona.name}" in item["tags"]


def test_memory_manager_file_backend_recall(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        pass

    persona = PersonaConfig(name="test", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(backend="file")
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    manager.record_turn("hello", "hi", model="swe-1-7", turn_number=1)
    result = manager.recall_context("hello")
    assert isinstance(result, RecallResult)
    assert "hello" in result.text


def test_recall_context_loaded_not_greater_than_total(tmp_path: Path) -> None:
    """The recall report should never say loaded > total."""

    class FakeClient:
        pass

    from diploid_agent.config import MemoryConfig, PersonaConfig

    persona = PersonaConfig(name="test", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(backend="file", max_chat_memory_chars=200)
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    # Make the short-term transcript much larger than the cap.
    for i in range(5):
        manager.record_turn(
            f"This is a fairly long user message number {i} to fill the transcript.",
            f"This is a corresponding assistant reply number {i} that also has length.",
            model="swe-1-7",
            turn_number=i + 1,
        )
    result = manager.recall_context("message")
    assert result.loaded <= result.total
    assert result.text in result.text  # text is non-empty


def test_recall_context_short_term_always_included(tmp_path: Path) -> None:
    """The most recent user message must appear even if long-term recall is trimmed."""

    class FakeClient:
        pass

    from diploid_agent.config import MemoryConfig, PersonaConfig

    persona = PersonaConfig(name="test", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(backend="file", max_chat_memory_chars=100)
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    for i in range(3):
        manager.record_turn(
            f"user turn {i}",
            f"assistant reply {i}",
            model="swe-1-7",
            turn_number=i + 1,
        )
    result = manager.recall_context("turn 2")
    assert "user turn 2" in result.text or "assistant reply 2" in result.text


def test_smart_short_term_summarizes_older_turns(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        def create_session(self, *args, **kwargs):
            class _Result:
                reply = "Summary of older turns."

            return _Result()

        def prompt(self, request, *, session_id=None, on_chunk=None, on_update=None):
            return self.create_session(
                request.prompt,
                cwd=request.cwd,
                model=request.model,
                soft_timeout=request.soft_timeout,
            )

    persona = PersonaConfig(name="test", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(
        backend="file",
        short_term_strategy="smart",
        short_term_turns=5,
        min_short_term_turns=2,
        max_short_term_chars=1000,
    )
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    for i in range(5):
        manager.record_turn(
            f"long user message {i} " * 10,
            f"long assistant reply {i} " * 10,
            model="swe-1-7",
            turn_number=i + 1,
        )
    result = manager._short_term_context()
    assert "Summary of older turns." in result
    assert "long user message 4" in result or "long assistant reply 4" in result
    assert "long user message 0" not in result

    chat_dir = tmp_path / "chat-1"
    assert not list(chat_dir.glob(".short-term-summary-*.md"))
    cache_files = list((chat_dir / ".cache").glob("*.md"))
    assert len(cache_files) == 1


def test_memory_manager_retain_appends_to_file(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        pass

    persona = PersonaConfig(name="test-persona", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(backend="file")
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    manager.retain("We agreed on tea.", tags=["agreement", "drink"], context="preference")

    text = manager._file_backend._load_memory_text()
    assert "We agreed on tea." in text
    assert "agreement" in text


def test_record_turn_uses_notice_when_reply_empty(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        pass

    persona = PersonaConfig(name="test-persona", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(backend="file")
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    manager.record_turn(
        "hi",
        "",
        model="m1",
        turn_number=1,
        notice="The agent was stopped before completing its reply.",
    )

    transcript = manager._load_transcript()
    assert len(transcript) == 2
    assert transcript[0] == {"role": "user", "content": "hi"}
    assert transcript[1] == {
        "role": "assistant",
        "content": "The agent was stopped before completing its reply.",
    }


def test_persona_memory_loads_and_truncates(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        pass

    persona = PersonaConfig(name="test-persona", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    memory_path = persona.profile_root / "MEMORY.md"
    memory_path.write_text("This is the persona memory content.")

    config = MemoryConfig(backend="file")
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )

    result = manager.persona_memory(max_chars=1000)
    assert result["text"] == "This is the persona memory content."
    assert result["total"] == 35
    assert result["truncated"] is False
    assert result["path"] == memory_path

    result = manager.persona_memory(max_chars=10)
    assert result["truncated"] is True
    assert result["loaded"] <= 10
    assert result["path"] == memory_path


def test_chat_memory_block_returns_last_blocks(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig
    from diploid_agent.engine.fake import FakeAgentEngine

    persona = PersonaConfig(name="test", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(backend="file")
    mgr = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeAgentEngine(),
    )
    fb = mgr._file_backend
    assert fb is not None
    fb.retain([MemoryItem(content="first summary", tags=["memory", "summary"])])
    fb.retain([MemoryItem(content="second summary", tags=["memory", "summary"])])
    block = mgr.chat_memory_block(max_chars=256)
    assert block is not None
    assert "second summary" in block


def test_summarize_mirrors_to_file_backend(tmp_path: Path, monkeypatch) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig
    from diploid_agent.engine.fake import FakeAgentEngine

    persona = PersonaConfig(name="test", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    engine = FakeAgentEngine(replies=["We agreed on Postgres."])
    config = MemoryConfig(
        backend="hindsight",
        n_turns_summarization=2,
        hindsight={
            "base_url": "http://127.0.0.1:1",
            "bank": "test",
            "spool_path": tmp_path / "spool.jsonl",
            "fallback_to_file": True,
        },
    )
    mgr = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=engine,
    )
    monkeypatch.setattr(mgr.backend, "health", lambda: False)
    mgr.record_turn("hi", "hello", model="m1", turn_number=1)
    mgr.record_turn("how are you", "fine", model="m1", turn_number=2)

    # The local file mirror should contain the summary even though the active backend is Hindsight.
    fb = mgr._file_backend
    assert fb is not None
    memory_text = fb._load_memory_text()
    assert "We agreed on Postgres." in memory_text

    # The active Hindsight backend should also have received the summary.
    spool_lines = (tmp_path / "spool.jsonl").read_text().splitlines()
    assert any("We agreed on Postgres." in line for line in spool_lines)
