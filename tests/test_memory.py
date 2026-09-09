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


def test_file_backend_append_system_note(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path, "chat-1")
    backend.append_system_note("[mesh-dsn] delivered")
    entries = backend.load_transcript()
    assert len(entries) == 1
    assert entries[0] == {"role": "system", "content": "[mesh-dsn] delivered"}


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


def test_file_backend_recall_searches_archive(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path, "chat-1")
    backend.retain([MemoryItem(content="Postgres is the current database", tags=["memory"])])
    backend._archive_path.write_text("## 2026-09-01 (memory)\n\nold project used SQLite\n")
    result = backend.recall("SQLite")
    assert "SQLite" in result
    assert "Memory (archive):" in result
    assert "Postgres" not in result


def test_file_backend_recall_active_outranks_archive(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path, "chat-1")
    backend.retain([MemoryItem(content="Postgres is the current database", tags=["memory"])])
    backend._archive_path.write_text("## 2026-09-01 (memory)\n\nold project used a SQLite database\n")
    result = backend.recall("database")
    assert "Postgres" in result
    assert "SQLite" in result
    active_pos = result.find("Postgres")
    archive_pos = result.find("SQLite")
    assert active_pos < archive_pos


def test_file_backend_recall_archive_no_match(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path, "chat-1")
    backend._archive_path.write_text("## 2026-09-01 (memory)\n\nold project used SQLite\n")
    result = backend.recall("completely unrelated")
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
    assert len(cache_files) >= 1


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


def test_retain_auto_promotes_matching_tags(tmp_path: Path) -> None:
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
    manager.retain("I prefer tea.", tags=["preference"])

    assert manager.promoted_memory_path.exists()
    assert "I prefer tea." in manager.promoted_memory_path.read_text()
    assert "promoted" in manager._file_backend._load_memory_text()


def test_retain_auto_promotes_matching_content_triggers(tmp_path: Path) -> None:
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
    manager.retain("We decided to use Postgres for the store.", tags=["memory"])

    assert manager.promoted_memory_path.exists()
    assert "Postgres" in manager.promoted_memory_path.read_text()


def test_retain_no_promote_tag_skips_auto_promote(tmp_path: Path) -> None:
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
    manager.retain("We decided to use Postgres.", tags=["memory", "no-promote"])

    assert not manager.promoted_memory_path.exists()


def test_promoted_memory_caps_and_dedupes(tmp_path: Path) -> None:
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        pass

    persona = PersonaConfig(name="test-persona", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(backend="file", max_promoted_lines=3)
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    for i in range(5):
        manager.promote(f"fact {i}")

    lines = manager.promoted_memory_path.read_text().splitlines()
    assert len(lines) == 3
    assert lines[0] == "- fact 2"
    assert lines[-1] == "- fact 4"

    # Re-promoting an existing fact is a no-op, not an append-and-collapse.
    manager.promote("fact 4")
    lines = manager.promoted_memory_path.read_text().splitlines()
    assert lines == ["- fact 2", "- fact 3", "- fact 4"]


def test_promoted_memory_dedupes_non_adjacent_lines(tmp_path: Path) -> None:
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
    path = manager.promoted_memory_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("- alpha\n- beta\n- alpha\n- gamma\n- beta\n", encoding="utf-8")

    manager._tidy_promoted_memory()

    assert path.read_text(encoding="utf-8").splitlines() == [
        "- alpha",
        "- beta",
        "- gamma",
    ]


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


def test_chat_memory_block_points_to_archive(tmp_path: Path) -> None:
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
    fb.retain([MemoryItem(content="old section " + "x" * 400, tags=["memory"])])
    fb.retain([MemoryItem(content="newest section", tags=["memory"])])

    archive = fb._memory_path.with_name("chat_MEMORY_archive.md")
    archive.write_text("# archived\n", encoding="utf-8")
    block = mgr.chat_memory_block(max_chars=200)
    assert block is not None
    assert "newest section" in block
    assert "chat_MEMORY_archive.md" in block

    # Without an archive sibling, a trimmed block carries no pointer.
    archive.unlink()
    block = mgr.chat_memory_block(max_chars=200)
    assert block is not None
    assert "archive" not in block.lower()

    # Under the cap, the block is untouched even if an archive exists.
    archive.write_text("# archived\n", encoding="utf-8")
    block = mgr.chat_memory_block(max_chars=100000)
    assert block is not None
    assert "archive" not in block.lower()


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


class _RecordingBackend:
    """Minimal backend stub that records retained items."""

    def __init__(self) -> None:
        self.items: list[MemoryItem] = []
        self.closed = False

    def retain(self, items: list[MemoryItem]) -> None:
        self.items.extend(items)

    def health(self) -> bool:
        return True

    def recall(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def stats(self) -> dict[str, Any]:
        return {}

    def append_system_note(self, text: str) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _recording_manager(tmp_path: Path, **mem_kwargs: Any):
    from diploid_agent.config import MemoryConfig, PersonaConfig

    class FakeClient:
        pass

    persona = PersonaConfig(name="test-persona", profile_root=tmp_path / "persona")
    persona.profile_root.mkdir(parents=True, exist_ok=True)
    config = MemoryConfig(
        backend="file",
        precompute_short_term_summary=False,
        **mem_kwargs,
    )
    manager = MemoryManager(
        config=config,
        persona=persona,
        sessions_root=tmp_path,
        chat_id="chat-1",
        devin_client=FakeClient(),
    )
    backend = _RecordingBackend()
    manager.backend = backend
    return manager, backend


def test_record_turn_final_segment_replaces_narration(tmp_path: Path) -> None:
    """With retain_final_segment, only the post-tool reply segment is retained;
    the transcript still records the full reply."""
    manager, backend = _recording_manager(
        tmp_path, retain_final_segment=True, retain_min_final_chars=10
    )
    full = "Let me check the logs first.\n\nThe real answer is 42."
    manager.record_turn(
        "hi",
        full,
        model="m1",
        turn_number=1,
        session_number=1,
        final_segment="The real answer is 42.",
    )
    assert len(backend.items) == 1
    content = backend.items[0].content
    assert content.endswith("Assistant: The real answer is 42.")
    assert "Let me check" not in content
    transcript = manager._load_transcript()
    assert transcript[1]["content"] == full


def test_record_turn_final_segment_short_falls_back(tmp_path: Path) -> None:
    """A final segment below retain_min_final_chars retains the full reply."""
    manager, backend = _recording_manager(
        tmp_path, retain_final_segment=True, retain_min_final_chars=200
    )
    full = "working narration " * 30 + "Done."
    manager.record_turn(
        "hi",
        full,
        model="m1",
        turn_number=1,
        session_number=1,
        final_segment="Done.",
    )
    assert backend.items[0].content.endswith(f"Assistant: {full}")


def test_record_turn_final_segment_disabled_by_default(tmp_path: Path) -> None:
    manager, backend = _recording_manager(tmp_path)
    manager.record_turn(
        "hi", "narration\n\nanswer", model="m1", turn_number=1, final_segment="answer"
    )
    assert "Assistant: narration\n\nanswer" in backend.items[0].content


def test_record_turn_bundles_turns(tmp_path: Path) -> None:
    """Pairs accumulate until retain_bundle_turns, then flush as one document."""
    manager, backend = _recording_manager(tmp_path, retain_bundle_turns=3)
    manager.record_turn("u1", "a1", model="m", turn_number=1, session_number=1)
    manager.record_turn("u2", "a2", model="m", turn_number=2, session_number=1)
    assert not backend.items
    manager.record_turn("u3", "a3", model="m", turn_number=3, session_number=1)
    assert len(backend.items) == 1
    item = backend.items[0]
    assert item.document_id == "turns-chat-1-000001-000001-000003"
    assert item.metadata["role"] == "pair_bundle"
    assert item.metadata["turns"] == [1, 2, 3]
    assert "User: u1" in item.content
    assert "Assistant: a3" in item.content
    assert "---" in item.content
    assert "session:1" in item.tags


def test_record_turn_single_turn_keeps_turn_document_id(tmp_path: Path) -> None:
    """With bundling disabled the per-turn document id is unchanged."""
    manager, backend = _recording_manager(tmp_path)
    manager.record_turn("u1", "a1", model="m", turn_number=7, session_number=2)
    assert backend.items[0].document_id == "turn-chat-1-000002-000007"
    assert backend.items[0].metadata["role"] == "pair"


def test_record_turn_flushes_on_session_change(tmp_path: Path) -> None:
    """A session boundary flushes the pending bundle before buffering the new
    session's turn, and close() flushes the remainder."""
    manager, backend = _recording_manager(tmp_path, retain_bundle_turns=4)
    manager.record_turn("u1", "a1", model="m", turn_number=1, session_number=1)
    manager.record_turn("u2", "a2", model="m", turn_number=2, session_number=1)
    manager.record_turn("u3", "a3", model="m", turn_number=1, session_number=2)
    assert len(backend.items) == 1
    assert backend.items[0].metadata["session"] == 1
    assert backend.items[0].metadata["turns"] == [1, 2]
    assert len(manager._turn_buffer) == 1
    manager.close()
    assert len(backend.items) == 2
    assert backend.items[1].metadata["session"] == 2
    assert backend.items[1].metadata["role"] == "pair"


def test_record_turn_buffer_survives_restart(tmp_path: Path) -> None:
    """Buffered pairs persist to disk and are reloaded by a new manager."""
    manager, backend = _recording_manager(tmp_path, retain_bundle_turns=4)
    manager.record_turn("u1", "a1", model="m", turn_number=1, session_number=1)
    manager.record_turn("u2", "a2", model="m", turn_number=2, session_number=1)
    assert not backend.items
    buffer_path = tmp_path / "chat-1" / "turn-retain-buffer.jsonl"
    assert buffer_path.exists()

    manager2, backend2 = _recording_manager(tmp_path, retain_bundle_turns=4)
    assert len(manager2._turn_buffer) == 2
    manager2.record_turn("u3", "a3", model="m", turn_number=3, session_number=1)
    manager2.record_turn("u4", "a4", model="m", turn_number=4, session_number=1)
    assert len(backend2.items) == 1
    assert backend2.items[0].metadata["turns"] == [1, 2, 3, 4]
    assert "User: u1" in backend2.items[0].content


def test_record_turn_flush_failure_keeps_buffer(tmp_path: Path) -> None:
    """A backend failure keeps the buffered pairs for the next flush attempt."""
    manager, backend = _recording_manager(tmp_path, retain_bundle_turns=2)

    class FailingBackend(_RecordingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def retain(self, items: list[MemoryItem]) -> None:
            self.calls += 1
            raise RuntimeError("backend down")

    failing = FailingBackend()
    manager.backend = failing
    manager.record_turn("u1", "a1", model="m", turn_number=1, session_number=1)
    manager.record_turn("u2", "a2", model="m", turn_number=2, session_number=1)
    assert failing.calls == 1
    assert len(manager._turn_buffer) == 2

    manager.backend = backend
    manager.record_turn("u3", "a3", model="m", turn_number=3, session_number=1)
    assert len(backend.items) == 1
    assert backend.items[0].metadata["turns"] == [1, 2, 3]


def test_record_turn_extra_items_retain_immediately(tmp_path: Path) -> None:
    """Plugin memory items are not delayed by the turn bundle buffer."""
    manager, backend = _recording_manager(tmp_path, retain_bundle_turns=4)
    extra = MemoryItem(content="body event", tags=["body"])
    manager.record_turn("u", "a", model="m", turn_number=1, session_number=1, extra_items=[extra])
    assert backend.items == [extra]
    assert len(manager._turn_buffer) == 1


def _fake_result(updates: list[dict[str, Any]] | None = None) -> Any:
    class _Result:
        pass

    result = _Result()
    result.updates = updates
    return result


def _msg(text: str) -> dict[str, Any]:
    return {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": text},
    }


def test_final_segment_reply_uses_last_tool_boundary() -> None:
    from diploid_agent.turn.process import TurnProcess

    result = _fake_result(
        [
            _msg("Working on it. "),
            {"sessionUpdate": "tool_call", "content": {}},
            {"sessionUpdate": "tool_call_update", "content": {}},
            {
                "sessionUpdate": "agent_message_chunk",
                "content": [{"type": "text", "text": "All done. "}],
            },
            _msg("Here is the answer."),
        ]
    )
    reply = "Working on it. All done. Here is the answer."
    assert TurnProcess._final_segment_reply(result, reply) == "All done. Here is the answer."


def test_final_segment_reply_no_tool_returns_none() -> None:
    from diploid_agent.turn.process import TurnProcess

    assert TurnProcess._final_segment_reply(_fake_result([_msg("hi")]), "hi") is None
    assert TurnProcess._final_segment_reply(_fake_result([]), "hi") is None
    assert TurnProcess._final_segment_reply(_fake_result(None), "hi") is None
    # Tool call last with no message after it -> no final segment.
    result = _fake_result([_msg("only narration"), {"sessionUpdate": "tool_call"}])
    assert TurnProcess._final_segment_reply(result, "only narration") is None


def test_final_segment_reply_all_post_tool_returns_reply() -> None:
    from diploid_agent.turn.process import TurnProcess

    result = _fake_result([{"sessionUpdate": "tool_call"}, _msg("whole reply")])
    assert TurnProcess._final_segment_reply(result, "whole reply") == "whole reply"
