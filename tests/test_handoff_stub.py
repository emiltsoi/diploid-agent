"""Tests for the harness-written handoff stub (context/handoff.py).

The stub is the floor under the handoff convention: when pressure folds a
session, a turn that died before writing `## next-self` must not leave the
next self with nothing — and the harness must never overwrite a section the
previous turn authored itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from diploid_agent.context.handoff import handoff_stub_path, write_handoff_stub


def _write(path: Path, **kwargs) -> bool:
    defaults = {"session_number": 3, "turn_number": 41, "arriving": "check the rollout"}
    defaults.update(kwargs)
    return write_handoff_stub(path, **defaults)


def test_stub_path_uses_chat_dir(tmp_path: Path) -> None:
    path = handoff_stub_path(tmp_path / "sessions", "chat/1")
    assert path == tmp_path / "sessions" / "chat_1" / "chat_self_state.md"


def test_writes_stub_into_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "chat-1" / "chat_self_state.md"
    assert _write(path) is True
    text = path.read_text(encoding="utf-8")
    assert "## next-self" in text
    assert "<!-- harness-handoff -->" in text
    assert "check the rollout" in text
    assert "session 3" in text


def test_reads_interrupted_turn_snapshot(tmp_path: Path) -> None:
    chat_dir = tmp_path / "chat-1"
    chat_dir.mkdir(parents=True)
    (chat_dir / "chat_interrupted_turn.json").write_text(
        json.dumps(
            {
                "current_intent": "reviewing the canary instrument",
                "last_side_effect": "exec sha256sum fleet-venv-extras-verify.py",
            }
        ),
        encoding="utf-8",
    )
    path = chat_dir / "chat_self_state.md"
    assert _write(path) is True
    text = path.read_text(encoding="utf-8")
    assert "reviewing the canary instrument" in text
    assert "exec sha256sum" in text


def test_appends_after_existing_content(tmp_path: Path) -> None:
    path = tmp_path / "chat-1" / "chat_self_state.md"
    path.parent.mkdir(parents=True)
    path.write_text("# self state\n\nfelt: warm\n", encoding="utf-8")
    assert _write(path) is True
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# self state")
    assert text.rstrip().endswith("<!-- /harness-handoff -->")


def test_model_authored_next_self_wins(tmp_path: Path) -> None:
    path = tmp_path / "chat-1" / "chat_self_state.md"
    path.parent.mkdir(parents=True)
    original = "# self state\n\n## next-self\n\nI wrote this myself.\n"
    path.write_text(original, encoding="utf-8")
    assert _write(path) is False
    assert path.read_text(encoding="utf-8") == original


def test_marker_block_refreshes_in_place(tmp_path: Path) -> None:
    path = tmp_path / "chat-1" / "chat_self_state.md"
    assert _write(path, arriving="first message") is True
    # A second fold refreshes the stub rather than stacking another.
    assert _write(path, arriving="second message") is True
    text = path.read_text(encoding="utf-8")
    assert text.count("<!-- harness-handoff -->") == 1
    assert "second message" in text
    assert "first message" not in text


def test_unwritable_path_returns_false(tmp_path: Path) -> None:
    # A directory where the file should be: every filesystem op fails politely.
    path = tmp_path / "chat-1" / "chat_self_state.md"
    path.parent.mkdir(parents=True)
    path.mkdir()
    assert _write(path) is False
