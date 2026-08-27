"""Tests for interactive question helpers."""

from diploid_agent.transport.interactive import (
    build_reply_keyboard,
    extract_ask_block,
)


def test_extract_ask_block_with_question_in_json() -> None:
    text = (
        "Which file should I edit?\n\n"
        "```ask\n"
        '{"question": "Which file should I edit?", "options": ["a.py", "b.py"]}\n'
        "```"
    )
    _visible, block = extract_ask_block(text)
    assert block is not None
    assert block.question == "Which file should I edit?"
    assert block.options == ["a.py", "b.py"]
    assert "```ask" not in _visible
    assert "a.py" not in _visible


def test_extract_ask_block_derives_question_from_text() -> None:
    text = 'Which file?\n\n```ask\n{"options": ["a.py", "b.py"]}\n```'
    _visible, block = extract_ask_block(text)
    assert block is not None
    assert block.question == "Which file?"
    assert block.options == ["a.py", "b.py"]


def test_extract_ask_block_returns_none_when_missing() -> None:
    text = "Just a normal reply."
    _visible, block = extract_ask_block(text)
    assert block is None
    assert _visible == text


def test_extract_ask_block_returns_none_on_invalid_json() -> None:
    text = "```ask\nnot json\n```"
    _visible, block = extract_ask_block(text)
    assert block is None
    assert "not json" in _visible


def test_build_reply_keyboard() -> None:
    markup = build_reply_keyboard(["A", "B"])
    assert markup["resize_keyboard"] is True
    assert markup["one_time_keyboard"] is True
    assert markup["keyboard"] == [[{"text": "A"}], [{"text": "B"}]]
