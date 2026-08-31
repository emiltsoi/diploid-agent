"""Tests for interactive question helpers."""

from diploid_agent.transport.interactive import (
    build_empty_inline_keyboard,
    build_inline_keyboard,
    build_keyboard_remove,
    build_reply_keyboard,
    extract_ask_block,
    is_ask_cancel_callback,
    parse_ask_callback_index,
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
    assert block.cancellable is True
    assert block.cancel_label == "Cancel"
    assert "```ask" not in _visible
    assert "a.py" not in _visible


def test_extract_ask_block_derives_question_from_text() -> None:
    text = 'Which file?\n\n```ask\n{"options": ["a.py", "b.py"]}\n```'
    _visible, block = extract_ask_block(text)
    assert block is not None
    assert block.question == "Which file?"
    assert block.options == ["a.py", "b.py"]
    assert block.cancellable is True
    assert block.cancel_label == "Cancel"


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


def test_build_reply_keyboard_with_cancel() -> None:
    markup = build_reply_keyboard(["A", "B"], cancel="Cancel")
    assert markup["keyboard"] == [
        [{"text": "A"}],
        [{"text": "B"}],
        [{"text": "Cancel"}],
    ]


def test_build_reply_keyboard_skips_duplicate_cancel() -> None:
    markup = build_reply_keyboard(["A", "Cancel"], cancel="Cancel")
    assert markup["keyboard"] == [[{"text": "A"}], [{"text": "Cancel"}]]


def test_extract_ask_block_cancellable() -> None:
    text = (
        "Should I continue?\n\n"
        "```ask\n"
        '{"question": "Should I continue?", "options": ["Yes", "No"], "cancellable": true}\n'
        "```"
    )
    _visible, block = extract_ask_block(text)
    assert block is not None
    assert block.cancellable is True
    assert block.cancel_label == "Cancel"


def test_extract_ask_block_custom_cancel_label() -> None:
    text = (
        "Should I continue?\n\n"
        "```ask\n"
        '{"question": "Should I continue?", "options": ["Yes", "No"], "cancellable": true, "cancel_label": "Dismiss"}\n'
        "```"
    )
    _visible, block = extract_ask_block(text)
    assert block is not None
    assert block.cancellable is True
    assert block.cancel_label == "Dismiss"


def test_extract_ask_block_not_cancellable() -> None:
    text = (
        "Should I continue?\n\n"
        "```ask\n"
        '{"question": "Should I continue?", "options": ["Yes", "No"], "cancellable": false}\n'
        "```"
    )
    _visible, block = extract_ask_block(text)
    assert block is not None
    assert block.cancellable is False
    assert block.cancel_label == ""


def test_extract_ask_block_strips_other_please_specify() -> None:
    text = (
        "Which file?\n\n"
        "```ask\n"
        '{"question": "Which file?", "options": ["a.py", "b.py", "Other (please specify)"]}\n'
        "```"
    )
    _visible, block = extract_ask_block(text)
    assert block is not None
    assert block.options == ["a.py", "b.py"]
    assert block.cancellable is True
    assert block.cancel_label == "Cancel"


def test_extract_ask_block_strips_other_please_specify_variant() -> None:
    text = (
        "Which file?\n\n"
        "```ask\n"
        '{"question": "Which file?", "options": ["a.py", "b.py", "others and specify"]}\n'
        "```"
    )
    _visible, block = extract_ask_block(text)
    assert block is not None
    assert block.options == ["a.py", "b.py"]
    assert block.cancellable is True
    assert block.cancel_label == "Cancel"


def test_build_keyboard_remove() -> None:
    markup = build_keyboard_remove()
    assert markup == {"remove_keyboard": True}


def test_build_inline_keyboard() -> None:
    markup = build_inline_keyboard(["A", "B"])
    assert markup["inline_keyboard"] == [
        [{"text": "A", "callback_data": "ask_0"}],
        [{"text": "B", "callback_data": "ask_1"}],
    ]


def test_build_inline_keyboard_with_cancel() -> None:
    markup = build_inline_keyboard(["A", "B"], cancel="Cancel")
    assert markup["inline_keyboard"] == [
        [{"text": "A", "callback_data": "ask_0"}],
        [{"text": "B", "callback_data": "ask_1"}],
        [{"text": "Cancel", "callback_data": "ask_cancel"}],
    ]


def test_build_inline_keyboard_skips_duplicate_cancel() -> None:
    markup = build_inline_keyboard(["A", "Cancel"], cancel="Cancel")
    assert markup["inline_keyboard"] == [
        [{"text": "A", "callback_data": "ask_0"}],
        [{"text": "Cancel", "callback_data": "ask_cancel"}],
    ]


def test_build_empty_inline_keyboard() -> None:
    markup = build_empty_inline_keyboard()
    assert markup == {"inline_keyboard": []}


def test_is_ask_cancel_callback() -> None:
    assert is_ask_cancel_callback("ask_cancel") is True
    assert is_ask_cancel_callback("ask_0") is False
    assert is_ask_cancel_callback("other") is False


def test_parse_ask_callback_index() -> None:
    assert parse_ask_callback_index("ask_0") == 0
    assert parse_ask_callback_index("ask_1") == 1
    assert parse_ask_callback_index("ask_cancel") is None
    assert parse_ask_callback_index("other_0") is None
    assert parse_ask_callback_index("ask_not_a_number") is None
