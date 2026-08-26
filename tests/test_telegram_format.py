"""Tests for telegram_format."""

from __future__ import annotations

from diploid_agent.transport.telegram_format import (
    _strip_mdv2,
    format_markdown_v2,
    separate_chunk_indicator_from_fence,
    split_telegram_text,
    utf16_len,
)


def test_utf16_len_counts_surrogate_pairs() -> None:
    assert utf16_len("a") == 1
    assert utf16_len("\U0001f600") == 2


def test_format_bold_and_italic() -> None:
    out = format_markdown_v2("**bold** and *italic* and _underscore_")
    assert _strip_mdv2(out) == "bold and italic and underscore"
    assert "*bold*" in out
    assert "_italic_" in out
    assert "_underscore_" in out


def test_format_bold_with_underscores() -> None:
    out = format_markdown_v2("__init__")
    assert _strip_mdv2(out) == "init"


def test_format_link() -> None:
    out = format_markdown_v2("[Example](https://example.com)")
    assert out == "[Example](https://example.com)"


def test_format_header() -> None:
    out = format_markdown_v2("## Title")
    assert out == "*Title*"


def test_format_inline_and_fenced_code() -> None:
    out = format_markdown_v2("Use `print(x)` and:\n\n```python\nprint(1)\n```")
    assert "`print(x)`" in out
    assert "```python" in out
    assert "print(1)" in out


def test_format_strikethrough_and_spoiler() -> None:
    out = format_markdown_v2("~~del~~ and ||secret||")
    assert _strip_mdv2(out) == "del and secret"
    assert "~del~" in out
    assert "||secret||" in out


def test_format_blockquote() -> None:
    out = format_markdown_v2("> quoted text")
    assert _strip_mdv2(out) == "> quoted text"
    assert out.startswith(">")


def test_format_citation_tags() -> None:
    text = 'See <ref_file file="/tmp/foo.py" /> and <ref_snippet file="/tmp/bar.py" lines="1-5" />.'
    out = format_markdown_v2(text)
    stripped = _strip_mdv2(out)
    assert "Source: /tmp/foo.py" in stripped
    assert "Source: /tmp/bar.py" in stripped
    assert "(lines 1-5)" in stripped


def test_convert_table_to_bullets() -> None:
    text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    out = format_markdown_v2(text)
    stripped = _strip_mdv2(out)
    assert "*1*" in out
    assert "b: 2" in stripped


def test_split_preserves_fenced_code() -> None:
    code = "```python\n" + "x\n" * 2000 + "```"
    chunks = split_telegram_text(code, max_length=1000)
    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk should contain balanced fence pairs or proper carry.
        assert chunk.count("```") % 2 == 0


def test_split_avoids_inline_backtick_split() -> None:
    text = "Start `abc` end" + " word" * 1000
    chunks = split_telegram_text(text, max_length=200, len_fn=len)
    for chunk in chunks:
        # Every chunk should have an even number of unescaped backticks.
        unescaped = chunk.count("`") - chunk.count("\\`")
        assert unescaped % 2 == 0


def test_separate_chunk_indicator_from_fence() -> None:
    assert separate_chunk_indicator_from_fence("``` (1/2)") == "```\n(1/2)"
    assert separate_chunk_indicator_from_fence("``` \\(1/2\\)") == "```\n\\(1/2\\)"
