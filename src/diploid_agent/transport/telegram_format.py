"""MarkdownV2 formatting and code-fence-aware splitting for Telegram."""

from __future__ import annotations

import re
from collections.abc import Callable

__all__ = [
    "_escape_mdv2",
    "_prefix_within_utf16_limit",
    "_safe_slice_budget",
    "_strip_mdv2",
    "convert_table_to_bullets",
    "format_markdown_v2",
    "separate_chunk_indicator_from_fence",
    "split_telegram_text",
    "utf16_len",
]

# Matches every character that Telegram MarkdownV2 requires to be backslash-escaped
# when it appears outside a code span or fenced code block.
_MDV2_ESCAPE_RE = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")

# Citation tags produced by the diploid agent.
_REF_FILE_RE = re.compile(r'<ref_file\s+file="([^"]+)"\s*/>')
_REF_SNIPPET_RE = re.compile(r'<ref_snippet\s+file="([^"]+)"(?:\s+lines="([^"]+)")?\s*/>')


# ─── Length and escape helpers ───────────────────────────────────────────────


def utf16_len(s: str) -> int:
    """Count UTF-16 code units in *s*.

    Telegram's message-length limit (4096) is measured in UTF-16 code units,
    not Unicode code-points. Characters outside the Basic Multilingual Plane
    are encoded as surrogate pairs and therefore consume two UTF-16 code units
    each.
    """
    return len(s.encode("utf-16-le")) // 2


def _prefix_within_utf16_limit(s: str, limit: int) -> str:
    """Return the longest prefix of *s* whose UTF-16 length is <= *limit*."""
    if utf16_len(s) <= limit:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if utf16_len(s[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo]


def _safe_slice_budget(s: str, budget: int, len_fn: Callable[[str], int]) -> int:
    """Return the largest codepoint offset *n* such that len_fn(s[:n]) <= budget."""
    if len_fn(s) <= budget:
        return len(s)
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len_fn(s[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _escape_mdv2(text: str) -> str:
    """Escape Telegram MarkdownV2 special characters with a preceding backslash."""
    return _MDV2_ESCAPE_RE.sub(r"\\\1", text)


def _strip_mdv2(text: str) -> str:
    """Remove MarkdownV2 escape backslashes and formatting markers.

    This is used as the fallback plain text when Telegram rejects a MarkdownV2
    payload, producing a readable string without stray syntax characters.
    """
    cleaned = re.sub(r"\\([_*\[\]()~`>#+\-=|{}.!\\])", r"\1", text)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"~([^~]+)~", r"\1", cleaned)
    cleaned = re.sub(r"\|\|([^|]+)\|\|", r"\1", cleaned)
    return cleaned


# ─── GFM pipe-table → bullet groups ──────────────────────────────────────────


TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*){1,}\|?\s*$")


def _split_table_row(row: str) -> list[str]:
    """Split a GFM table row into stripped cell values."""
    s = row.strip()
    s = s.removeprefix("|")
    s = s.removesuffix("|")
    return [c.strip() for c in s.split("|")]


def _is_table_row(line: str) -> bool:
    """Return True if *line* could plausibly be a table data row."""
    stripped = line.strip()
    return bool(stripped) and "|" in stripped


def _render_table_block(table_block: list[str]) -> str:
    """Render a detected GFM table as bold-heading + bullet groups."""
    if len(table_block) < 3:
        return "\n".join(table_block)

    headers = _split_table_row(table_block[0])
    if len(headers) < 2:
        return "\n".join(table_block)

    first_data = _split_table_row(table_block[2]) if len(table_block) > 2 else []
    has_row_label_col = len(first_data) == len(headers) + 1

    rendered_groups: list[str] = []
    for index, row in enumerate(table_block[2:], start=1):
        cells = _split_table_row(row)
        if has_row_label_col:
            heading = cells[0] if cells and cells[0] else f"Row {index}"
            data_cells = cells[1:]
        else:
            heading = (
                cells[0]
                if cells and cells[0]
                else next((cell for cell in cells if cell), f"Row {index}")
            )
            data_cells = cells

        if len(data_cells) < len(headers):
            data_cells.extend([""] * (len(headers) - len(data_cells)))
        elif len(data_cells) > len(headers):
            data_cells = data_cells[: len(headers)]

        bullets: list[str] = []
        for header, value in zip(headers, data_cells):
            if not has_row_label_col and value == heading:
                continue
            bullets.append(f"• {header}: {value}")

        group_lines = [f"**{heading}**", *bullets]
        rendered_groups.append("\n".join(group_lines))

    return "\n\n".join(rendered_groups)


def convert_table_to_bullets(text: str) -> str:
    """Rewrite GFM pipe tables into bold-heading + bullet groups.

    Tables inside fenced code blocks are left alone.
    """
    if "|" not in text or "-" not in text:
        return text

    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and TABLE_SEPARATOR_RE.match(lines[i + 1]):
            table_block = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and _is_table_row(lines[j]):
                table_block.append(lines[j])
                j += 1
            out.append(_render_table_block(table_block))
            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


# ─── Citation tags → source notes ────────────────────────────────────────────


def _replace_citations(text: str) -> str:
    """Replace <ref_file /> and <ref_snippet /> tags with short source notes."""

    def _snippet_repl(m: re.Match) -> str:
        path = m.group(1)
        lines = m.group(2)
        if lines:
            return f"Source: {path} (lines {lines})"
        return f"Source: {path}"

    text = _REF_FILE_RE.sub(r"Source: \1", text)
    text = _REF_SNIPPET_RE.sub(_snippet_repl, text)
    return text


# ─── Markdown → Telegram MarkdownV2 ──────────────────────────────────────────


def format_markdown_v2(content: str) -> str:
    """Convert standard markdown to Telegram MarkdownV2 format."""
    if not content:
        return content

    placeholders: dict[str, str] = {}
    counter = [0]

    def _ph(value: str) -> str:
        """Stash *value* behind a placeholder token that survives escaping."""
        key = f"\x00PH{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = value
        return key

    text = convert_table_to_bullets(content)

    # 1) Protect fenced code blocks (``` ... ```). Inside them \ and ` must
    # be escaped per the MarkdownV2 spec.
    def _protect_fenced(m: re.Match) -> str:
        raw = m.group(0)
        if "\n" in raw[3:]:
            open_end = raw.index("\n") + 1
        else:
            open_end = 3
        opening = raw[:open_end]
        body_and_close = raw[open_end:]
        body = body_and_close[:-3]
        body = body.replace("\\", "\\\\").replace("`", "\\`")
        return _ph(opening + body + "```")

    text = re.sub(
        r"(```(?:[^\n]*\n)?[\s\S]*?```)",
        _protect_fenced,
        text,
    )

    # 2) Protect inline code spans. Telegram MarkdownV2 uses single backticks
    # for code; to include a backtick in the content, escape it with \.
    def _protect_inline(m: re.Match) -> str:
        inner = m.group(2)
        # Standard Markdown trims a single leading/trailing space when both are present.
        if inner.startswith(" ") and inner.endswith(" ") and len(inner) > 2:
            inner = inner[1:-1]
        # Telegram inline code: single backticks, escape \ and ` inside.
        escaped = inner.replace("\\", "\\\\").replace("`", "\\`")
        return _ph(f"`{escaped}`")

    text = re.sub(
        r"(?<!`)(`+)(.+?)\1(?!`)",
        _protect_inline,
        text,
    )

    # 3) Convert citation tags to short source notes.
    text = _replace_citations(text)

    # 4) Convert markdown links. Escape display text; inside the URL only )
    # and \ need escaping.
    def _convert_link(m: re.Match) -> str:
        display = _escape_mdv2(m.group(1))
        url = m.group(2).replace("\\", "\\\\").replace(")", "\\)")
        return _ph(f"[{display}]({url})")

    text = re.sub(
        r"\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
        _convert_link,
        text,
    )

    # 5) Convert headers (## Title) → *Title* (MarkdownV2 bold).
    def _convert_header(m: re.Match) -> str:
        inner = m.group(1).strip()
        inner = re.sub(r"\*\*(.+?)\*\*", r"\1", inner)
        return _ph(f"*{_escape_mdv2(inner)}*")

    text = re.sub(
        r"^#{1,6}\s+(.+)$",
        _convert_header,
        text,
        flags=re.MULTILINE,
    )

    # 6) Convert bold: **text** and __text__ → *text* (MarkdownV2 bold).
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: _ph(f"*{_escape_mdv2(m.group(1))}*"),
        text,
    )
    text = re.sub(
        r"(?<!\w)__(.+?)__(?!\w)",
        lambda m: _ph(f"*{_escape_mdv2(m.group(1))}*"),
        text,
    )

    # 7) Convert italic: *text* and _text_ → _text_ (MarkdownV2 italic).
    text = re.sub(
        r"\*([^*\n]+)\*",
        lambda m: _ph(f"_{_escape_mdv2(m.group(1))}_"),
        text,
    )
    text = re.sub(
        r"(?<!\w)_([^_\n]+)_(?!\w)",
        lambda m: _ph(f"_{_escape_mdv2(m.group(1))}_"),
        text,
    )

    # 8) Convert strikethrough and spoiler.
    text = re.sub(
        r"~~(.+?)~~",
        lambda m: _ph(f"~{_escape_mdv2(m.group(1))}~"),
        text,
    )
    text = re.sub(
        r"\|\|(.+?)\|\|",
        lambda m: _ph(f"||{_escape_mdv2(m.group(1))}||"),
        text,
    )

    # 9) Convert blockquotes. Support expandable quotes (**> ... ||).
    def _convert_blockquote(m: re.Match) -> str:
        prefix = m.group(1)
        body = m.group(2)
        if prefix.startswith("**") and body.endswith("||"):
            return _ph(f"{prefix} {_escape_mdv2(body[:-2])}||")
        return _ph(f"{prefix} {_escape_mdv2(body)}")

    text = re.sub(
        r"^((?:\*\*)?>{1,3}) (.+)$",
        _convert_blockquote,
        text,
        flags=re.MULTILINE,
    )

    # 10) Escape remaining special characters in plain text.
    text = _escape_mdv2(text)

    # 11) Restore placeholders in reverse insertion order so nested references
    # resolve correctly.
    for key in reversed(list(placeholders.keys())):
        text = text.replace(key, placeholders[key])

    # 12) Safety net: escape bare ( ) { } outside code spans/blocks.
    _code_split = re.split(r"(```[\s\S]*?```|`[^`]+`)", text)
    _safe_parts: list[str] = []
    for idx, seg in enumerate(_code_split):
        if idx % 2 == 1:
            _safe_parts.append(seg)
            continue

        def _esc_bare(m: re.Match, _seg: str = seg) -> str:
            s = m.start()
            ch = m.group(0)
            if s > 0 and _seg[s - 1] == "\\":
                return ch
            if ch == "(" and s > 0 and _seg[s - 1] == "]":
                return ch
            if ch == ")":
                before = _seg[:s]
                if "](http" in before or "](" in before:
                    depth = 0
                    for j in range(s - 1, max(s - 2000, -1), -1):
                        if _seg[j] == "(":
                            depth -= 1
                            if depth < 0:
                                if j > 0 and _seg[j - 1] == "]":
                                    return ch
                                break
                        elif _seg[j] == ")":
                            depth += 1
            return "\\" + ch

        _safe_parts.append(re.sub(r"[(){}]", _esc_bare, seg))

    text = "".join(_safe_parts)
    return text


# ─── Code-fence-aware text splitting ─────────────────────────────────────────


def split_telegram_text(
    content: str,
    max_length: int = 4096,
    len_fn: Callable[[str], int] | None = None,
    reserve: int = 12,
) -> list[str]:
    """Split a long message into chunks, preserving code block boundaries.

    When a split falls inside a triple-backtick code block, the fence is closed
    at the end of the current chunk and reopened (with the original language
    tag) at the start of the next chunk. Splits inside an inline `` `...` ``
    span are avoided by backtracking before the last unescaped backtick.

    The *reserve* argument leaves room for a chunk indicator such as `` (1/3)``
    that the caller may append to each chunk.
    """
    _len = len_fn or utf16_len

    if _len(content) <= max_length:
        return [content]

    FENCE_CLOSE = "\n```"
    chunks: list[str] = []
    remaining = content
    carry_lang: str | None = None

    def _code_state(snippet: str, starting_in_code: bool, starting_lang: str) -> tuple[bool, str]:
        in_code = starting_in_code
        lang = starting_lang
        for line in snippet.split("\n"):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    in_code = False
                    lang = ""
                else:
                    in_code = True
                    tag = stripped[3:].strip()
                    lang = tag.split()[0] if tag else ""
        return in_code, lang

    while remaining:
        prefix = f"```{carry_lang}\n" if carry_lang is not None else ""
        _final_in_code, _final_lang = _code_state(
            remaining, carry_lang is not None, carry_lang or ""
        )

        # Does the rest fit in one final chunk (leaving room for a marker and
        # a closing fence if we are inside a code block)?
        final_extra = _len(FENCE_CLOSE) if _final_in_code else 0
        if _len(prefix) + _len(remaining) + final_extra <= max_length - reserve:
            final_chunk = prefix + remaining
            if _final_in_code:
                in_code, _ = _code_state(remaining, True, _final_lang)
                if in_code:
                    final_chunk += FENCE_CLOSE
            chunks.append(final_chunk)
            break

        headroom = max_length - reserve - _len(prefix) - _len(FENCE_CLOSE)
        if headroom < 1:
            headroom = max(1, max_length // 2)

        if _len is len:
            cp_limit = min(headroom, len(remaining))
        else:
            cp_limit = _safe_slice_budget(remaining, headroom, _len)

        region = remaining[:cp_limit]
        split_at = region.rfind("\n")
        if split_at < cp_limit // 2:
            split_at = region.rfind(" ")
        if split_at < 1:
            split_at = cp_limit
        else:
            # Include the delimiter in the first chunk so the next chunk starts
            # cleanly (preserves indentation for code blocks).
            split_at += 1

        # Avoid splitting inside an inline code span (`...`). If the candidate
        # contains a fenced code block, the fence-handling below will close and
        # reopen it; in that case the odd backtick count comes from a fence,
        # not an inline span, so we should not backtrack here.
        candidate = remaining[:split_at]
        has_fenced_block = bool(re.search(r"(?m)^```", candidate))
        if not has_fenced_block:
            unescaped_backticks = candidate.count("`") - candidate.count("\\`")
            if unescaped_backticks % 2 == 1:
                last_bt = candidate.rfind("`")
                while last_bt > 0 and candidate[last_bt - 1] == "\\":
                    last_bt = candidate.rfind("`", 0, last_bt)
                if last_bt > 0:
                    safe_split = max(
                        candidate.rfind(" ", 0, last_bt),
                        candidate.rfind("\n", 0, last_bt),
                    )
                    if safe_split > cp_limit // 4:
                        # Keep the delimiter with the first chunk.
                        split_at = safe_split + 1 if candidate[safe_split] != "\\" else safe_split
                    else:
                        split_at = last_bt
                else:
                    split_at = max(1, cp_limit)

        chunk_body = remaining[:split_at]
        remaining = remaining[split_at:]

        full_chunk = prefix + chunk_body

        in_code, lang = _code_state(chunk_body, carry_lang is not None, carry_lang or "")
        if in_code:
            full_chunk += FENCE_CLOSE
            carry_lang = lang
        else:
            carry_lang = None

        chunks.append(full_chunk)

    return chunks


# ─── Chunk indicator / fence separation ──────────────────────────────────────


_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(r"(?m)^``` (?P<indicator>(?:\\)?\(\d+/\d+(?:\\)?\))$")


def separate_chunk_indicator_from_fence(text: str) -> str:
    """Move ``(N/M)`` chunk markers off Telegram code-fence lines.

    When a chunk had to close an in-progress fenced code block and a caller
    then appends a chunk indicator, the result can be a line like
    `` ``` (1/2)`` (or the MarkdownV2-escaped equivalent). Telegram does not
    treat that as a clean closing fence, so put the indicator on its own line
    immediately after the closing fence.
    """
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(r"```\n\g<indicator>", text)
