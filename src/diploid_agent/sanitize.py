"""Lone-surrogate scrubbing for text that must survive UTF-8 encoding.

A lone UTF-16 surrogate (U+D800–U+DFFF) can exist in a Python ``str`` but can
never be encoded as UTF-8. On Windows the common source is a locale-codec
``surrogateescape`` decode (e.g. an MCP child reading stdin under GBK): the
surrogate then rides transcripts, wake payloads, and wire envelopes into every
later ``json.dumps(...).encode("utf-8")`` and wedges it. Scrubbing to U+FFFD at
record/encode time keeps a malformed byte lossy-but-valid instead of fatal.
"""

from __future__ import annotations

import re
from typing import Any

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def sanitize_surrogates(s: str) -> str:
    """Replace lone UTF-16 surrogates with U+FFFD; valid text is untouched."""
    return _SURROGATE_RE.sub("\ufffd", s)


def sanitize_json_value(value: Any) -> Any:
    """Recursively scrub lone surrogates from every string in a JSON value."""
    if isinstance(value, str):
        return sanitize_surrogates(value)
    if isinstance(value, list):
        return [sanitize_json_value(v) for v in value]
    if isinstance(value, dict):
        return {sanitize_json_value(k): sanitize_json_value(v) for k, v in value.items()}
    return value
