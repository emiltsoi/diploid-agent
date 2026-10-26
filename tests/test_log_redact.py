"""LogRecord-factory redaction of Telegram bot tokens."""

import logging

import diploid_agent  # noqa: F401  (installs the record factory at import)
from diploid_agent._redact import _scrub

TOKEN_URL = "https://api.telegram.org/bot123456789:AAFaKE_t0ken-Str1ng_123456789/sendMessage"


def test_scrub_strips_bot_token() -> None:
    assert _scrub(f"GET {TOKEN_URL} 400") == (
        "GET https://api.telegram.org/bot<redacted>/sendMessage 400"
    )
    assert _scrub("no token here") == "no token here"


def test_record_msg_and_args_scrubbed(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="test_redact"):
        logging.getLogger("test_redact").error("send failed for %s", TOKEN_URL)
    record = caplog.records[-1]
    assert "AAFaKE_t0ken" not in record.getMessage()
    assert "bot<redacted>" in record.getMessage()


def test_record_exc_info_scrubbed(caplog) -> None:
    try:
        raise RuntimeError(f"Client error '400' for url '{TOKEN_URL}'")
    except RuntimeError:
        with caplog.at_level(logging.ERROR, logger="test_redact"):
            logging.getLogger("test_redact").exception("send failed")
    record = caplog.records[-1]
    # exc_info was pre-rendered into exc_text with the token scrubbed.
    assert record.exc_info is None
    assert record.exc_text is not None
    assert "AAFaKE_t0ken" not in record.exc_text
    assert "bot<redacted>" in record.exc_text


def test_record_nested_arg_scrubbed(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="test_redact"):
        logging.getLogger("test_redact").error("send failed: %s", {"url": TOKEN_URL})
    message = caplog.records[-1].getMessage()
    assert "AAFaKE_t0ken" not in message
    assert "bot<redacted>" in message


def test_record_mapping_args_scrubbed(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="test_redact"):
        logging.getLogger("test_redact").error("send failed for %(url)s", {"url": TOKEN_URL})
    message = caplog.records[-1].getMessage()
    assert "AAFaKE_t0ken" not in message


def test_records_without_token_untouched(caplog) -> None:
    try:
        raise ValueError("plain failure")
    except ValueError:
        with caplog.at_level(logging.ERROR, logger="test_redact"):
            logging.getLogger("test_redact").exception("no secrets %s", "x")
    record = caplog.records[-1]
    assert record.exc_info is not None  # still a live traceback
