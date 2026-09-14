"""Speech-to-text for inbound Telegram attachments.

Runs on the turn worker inside the poller (``_ingest_attachments``), never on
the poll loop. Providers are config-selected; ``none`` disables transcription
entirely. A provider failure returns ``None`` — the caller annotates the
message instead of dropping it.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any

from diploid_agent.config import TelegramConfig

logger = logging.getLogger("telegram_poll")

# Attachment kinds worth transcribing. ``video`` is deliberately excluded:
# long-form video is too expensive to run through a CPU whisper for a maybe.
TRANSCRIBABLE_KINDS = frozenset({"voice", "audio", "video_note"})

_COMMAND_TIMEOUT = 60.0

_model_cache: dict[str, Any] = {}
_model_cache_lock = threading.Lock()


def _transcribe_command(path: Path, command: str) -> str | None:
    """Run ``command <file>`` and return its stdout as the transcript."""
    if not command.strip():
        logger.warning("stt_provider=command but stt_command is empty")
        return None
    try:
        proc = subprocess.run(
            [*shlex.split(command), str(path)],
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT,
            check=False,
        )
    except Exception:
        logger.exception("stt command failed for %s", path)
        return None
    if proc.returncode != 0:
        logger.warning(
            "stt command exited %s for %s: %s",
            proc.returncode,
            path,
            proc.stderr.strip()[:200],
        )
        return None
    transcript = proc.stdout.strip()
    return transcript or None


def _transcribe_faster_whisper(path: Path, model_name: str) -> str | None:
    """Transcribe with faster-whisper, keeping one loaded model per size."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.warning(
            "stt_provider=faster-whisper but faster-whisper is not installed in the poller env"
        )
        return None
    with _model_cache_lock:
        model = _model_cache.get(model_name)
        if model is None:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            _model_cache[model_name] = model
    segments, _info = model.transcribe(str(path))
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text or None


def transcribe(path: Path, config: TelegramConfig) -> str | None:
    """Return a transcript for a downloaded attachment, or None."""
    provider = config.stt_provider
    if provider == "none":
        return None
    if provider == "command":
        return _transcribe_command(path, config.stt_command)
    if provider == "faster-whisper":
        return _transcribe_faster_whisper(path, config.stt_model)
    logger.warning("Unknown stt_provider %r — expected none/command/faster-whisper", provider)
    return None
