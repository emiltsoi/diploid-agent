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
import time
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


_TTS_TIMEOUT = 90.0

# Loaded PiperVoice instances (~60MB+ RSS each) are evicted after this idle
# window so a quiet bot does not pin the model resident forever.
_VOICE_IDLE_TTL = 300.0

_voice_cache: dict[str, tuple[Any, float]] = {}


def _synthesize_command(text: str, command: str, work_dir: Path) -> Path | None:
    """Run ``tts_command`` with the text on stdin; stdout must be audio bytes.

    Ogg/Opus output is sent as a Telegram voice note; anything else is sent as
    a plain audio file. The contract is deliberately tiny so a whisper.cpp-
    style binary, a piper wrapper, or a host-side speech bridge all fit.
    """
    if not command.strip():
        logger.warning("tts_provider=command but tts_command is empty")
        return None
    try:
        proc = subprocess.run(
            shlex.split(command),
            input=text.encode(),
            capture_output=True,
            timeout=_TTS_TIMEOUT,
            check=False,
        )
    except Exception:
        logger.exception("tts command failed")
        return None
    if proc.returncode != 0 or not proc.stdout:
        logger.warning(
            "tts command exited %s with %d bytes",
            proc.returncode,
            len(proc.stdout),
        )
        return None
    dest = work_dir / "say.ogg" if proc.stdout[:4] == b"OggS" else work_dir / "say.bin"
    dest.write_bytes(proc.stdout)
    return dest


def _synthesize_piper(text: str, model_path: str, work_dir: Path) -> Path | None:
    """Synthesize with piper (wav) then ffmpeg → ogg/opus for sendVoice."""
    if not model_path:
        logger.warning("tts_provider=piper but tts_model_path is empty")
        return None
    model_path = str(Path(model_path).expanduser())
    try:
        from piper import PiperVoice
    except ImportError:
        logger.warning("tts_provider=piper but piper-tts is not installed in the poller env")
        return None
    now = time.monotonic()
    with _model_cache_lock:
        for key, (_, last_used) in list(_voice_cache.items()):
            if now - last_used > _VOICE_IDLE_TTL:
                _voice_cache.pop(key, None)
        entry = _voice_cache.get(model_path)
        if entry is None:
            entry = (PiperVoice.load(model_path), now)
        else:
            entry = (entry[0], now)
        _voice_cache[model_path] = entry
        voice = entry[0]

    import wave

    wav_path = work_dir / "say.wav"
    try:
        # piper-tts >= 1.3: synthesize(text) yields AudioChunk objects;
        # older releases take a wave file handle as the second argument.
        # Only the call is shimmed — a TypeError raised mid-synthesis is a real
        # failure, not a signature mismatch, and must not double-run the old API.
        try:
            chunks = voice.synthesize(text)
        except TypeError:
            with wave.open(str(wav_path), "wb") as wav:
                voice.synthesize(text, wav)
        else:
            params_set = False
            with wave.open(str(wav_path), "wb") as wav:
                for chunk in chunks:
                    if not params_set:
                        wav.setnchannels(chunk.sample_channels)
                        wav.setsampwidth(chunk.sample_width)
                        wav.setframerate(chunk.sample_rate)
                        params_set = True
                    wav.writeframes(chunk.audio_int16_bytes)
    except Exception:
        logger.exception("piper synthesis failed")
        return None

    ogg_path = work_dir / "say.ogg"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet", "-i", str(wav_path), "-c:a", "libopus", str(ogg_path)],
            capture_output=True,
            timeout=_TTS_TIMEOUT,
            check=False,
        )
    except Exception:
        logger.exception("ffmpeg wav→ogg failed")
        return None
    if proc.returncode != 0 or not ogg_path.exists():
        logger.warning("ffmpeg exited %s: %s", proc.returncode, proc.stderr[-200:])
        return None
    return ogg_path


def synthesize(text: str, config: TelegramConfig, work_dir: Path) -> Path | None:
    """Synthesize ``text`` into an audio file under ``work_dir``, or None."""
    provider = config.tts_provider
    if provider == "none":
        return None
    if provider == "command":
        return _synthesize_command(text, config.tts_command, work_dir)
    if provider == "piper":
        return _synthesize_piper(text, config.tts_model_path, work_dir)
    logger.warning("Unknown tts_provider %r — expected none/command/piper", provider)
    return None


def synthesize_bounded(
    text: str,
    config: TelegramConfig,
    work_dir: Path,
    timeout: float = _TTS_TIMEOUT,
) -> Path | None:
    """synthesize() on a daemon thread with a join deadline.

    A wedged provider (e.g. a piper inference that never returns) surfaces as
    None after ``timeout`` — the daemon thread dies with the process rather
    than holding a send path forever.
    """
    outcome: list[Any] = [None, None]

    def _run() -> None:
        try:
            outcome[0] = synthesize(text, config, work_dir)
        except Exception as exc:  # noqa: BLE001
            outcome[1] = exc

    worker = threading.Thread(target=_run, daemon=True, name="tts-synth")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.warning("TTS synthesis exceeded %ss; falling back to text", timeout)
        return None
    if outcome[1] is not None:
        raise outcome[1]
