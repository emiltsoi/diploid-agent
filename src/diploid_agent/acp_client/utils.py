"""ACP client utility helpers."""

from __future__ import annotations

import logging
import os
import shutil
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)


def _devin_credentials_paths() -> list[Path]:
    """Candidate locations for the Devin CLI credentials file."""
    paths = [Path.home() / ".local" / "share" / "devin" / "credentials.toml"]
    appdata = os.environ.get("APPDATA")
    if appdata:
        paths.append(Path(appdata) / "devin" / "credentials.toml")
    return paths


def _extract_windsurf_key(data: dict) -> str | None:
    """Pull the API key out of whichever credentials.toml schema is present."""
    if key := data.get("windsurf_api_key"):
        return key
    payload = data.get("payload")
    if isinstance(payload, dict):
        return payload.get("windsurf_api_key") or payload.get("api_key")
    return data.get("api_key")


def _load_windsurf_api_key() -> str | None:
    """Return the Windsurf API key from env or the Devin CLI credentials file."""
    if os.environ.get("WINDSURF_API_KEY"):
        return os.environ["WINDSURF_API_KEY"]

    for creds_path in _devin_credentials_paths():
        if creds_path.exists():
            try:
                data = tomllib.loads(creds_path.read_text())
            except (OSError, tomllib.TOMLDecodeError) as exc:
                logger.warning("Failed to read Devin credentials from %s: %s", creds_path, exc)
                continue
            if key := _extract_windsurf_key(data):
                return key
    return None


def _normalize_model(model: str) -> str:
    """Return the canonical ACP model id for an alias.

    The Devin CLI and docs use dotted aliases like `swe-1.7`, but ACP's
    `session/set_config_option` expects the dashed form `swe-1-7`.
    """
    return model.replace(".", "-")


def _devin_default_start_args(model: str) -> list[str]:
    """Return the default start arguments for the Devin ACP binary."""
    return ["acp", "--model", model]


def _resolve_agent_bin(agent_bin: str | Path) -> Path:
    """Resolve an agent binary path, falling back to PATH by file name."""
    p = Path(agent_bin).expanduser()
    if p.exists():
        return p
    name = p.name if p.name != "." else str(agent_bin)
    found = shutil.which(name)
    if found:
        return Path(found)
    raise RuntimeError(f"agent binary not found: {agent_bin}")


# Backward-compatible alias (deprecated).
_resolve_devin_bin = _resolve_agent_bin
