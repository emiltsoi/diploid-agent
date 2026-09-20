"""Capture real ACP ``session/update`` payloads into a fixture file.

Runs a real ``devin acp`` child through ``AcpClient`` and writes every
``on_update`` payload as JSONL to ``tests/fixtures/acp_updates.jsonl``.
Tests replay that file into ``TurnStream`` so fixtures model the actual
wire shape instead of guesses.

Manual tool — not collected by pytest. Requires ``devin`` on PATH and
credentials (WINDSURF_API_KEY / DEVIN_API_KEY / ~/.codeium).

Usage:
    .venv/bin/python tests/capture_acp_updates.py [fixture_path]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from diploid_agent.acp_client.client import AcpClient

FIXTURE = Path(__file__).parent / "fixtures" / "acp_updates.jsonl"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE
    out.parent.mkdir(parents=True, exist_ok=True)

    updates: list[dict] = []
    client = AcpClient(timeout=240.0)
    workdir = tempfile.mkdtemp(prefix="acp-capture-")
    Path(workdir, "marker.txt").write_text("hello fixture\n")

    try:
        result = client.create_session(
            "Run `cat marker.txt` and `ls` in the current directory, "
            "then edit marker.txt to say 'updated'.",
            cwd=workdir,
            on_update=updates.append,
            soft_timeout=180.0,
            timeout=240.0,
        )
    finally:
        client.close()

    with out.open("w") as f:
        for u in updates:
            f.write(json.dumps(u) + "\n")

    kinds = [u.get("sessionUpdate") for u in updates]
    print(f"stop_reason={result.stop_reason} updates={len(updates)}")
    print(f"kinds: {sorted({k for k in kinds if k})}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("DIPLOID_CONTROL_DIR", tempfile.mkdtemp(prefix="acp-cap-ctl-"))
    sys.exit(main())
