#!/usr/bin/env python3
"""Probe an in-place model switch on a live devin acp session.

Mirrors the production `/model X --in-place` path (turn/session.py
`_switch_model_in_place` -> `session/set_config_option`, configId="model")
against a throwaway session so a real switch attempt can be tested without
risking a live chat session.

Sequence: spawn `devin acp --model <flag>` (same flag the live child runs),
`session/new` (dumps the picker's accepted model list from configOptions),
pin the session to `--from-model` like the session-record re-apply does,
prompt once, then switch to `--to-model` and prompt again to prove the
session survived the switch.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diploid_probe import AcpTransport, _load_api_key


def _picker_models(session_result: dict) -> list[str]:
    """Same extraction as AcpSessionOps._extract_model_options."""
    for opt in session_result.get("configOptions", []):
        if opt.get("id") == "model":
            return [o["value"] for o in opt.get("options", [])]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="In-place model switch probe")
    parser.add_argument("--engine-bin", type=Path, default=Path(shutil.which("devin") or "devin"))
    parser.add_argument("--flag-model", default="swe-2-max", help="devin acp --model flag")
    parser.add_argument("--from-model", default="swe-2-high", help="model pinned at session start")
    parser.add_argument("--to-model", default="swe-2-max", help="in-place switch target")
    parser.add_argument("--mode", default="bypass")
    args = parser.parse_args()

    api_key = _load_api_key()
    if not api_key:
        print("No API key (WINDSURF_API_KEY or credentials.toml).", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="acp-model-switch-") as tmpdir:

        async def _run() -> int:
            t = AcpTransport(args.engine_bin, args.flag_model, api_key, Path(tmpdir))
            try:
                await t.start()
                created = await t._call("session/new", {"cwd": tmpdir, "mcpServers": []})
                sid = created["sessionId"]
                picker = _picker_models(created)
                print(f"[probe] session={sid}", file=sys.stderr)
                print(f"[probe] picker models ({len(picker)}): {picker}", file=sys.stderr)
                print(
                    f"[probe] to-model {args.to_model!r} in picker: {args.to_model in picker}",
                    file=sys.stderr,
                )

                await t.set_config_option(sid, "mode", args.mode)
                await t.set_config_option(sid, "model", args.from_model)
                print(f"[probe] pinned to {args.from_model}", file=sys.stderr)

                await t.prompt(sid, "Reply with exactly: alive")
                print("\n[probe] pre-switch prompt OK", file=sys.stderr)

                try:
                    await t.set_config_option(sid, "model", args.to_model)
                except RuntimeError as exc:
                    print(f"[probe] in-place switch REJECTED: {exc}", file=sys.stderr)
                else:
                    print(f"[probe] in-place switch ACCEPTED -> {args.to_model}", file=sys.stderr)

                try:
                    await t.prompt(sid, "Reply with exactly: still alive")
                except RuntimeError as exc:
                    print(f"[probe] post-switch prompt FAILED: {exc}", file=sys.stderr)
                    return 2
                print("\n[probe] post-switch prompt OK — session survived", file=sys.stderr)
            finally:
                await t.close()
            return 0

        try:
            return asyncio.run(_run())
        except (OSError, RuntimeError, LookupError, TypeError, ValueError) as exc:
            print(f"[probe] failed: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
