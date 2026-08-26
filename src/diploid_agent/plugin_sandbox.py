"""CLI that validates a plugin module in an isolated process."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from diploid_agent.config import PluginConfig
from diploid_agent.engine.base import TurnRequest, TurnResult
from diploid_agent.models import SessionRecord
from diploid_agent.plugins.base import StatePlugin, TurnInfo
from diploid_agent.plugins.contexts import (
    EngineCallContext,
    EngineResultContext,
    PromptBuildContext,
    RecordTurnContext,
    TurnStartContext,
)
from diploid_agent.testing.fake_runtime import FakePluginRuntime

_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _dump(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False))


def _validate_module(module: str) -> None:
    if not _MODULE_NAME_RE.match(module) or ".." in module:
        raise ValueError(f"Invalid or unsafe plugin module name: {module}")
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None or spec.origin in ("built-in", "frozen"):
        raise ImportError(f"Plugin module {module} cannot be loaded or is not a file")
    mod = importlib.import_module(module)
    if not hasattr(mod, "Plugin"):
        raise ImportError(f"Plugin module {module} must expose a 'Plugin' class")


def _build_config(args: argparse.Namespace) -> PluginConfig:
    data: dict[str, Any] = {"name": args.name or "sandbox"}
    if args.state_file:
        data["state_file"] = args.state_file
    if args.prompt_slot:
        data["prompt_slot"] = args.prompt_slot
    if args.config_json:
        data["config"] = json.loads(args.config_json)
    return PluginConfig(**data)


def _exercise_plugin(plugin: StatePlugin, chat_id: str) -> None:
    """Run a candidate plugin through a synthetic, safe turn cycle."""
    record = SessionRecord(
        chat_id=chat_id,
        session_number=1,
        session_id="sandbox-session",
        model="swe-1-7",
        persona="sandbox",
        cwd=str(Path("/tmp")),
        created_at=0.0,
        updated_at=0.0,
        turn_number=1,
    )

    plugin.start()
    plugin.health()

    plugin.before_turn(
        TurnStartContext(
            chat_id=chat_id,
            user_message="hello",
            model="swe-1-7",
            record=None,
            now=0.0,
        )
    )

    plugin.before_build_prompt(
        PromptBuildContext(
            chat_id=chat_id,
            record=None,
            model="swe-1-7",
            is_first=False,
        )
    )

    plugin.before_engine_call(
        EngineCallContext(
            chat_id=chat_id,
            request=TurnRequest(prompt="hello", cwd=Path("/tmp")),
            session_id="sandbox-session",
            record=None,
        )
    )

    plugin.after_engine_call(
        EngineResultContext(
            chat_id=chat_id,
            record=None,
            result=TurnResult(reply="ok", session_id="sandbox-session"),
            reply="ok",
        )
    )

    plugin.before_record_turn(
        RecordTurnContext(
            chat_id=chat_id,
            record=record,
            turn_number=1,
            reply="ok",
        )
    )

    plugin.after_turn(
        TurnInfo(
            chat_id=chat_id,
            session_id="sandbox-session",
            session_number=1,
            turn_number=1,
            updated_at=0.0,
            last_stop_reason="completed",
            user_message="hello",
            reply="ok",
        )
    )

    plugin.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a plugin in isolation.")
    parser.add_argument("--module", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--chat-id", default="sandbox")
    parser.add_argument("--sessions-root", default=None)
    parser.add_argument("--state-file", default=None)
    parser.add_argument("--prompt-slot", default="persona_state")
    parser.add_argument("--plugin-path", action="append", default=[])
    parser.add_argument("--config-json", default=None)
    args = parser.parse_args(argv)

    for p in args.plugin_path:
        if p and str(p) not in sys.path:
            sys.path.append(str(p))

    sessions_root = Path(args.sessions_root) if args.sessions_root else Path(tempfile.mkdtemp())
    sessions_root.mkdir(parents=True, exist_ok=True)
    chat_dir = sessions_root / args.chat_id.replace("/", "_")
    chat_dir.mkdir(parents=True, exist_ok=True)

    try:
        _validate_module(args.module)
        config = _build_config(args)
        mod = importlib.import_module(args.module)
        fake_runtime = FakePluginRuntime(sessions_root=sessions_root, chat_id=args.chat_id)
        plugin: StatePlugin = mod.Plugin(config, args.chat_id, sessions_root, runtime=fake_runtime)

        _exercise_plugin(plugin, args.chat_id)

        _dump(
            {
                "ok": True,
                "module": args.module,
                "name": config.name,
                "checks": [
                    "start",
                    "health",
                    "before_turn",
                    "before_build_prompt",
                    "before_engine_call",
                    "after_engine_call",
                    "before_record_turn",
                    "after_turn",
                    "stop",
                ],
            }
        )
        return 0
    except Exception:  # noqa: BLE001
        _dump(
            {
                "ok": False,
                "module": args.module,
                "error": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
