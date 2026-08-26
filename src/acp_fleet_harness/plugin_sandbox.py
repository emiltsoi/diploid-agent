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

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.plugins.base import StatePlugin

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
        plugin: StatePlugin = mod.Plugin(config, args.chat_id, sessions_root, runtime=None)
        plugin.start()
        plugin.stop()
        _dump({"ok": True, "module": args.module, "name": config.name})
        return 0
    except Exception:  # noqa: BLE001
        _dump({
            "ok": False,
            "module": args.module,
            "error": traceback.format_exc(),
        })
        return 1


if __name__ == "__main__":
    sys.exit(main())
