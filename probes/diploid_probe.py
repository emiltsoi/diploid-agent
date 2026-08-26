#!/usr/bin/env python3
"""ACP probe: verify a headless ACP binary works with on-disk or env credentials.

This spawns an ACP agent binary (default `devin acp`), drives it through a raw
ACP v1 JSON-RPC exchange, and prints the streamed reply. It proves the auth
issue is environmental: the ACP process needs either `WINDSURF_API_KEY`,
`ACP_API_KEY`, or the `~/.local/share/devin/credentials.toml` file that
`devin auth login` writes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any


def _load_api_key() -> str | None:
    """Return API key from env or the Devin CLI credentials file."""
    if os.environ.get("WINDSURF_API_KEY"):
        return os.environ["WINDSURF_API_KEY"]
    if os.environ.get("ACP_API_KEY"):
        return os.environ["ACP_API_KEY"]

    creds_path = Path.home() / ".local" / "share" / "devin" / "credentials.toml"
    if creds_path.exists():
        try:
            data = tomllib.loads(creds_path.read_text())
            return data.get("windsurf_api_key")
        except (OSError, ValueError):
            return None
    return None


class AcpTransport:
    """Minimal async ACP v1 client over stdio."""

    def __init__(self, agent_bin: Path, model: str, api_key: str, cwd: Path):
        self.agent_bin = agent_bin
        self.model = model
        self.api_key = api_key
        self.cwd = cwd
        self.proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None

    async def start(self) -> dict[str, Any]:
        env = os.environ.copy()
        env["WINDSURF_API_KEY"] = self.api_key
        env["ACP_API_KEY"] = self.api_key

        self.proc = await asyncio.create_subprocess_exec(
            str(self.agent_bin),
            "acp",
            "--model",
            self.model,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )

        self._reader_task = asyncio.create_task(self._reader())

        init = await self._call(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {
                    "name": "diploid-agent/acp-probe",
                    "version": "0.1.0",
                },
            },
        )
        return init

    async def new_session(self) -> str:
        resp = await self._call(
            "session/new",
            {"cwd": str(self.cwd), "mcpServers": []},
        )
        return resp["sessionId"]

    async def set_config_option(self, session_id: str, config_id: str, value: Any) -> None:
        await self._call(
            "session/set_config_option",
            {"sessionId": session_id, "configId": config_id, "value": value},
        )

    async def prompt(self, session_id: str, text: str, timeout: float = 120.0) -> dict[str, Any]:
        """Send a prompt and return (stop_reason, reply_text, usage)."""
        reply_chunks: list[str] = []

        # We need a custom handler for the prompt turn because the response to
        # the `session/prompt` request only contains the stop reason; the actual
        # assistant text arrives as `session/update` notifications.
        prompt_id = self._next_id + 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[prompt_id] = future

        def handler(msg: dict[str, Any]) -> bool:
            if msg.get("method") != "session/update":
                return False
            update = msg.get("params", {}).get("update", {})
            kind = update.get("sessionUpdate")
            if kind == "agent_message_chunk":
                content = update.get("content", {})
                if content.get("type") == "text":
                    reply_chunks.append(content["text"])
                    print(content["text"], end="", flush=True)
            return False

        self._message_handler = handler

        try:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": prompt_id,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": text}],
                    },
                }
            )
            msg = await asyncio.wait_for(future, timeout=timeout)
            if "error" in msg:
                raise RuntimeError(f"session/prompt failed: {msg['error']}")
            result = msg.get("result", {})
        finally:
            self._message_handler = None

        return {
            "stop_reason": result.get("stopReason"),
            "reply": "".join(reply_chunks),
            "usage": result.get("usage"),
        }

    async def close(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5.0)
            except TimeoutError:
                self.proc.kill()

    # ---------------------------------------------------------------- internal

    _message_handler: Any | None = None

    async def _call(self, method: str, params: Any) -> Any:
        self._next_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
            "params": params,
        }
        future = asyncio.get_running_loop().create_future()
        self._pending[self._next_id] = future
        await self._send(msg)
        resp = await future
        if "error" in resp:
            raise RuntimeError(f"ACP error on {method}: {resp['error']}")
        return resp["result"]

    async def _send(self, msg: dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("ACP process not started")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode()
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def _respond(self, req_id: int, result: Any) -> None:
        await self._send({"jsonrpc": "2.0", "id": req_id, "result": result})

    async def _reader(self) -> None:
        while True:
            assert self.proc is not None and self.proc.stdout is not None
            try:
                line = await self.proc.stdout.readline()
            except (OSError, ValueError):
                break
            if not line:
                break

            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                continue

            # Agent-to-client request (e.g. permission prompt).
            if "method" in msg and "id" in msg:
                await self._handle_request(msg)
                continue

            # Notification.
            if "id" not in msg:
                if self._message_handler is not None:
                    self._message_handler(msg)
                continue

            # Response to one of our calls.
            future = self._pending.pop(msg["id"], None)
            if future is not None and not future.done():
                future.set_result(msg)

    async def _handle_request(self, msg: dict[str, Any]) -> None:
        method = msg["method"]
        req_id = msg["id"]
        params = msg.get("params", {})

        if method == "session/request_permission":
            options = params.get("options", [])
            option_id = options[0]["optionId"] if options else "allow"
            await self._respond(req_id, {"outcome": {"outcome": "selected", "optionId": option_id}})
        elif method == "fs/read_text_file":
            # Only called if we advertise fs.readTextFile=True.
            try:
                text = Path(params["path"]).read_text()
            except (OSError, TypeError, ValueError) as exc:
                await self._respond(req_id, {"error": {"code": -32000, "message": str(exc)}})
            else:
                await self._respond(req_id, {"content": text})
        elif method == "fs/write_text_file":
            try:
                Path(params["path"]).write_text(params["content"])
            except (OSError, TypeError, ValueError) as exc:
                await self._respond(req_id, {"error": {"code": -32000, "message": str(exc)}})
            else:
                await self._respond(req_id, None)
        else:
            await self._respond(
                req_id,
                {"error": {"code": -32601, "message": f"Method not found: {method}"}},
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="ACP auth and steering probe")
    default_agent_bin = shutil.which("devin") or str(Path.home() / ".local" / "bin" / "devin")
    parser.add_argument(
        "--engine-bin",
        type=Path,
        default=Path(default_agent_bin),
        help="Path to the ACP agent binary (default: devin)",
    )
    parser.add_argument("--devin-bin", type=Path, dest="engine_bin")
    parser.add_argument("--model", default="swe-1-7")
    parser.add_argument(
        "--mode", default="bypass", help="Session mode (bypass, accept-edits, ask, plan, smart)"
    )
    args = parser.parse_args()

    if not args.engine_bin.exists():
        print(f"Agent binary not found: {args.engine_bin}", file=sys.stderr)
        return 1

    api_key = _load_api_key()
    if not api_key:
        print(
            "No WINDSURF_API_KEY/ACP_API_KEY in environment and no ~/.local/share/devin/credentials.toml.\n"
            "Run `devin auth login` from this environment, or set WINDSURF_API_KEY/ACP_API_KEY.",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory(prefix="acp-probe-") as tmpdir:
        cwd = Path(tmpdir)

        async def _run() -> int:
            transport = AcpTransport(args.engine_bin, args.model, api_key, cwd)
            try:
                init = await transport.start()
                print(
                    f"Connected to {init['agentInfo']['title']} (protocol v{init['protocolVersion']})",
                    file=sys.stderr,
                )

                session_id = await transport.new_session()
                print(f"Session: {session_id}", file=sys.stderr)

                await transport.set_config_option(session_id, "mode", args.mode)
                print(f"Mode set to {args.mode}", file=sys.stderr)

                await transport.set_config_option(session_id, "model", args.model)
                print(f"Model set to {args.model}", file=sys.stderr)

                result = await transport.prompt(
                    session_id,
                    "Introduce yourself in one short sentence, then answer: what is 7×8?",
                )
                print("\n", file=sys.stdout)  # newline after streamed reply
                print(f"\nStop reason: {result['stop_reason']}", file=sys.stderr)
                print(f"Usage: {result['usage']}", file=sys.stderr)
            finally:
                await transport.close()
            return 0

        try:
            return asyncio.run(_run())
        except (OSError, RuntimeError, LookupError, TypeError, ValueError) as exc:
            print(f"ACP probe failed: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
