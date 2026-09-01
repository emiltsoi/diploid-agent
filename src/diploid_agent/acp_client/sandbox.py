"""Isolated HOME and fake systemctl setup for the ACP child."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEMCTL_WRAPPER = '''#!/usr/bin/env python3
"""Sandboxed systemctl/reboot/shutdown wrapper for the ACP child."""
import json
import os
import socket
import sys

DANGEROUS = {
    "start", "stop", "restart", "reload", "reload-or-restart",
    "try-restart", "poweroff", "reboot", "halt", "shutdown",
    "suspend", "hibernate", "hybrid-sleep", "default", "rescue", "emergency",
}

DEFAULT_COMMANDS = {
    "reboot": "reboot",
    "poweroff": "poweroff",
    "shutdown": "poweroff",
    "halt": "halt",
}


def _send_request(command: str, unit: str, reason: str) -> int:
    control_socket = os.environ.get("DIPLOID_CONTROL_SOCKET")
    service_name = os.environ.get("DIPLOID_SERVICE_NAME", "unknown.service")
    if not control_socket:
        if command in DANGEROUS:
            print(f"{os.path.basename(sys.argv[0])}: {command} {unit}: "
                  "not permitted in ACP sandbox", file=sys.stderr)
            return 1
        print(f"{os.path.basename(sys.argv[0])}: no control socket available", file=sys.stderr)
        return 0

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)
            s.connect(control_socket)
            payload = {
                "action": "restart_service" if command in ("restart", "reboot", "poweroff", "halt", "shutdown") else "service_command",
                "service": unit or service_name,
                "command": command,
                "reason": reason,
            }
            s.sendall(json.dumps(payload).encode("utf-8"))
            data = b""
            while True:
                chunk = s.recv(1024)
                if not chunk:
                    break
                data += chunk
            ack = json.loads(data.decode("utf-8")) if data else {}
            print(f"{command} {payload['service']} scheduled via harness ({ack.get('status', 'ok')}).")
            return 0
    except Exception as exc:
        print(f"{os.path.basename(sys.argv[0])}: failed to contact harness: {exc}", file=sys.stderr)
        return 1


def _main() -> int:
    exe = os.path.basename(sys.argv[0])
    if exe in DEFAULT_COMMANDS:
        unit = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DIPLOID_SERVICE_NAME", "unknown.service")
        return _send_request(DEFAULT_COMMANDS[exe], unit, " ".join(sys.argv[1:]))

    # systemctl [OPTIONS] COMMAND [UNIT...]
    args = sys.argv[1:]
    command = None
    units = []
    for a in args:
        if a.startswith("-"):
            continue
        if command is None:
            command = a
        else:
            units.append(a)

    if command in {"status", "is-active", "is-enabled", "is-failed", "show", "list-units", "cat"}:
        print(f"systemctl: {command}: sandboxed (no real service manager)")
        return 0

    if command is None:
        print("systemctl: no command given")
        return 1

    if command not in DANGEROUS:
        print(f"systemctl: command '{command}' is not supported in the ACP sandbox")
        return 1

    unit = units[0] if units else os.environ.get("DIPLOID_SERVICE_NAME", "unknown.service")
    return _send_request(command, unit, " ".join(sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(_main())
'''


class AcpSandbox:
    """Prepare and manage the isolated HOME for the ACP child."""

    def __init__(self, service_name: str | None = None) -> None:
        self._service_name = service_name
        self.devin_home: Path | None = None

    def prepare(self, mcp_servers: list[dict[str, Any]] | None = None) -> None:
        """Create an isolated HOME for the ACP child process and write configs.

        Devin's bundled MCP config (e.g. `~/.codeium/windsurf/mcp_config.json`)
        can include MCP servers that deadlock or saturate and block `devin acp`
        startup indefinitely. We create a sanitized home directory and write the
        active MCP server list into `mcp_config.json` before `devin acp` starts,
        because devin 3000.6.7+ loads servers from that file.
        """
        if self.devin_home is not None and self.devin_home.exists():
            # Re-sanitize the MCP config on every (re)start in case the previous
            # `devin` child wrote to it.
            self.write_mcp_configs(mcp_servers)
            return

        self.devin_home = Path(tempfile.mkdtemp(prefix="acp-home-"))
        config_dir = self.devin_home / ".config" / "devin"
        config_dir.mkdir(parents=True, exist_ok=True)
        codeium_dir = self.devin_home / ".codeium" / "windsurf"
        codeium_dir.mkdir(parents=True, exist_ok=True)

        # Sandbox the ACP child: create a private runtime dir and fake systemctl
        # wrapper so the model cannot run raw `systemctl --user restart` from the
        # child. Restart requests are routed through the harness control socket.
        (self.devin_home / ".run").mkdir(parents=True, exist_ok=True)
        self._write_sandbox_binaries()

        user_config = Path.home() / ".config" / "devin" / "config.json"
        if user_config.exists():
            try:
                (config_dir / "config.json").write_text(user_config.read_text())
            except OSError:
                logger.warning("Failed to copy devin config to ACP home")
        else:
            (config_dir / "config.json").write_text(
                json.dumps({"version": 1, "permissions": {"allow": ["*"]}})
            )

        self.write_mcp_configs(mcp_servers)

    def cleanup(self) -> None:
        """Remove the isolated HOME created for the ACP child."""
        home = self.devin_home
        self.devin_home = None
        if home is not None and home.exists():
            try:
                shutil.rmtree(home)
            except OSError:
                logger.warning("Failed to remove ACP home %s", home)

    def write_mcp_configs(
        self,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        """Write MCP server definitions into the isolated HOME.

        `devin acp` 3000.6.7+ loads MCP servers from `mcp_config.json` at
        process startup, so the isolated home must contain the active servers
        before the child is spawned.
        """
        if self.devin_home is None:
            return

        servers: dict[str, dict[str, Any]] = {}
        for server in self.normalize_mcp_servers(mcp_servers):
            name = str(server.get("name", ""))
            if not name or server.get("disabled"):
                continue
            if name in servers:
                logger.warning("Duplicate MCP server %s in active list; using last", name)
            entry: dict[str, Any] = {
                "command": server.get("command", "python"),
                "args": server.get("args", []),
            }
            if "cwd" in server:
                entry["cwd"] = server["cwd"]
            env = server.get("env", [])
            if isinstance(env, list):
                env_dict: dict[str, str] = {}
                for e in env:
                    if isinstance(e, str) and "=" in e:
                        key, value = e.split("=", 1)
                        env_dict[key] = value
                if env_dict:
                    entry["env"] = env_dict
            elif isinstance(env, dict) and env:
                entry["env"] = dict(env)
            if "instructions" in server:
                entry["instructions"] = server["instructions"]
            servers[name] = entry

        mcp_config = json.dumps({"mcpServers": servers}, indent=2)
        try:
            (self.devin_home / ".config" / "devin" / "mcp_config.json").write_text(mcp_config)
            (self.devin_home / ".codeium" / "windsurf" / "mcp_config.json").write_text(mcp_config)
        except OSError:
            logger.warning("Failed to write mcp_config.json")

    def _write_sandbox_binaries(self) -> None:
        """Install fake system control binaries into the isolated HOME.

        The ACP child inherits the real host `PATH` and can run
        `systemctl --user restart <service>` while in `bypass` mode. These
        wrappers intercept those calls and forward them to the harness control
        socket, where the restart can be scheduled gracefully.
        """
        if self.devin_home is None:
            return

        bin_dir = self.devin_home / ".local" / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)

        wrapper_path = bin_dir / "systemctl"
        wrapper_path.write_text(_SYSTEMCTL_WRAPPER)
        wrapper_path.chmod(0o755)

        for name in ("reboot", "poweroff", "shutdown", "halt"):
            target = bin_dir / name
            if target.exists() or target.is_symlink():
                try:
                    target.unlink()
                except OSError:
                    pass
            try:
                os.link(str(wrapper_path), str(target))
            except OSError:
                # Fall back to a copy if hard-linking across devices fails.
                target.write_text(_SYSTEMCTL_WRAPPER)
                target.chmod(0o755)

    def normalize_mcp_servers(
        self,
        mcp_servers: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Return the requested MCP servers with any `lean-ctx` entries dropped.

        `lean-ctx` has been removed from this setup because the shared daemon is
        a single point of failure and can hang `devin acp` startup. If a caller
        still passes it, strip it out and keep the other servers.

        The ACP `session/new` payload expects `env` as a map of strings, while
        the harness stores it as a list of `KEY=VALUE` strings. Convert any
        non-empty list to a dict before sending it to the ACP child.
        """
        if not mcp_servers:
            return []

        out: list[dict[str, Any]] = []

        for server in list(mcp_servers):
            name = str(server.get("name", ""))
            command = str(server.get("command", ""))
            if name == "lean-ctx" or Path(command).name == "lean-ctx":
                logger.warning(
                    "lean-ctx MCP server requested but is disabled in this setup; dropping"
                )
                continue
            server = dict(server)
            env = server.get("env")
            if isinstance(env, list):
                env_map: dict[str, str] = {}
                for entry in env:
                    if isinstance(entry, str) and "=" in entry:
                        key, value = entry.split("=", 1)
                        env_map[key] = value
                server["env"] = env_map
            out.append(server)

        return out

    def mcp_servers_key(self, mcp_servers: list[dict[str, Any]]) -> str:
        """Return a stable comparison key for a list of MCP server definitions."""
        simplified = [
            {
                "name": str(s.get("name", "")),
                "command": str(s.get("command", "")),
                "args": list(s.get("args", [])),
                "env": list(s.get("env", []))
                if isinstance(s.get("env"), list)
                else dict(s.get("env", {})),
            }
            for s in mcp_servers
        ]
        return json.dumps(simplified, sort_keys=True)
