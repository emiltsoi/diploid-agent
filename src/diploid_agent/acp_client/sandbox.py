"""Isolated HOME and fake systemctl setup for the ACP subprocess."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from diploid_agent._compat import IS_WINDOWS

logger = logging.getLogger(__name__)

_SYSTEMCTL_WRAPPER = '''#!/usr/bin/env python3
"""Sandboxed systemctl/reboot/shutdown wrapper for the ACP subprocess."""
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
                "token": os.environ.get("DIPLOID_CONTROL_TOKEN", ""),
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
    """Prepare and manage the isolated HOME for the ACP subprocess."""

    def __init__(self, service_name: str | None = None) -> None:
        self._service_name = service_name
        self.devin_home: Path | None = None
        self._junctions: list[Path] = []

    def prepare(self, mcp_servers: list[dict[str, Any]] | None = None) -> None:
        """Create an isolated HOME for the ACP subprocess and write configs.

        Devin's bundled MCP config (e.g. `~/.codeium/windsurf/mcp_config.json`)
        can include MCP servers that deadlock or saturate and block `devin acp`
        startup indefinitely. We create a sanitized home directory and write the
        active MCP server list into `mcp_config.json` before `devin acp` starts,
        because devin 3000.6.7+ loads servers from that file.
        """
        if self.devin_home is not None and self.devin_home.exists():
            # Re-sanitize the MCP config on every (re)start in case the previous
            # `devin` subprocess wrote to it.
            self.write_mcp_configs(mcp_servers)
            return

        self.devin_home = Path(tempfile.mkdtemp(prefix="acp-home-"))
        if IS_WINDOWS:
            self._prepare_windows_home()
        config_dir = self.devin_home / ".config" / "devin"
        config_dir.mkdir(parents=True, exist_ok=True)
        codeium_dir = self.devin_home / ".codeium" / "windsurf"
        codeium_dir.mkdir(parents=True, exist_ok=True)

        # Sandbox the ACP subprocess: create a private runtime dir and fake systemctl
        # wrapper so the model cannot run raw `systemctl --user restart` from the
        # subprocess. Restart requests are routed through the harness control socket.
        (self.devin_home / ".run").mkdir(parents=True, exist_ok=True)
        if not IS_WINDOWS:
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

    def _prepare_windows_home(self) -> None:
        """Junction shared devin state into the isolated home on Windows.

        Windows devin resolves config/state via %APPDATA%, not $HOME, so the
        isolated home must shadow AppData: mcp_config.json and config.json
        stay isolated (plugin servers, permissions), while cli/, User/, and
        mcp/ are junctioned to the real dirs so sessions.db, session_locks,
        and acp-messages stay shared across children and custody mouths.
        Junctions are reparse points that os.path.islink() does not detect;
        they must be rmdir'd before shutil.rmtree or the real targets get
        emptied on cleanup.
        """
        import _winapi

        assert self.devin_home is not None
        real_roaming = (
            Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "devin"
        )
        iso_devin = self.devin_home / "AppData" / "Roaming" / "devin"
        iso_devin.mkdir(parents=True, exist_ok=True)
        (self.devin_home / "AppData" / "Local").mkdir(parents=True, exist_ok=True)
        for sub in ("cli", "User", "mcp"):
            target = real_roaming / sub
            if target.is_dir():
                link = iso_devin / sub
                _winapi.CreateJunction(str(target), str(link))
                self._junctions.append(link)
        real_local = (
            Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "devin"
        )
        if real_local.is_dir():
            link = self.devin_home / "AppData" / "Local" / "devin"
            _winapi.CreateJunction(str(real_local), str(link))
            self._junctions.append(link)
        # User site-packages lives under %APPDATA%\Python — without this
        # junction, children spawned under the isolated env lose every
        # user-installed package and die with ModuleNotFoundError before
        # the MCP initialize handshake completes.
        real_python = real_roaming.parent / "Python"
        if real_python.is_dir():
            link = iso_devin.parent / "Python"
            _winapi.CreateJunction(str(real_python), str(link))
            self._junctions.append(link)
        user_config = real_roaming / "config.json"
        if user_config.exists():
            try:
                (iso_devin / "config.json").write_text(user_config.read_text())
            except OSError:
                logger.warning("Failed to copy devin config to ACP home")
        else:
            (iso_devin / "config.json").write_text(
                json.dumps({"version": 1, "permissions": {"allow": ["*"]}})
            )

    def cleanup(self) -> None:
        """Remove the isolated HOME created for the ACP subprocess."""
        home = self.devin_home
        self.devin_home = None
        if home is not None and home.exists():
            for junction in self._junctions:
                try:
                    junction.rmdir()
                except OSError:
                    logger.warning("Failed to remove junction %s", junction)
            self._junctions.clear()
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
        before the subprocess is spawned.
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
            if IS_WINDOWS:
                # MCP stdio is UTF-8 JSON-RPC, but a Windows child defaults to
                # the locale codec (e.g. GBK+surrogateescape): a UTF-8 em-dash
                # becomes <GBK char>\udc94, and that lone surrogate then wedges
                # any peer's json.dumps().encode("utf-8") on record. Pin the
                # child's stdio to UTF-8 unless the server opts out explicitly.
                entry.setdefault("env", {}).setdefault("PYTHONIOENCODING", "utf-8")
            servers[name] = entry

        mcp_config = json.dumps({"mcpServers": servers}, indent=2)
        paths = [
            self.devin_home / ".config" / "devin" / "mcp_config.json",
            self.devin_home / ".codeium" / "windsurf" / "mcp_config.json",
        ]
        if IS_WINDOWS:
            paths.append(self.devin_home / "AppData" / "Roaming" / "devin" / "mcp_config.json")
        for path in paths:
            try:
                path.write_text(mcp_config)
            except OSError:
                logger.warning("Failed to write mcp_config.json to %s", path)

    def _write_sandbox_binaries(self) -> None:
        """Install fake system control binaries into the isolated HOME.

        The ACP subprocess inherits the real host `PATH` and can run
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
        non-empty list to a dict before sending it to the ACP subprocess.
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
