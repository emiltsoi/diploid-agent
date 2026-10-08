"""macOS gate probe — G-B (LAN reach) / G-C (vault read) / G-D (bind) for the
diploid harness, run inside a launchd→fleet-host spawn chain.

Writes one JSON object per check to stdout so the plist's StandardOutPath is
the evidence artifact. Exits 0 iff every check passes.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import urllib.request

RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": ok, "detail": detail})
    print(json.dumps(RESULTS[-1]), flush=True)


def http_ok(url: str, timeout: float = 8.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 500, f"HTTP {r.status}"
    except Exception as exc:  # noqa: BLE001 — probe records, never raises
        return False, f"{type(exc).__name__}: {exc}"


def tcp_ok(host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "connected"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def bind_ok(host: str, port: int) -> tuple[bool, str]:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.close()
        return True, "bound"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    record("context", True, f"pid={os.getpid()} argv0={sys.argv[0]} uid={os.getuid()}")

    ok, d = http_ok("https://api.telegram.org/")
    record("G-B.internet_telegram_api", ok, d)

    ok, d = http_ok("http://192.168.1.34:8888/")
    record("G-B.hindsight", ok, d)

    ok, d = tcp_ok("192.168.1.51", 8646)
    record("G-B.mesh_registry_lan", ok, d)

    ok, d = bind_ok("0.0.0.0", 4003)
    record("G-D.bind_4003", ok, d)

    ok, d = bind_ok("0.0.0.0", 4004)
    record("G-D.bind_4004", ok, d)

    try:
        entries = os.listdir("/nas")
        ok, d = True, f"{len(entries)} entries"
    except Exception as exc:
        ok, d = False, f"{type(exc).__name__}: {exc}"
    record("G-C.vault_listdir_/nas", ok, d)

    shelf = "/nas/emiltsoi/Agents/vault/common-room/shelf/aurelia"
    try:
        entries = os.listdir(shelf)
        ok, d = True, f"{len(entries)} shelf files"
    except Exception as exc:
        ok, d = False, f"{type(exc).__name__}: {exc}"
    record("G-C.vault_shelf_read", ok, d)

    try:
        out = subprocess.run(
            ["devin", "--version"], capture_output=True, text=True, timeout=30, check=False
        )
        record("G-A.devin_in_chain", out.returncode == 0, out.stdout.strip() or out.stderr.strip())
    except Exception as exc:
        record("G-A.devin_in_chain", False, f"{type(exc).__name__}: {exc}")

    return 0 if all(r["ok"] for r in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
