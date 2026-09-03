#!/usr/bin/env python3
"""Active watchdog: polls /health, attempts rollback, records incidents, then restarts."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Any

import httpx


def _call(
    method: str,
    url: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        if method == "GET":
            resp = httpx.get(url, headers=headers, timeout=10.0)
        else:
            resp = httpx.post(url, headers=headers, json=json_body, timeout=10.0)
        return resp.json()
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-url", default="http://127.0.0.1:4003")
    parser.add_argument("--service-name", default="diploid-agent")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--failure-threshold", type=int, default=3)
    parser.add_argument("--rollback-steps", type=int, default=1)
    parser.add_argument("--api-key")
    parser.add_argument("--user", action="store_true", default=True)
    args = parser.parse_args(argv)

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    base = args.harness_url.rstrip("/")
    fail_count = 0

    while True:
        health = _call("GET", f"{base}/health", headers)
        if health and health.get("status") == "ok":
            fail_count = 0
        else:
            fail_count += 1

        if fail_count >= args.failure_threshold:
            # First try a plugin rollback to avoid a full restart.
            _call("POST", f"{base}/config/rollback", headers, {"steps": args.rollback_steps})
            time.sleep(args.poll_interval)
            health = _call("GET", f"{base}/health", headers)
            if health and health.get("status") == "ok":
                _call(
                    "POST",
                    f"{base}/plugin-incidents",
                    headers,
                    {
                        "plugin": "harness",
                        "phase": "watchdog",
                        "error": "Rolled back to restore health",
                        "action": "rollback",
                    },
                )
                fail_count = 0
                continue

            # Record the incident, then restart the service.
            _call(
                "POST",
                f"{base}/plugin-incidents",
                headers,
                {
                    "plugin": "harness",
                    "phase": "watchdog",
                    "error": f"Health failed {fail_count} times",
                    "action": "restart",
                },
            )
            cmd = ["systemctl"]
            if args.user:
                cmd.append("--user")
            cmd.extend(["restart", args.service_name])
            try:
                subprocess.run(cmd, check=True)
            except Exception as exc:  # noqa: BLE001
                print(f"watchdog failed to restart {args.service_name}: {exc}", file=sys.stderr)
            fail_count = 0
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
