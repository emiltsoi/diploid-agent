#!/usr/bin/env python3
"""Standalone waker: polls WakeQueue and POSTs to /wake.

Run as a systemd service or from cron. It reads the wake queue directly
from disk so it can function even when the main harness is down; once an
event is due, it calls POST /wake on the harness. The /wake endpoint
consumes the event on success, so this probe only fails the event when
the harness is busy or the request fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

from diploid_agent.runtime.wake_queue import WakeQueue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake-queue", type=Path, required=True)
    parser.add_argument("--harness-url", default="http://127.0.0.1:4003")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--lease-seconds", type=float, default=300.0)
    parser.add_argument("--retry-after", type=float, default=30.0)
    parser.add_argument("--api-key")
    args = parser.parse_args()

    q = WakeQueue(args.wake_queue)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    while True:
        for event in q.pop_due(now=time.time(), lease_seconds=args.lease_seconds):
            try:
                resp = httpx.post(
                    urljoin(args.harness_url.rstrip("/") + "/", "wake"),
                    json={
                        "chat_id": event.chat_id,
                        "reason": event.reason,
                        "event_id": event.id,
                    },
                    headers=headers,
                    timeout=60.0,
                )
                if resp.status_code == 200:
                    # The /wake endpoint completed the event on success.
                    pass
                else:
                    q.fail(event.id, retry_after=args.retry_after)
                    print(
                        f"wake failed for {event.id}: {resp.status_code} {resp.text}",
                        file=sys.stderr,
                    )
            except Exception as exc:  # noqa: BLE001
                q.fail(event.id, retry_after=args.retry_after)
                print(f"wake error for {event.id}: {exc}", file=sys.stderr)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
