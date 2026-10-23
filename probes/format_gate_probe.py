#!/usr/bin/env python3
"""format-gate probe — first seat-side bite-replay encoding.

Bite (2026-10-17): the pre-push verify list ran `ruff check` + pytest but
never `ruff format --check`. A lint-clean, suite-green push stack went red
in CI on format alone — two commits failed before a `ruff format` sweep.

discriminates: the unrun-gate mechanism — the local verify instruction
(AGENTS.md) drifting from CI's actual gate set (.github/workflows/ci.yml),
or a format-dirty tree.

fails_when: AGENTS.md lacks any gate CI enforces, OR `ruff check` /
`ruff format --check` exit nonzero on the tree.

Bound: cheap — two ruff invocations plus file reads, no pytest (the suite
is ~4min; `--with-tests` opts in for completeness).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
AGENTS = REPO / "AGENTS.md"
CI = REPO / ".github" / "workflows" / "ci.yml"
RUFF = REPO / ".venv" / "bin" / "ruff"
PYTEST_CMD = f"{REPO / '.venv' / 'bin' / 'python'} -m pytest tests -q"


def ci_gates() -> list[str]:
    """Gate commands CI enforces, in workflow order."""
    gates: list[str] = []
    for line in CI.read_text().splitlines():
        m = re.match(r"\s*run:\s*(.+?)\s*$", line)
        if m and re.search(r"ruff|pytest", m.group(1)):
            gates.append(m.group(1))
    return gates


def main() -> int:
    with_tests = "--with-tests" in sys.argv[1:]
    failures: list[str] = []

    gates = ci_gates()
    if not gates:
        failures.append("no ruff/pytest run: lines found in ci.yml — gate derivation broke")

    agents_text = AGENTS.read_text()
    for gate in gates:
        core = re.sub(r"^\.venv/bin/", "", gate)  # AGENTS.md prefixes .venv/bin/
        if core not in agents_text and gate not in agents_text:
            failures.append(f"AGENTS.md verify block missing CI gate: {gate}")

    for cmd in (
        [str(RUFF), "check", "src", "tests"],
        [str(RUFF), "format", "--check", "src", "tests"],
    ):
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if proc.returncode != 0:
            failures.append(f"`{' '.join(cmd[1:])}` exited {proc.returncode}\n{proc.stdout.strip()}")

    if with_tests:
        proc = subprocess.run(
            PYTEST_CMD.split(), cwd=REPO, capture_output=True, text=True
        )
        if proc.returncode != 0:
            failures.append(f"pytest exited {proc.returncode}\n{proc.stdout.strip()[-500:]}")

    if failures:
        print("format-gate probe: RED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"format-gate probe: GREEN ({len(gates)} CI gates mirrored in AGENTS.md; tree format-clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
