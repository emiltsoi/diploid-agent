#!/usr/bin/env bash
# harness-run.sh — run a harness ingress + telegram poller pair for any persona.
#
# Usage: harness-run.sh <harness-yaml> <listen-port>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

# Unset Windsurf IDE markers so that a standalone `devin acp` can use the
# credentials file / WINDSURF_API_KEY instead of waiting for an IDE host.
unset ACP_BACKEND WINDSURF_IDE_TYPE WINDSURF_EXT_HOST_PID

# Do not source secrets here. The systemd unit loads them via EnvironmentFile,
# and a manual run can source them before invoking this script.
CONFIG="${1:-config/harness.yaml}"
PORT="${2:-4003}"

# Start the Telegram poller in the background.
.venv/bin/python -m diploid_agent.telegram_poll \
  --config "$CONFIG" \
  --harness-url "http://127.0.0.1:$PORT" &
POLLER_PID=$!

# Start the FastAPI ingress in the background.
.venv/bin/python -m diploid_agent.telegram_ingress \
  --config "$CONFIG" &
INGRESS_PID=$!

# If the script is stopped, stop both children.
cleanup() {
  kill "$POLLER_PID" "$INGRESS_PID" 2>/dev/null || true
  for pid in "$POLLER_PID" "$INGRESS_PID"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# Wait for either child to exit. As soon as one dies, the script exits with
# that child's exit code so systemd can restart the pair on failure.
set +e
wait -n
EXIT_CODE=$?
set -e
exit $EXIT_CODE
