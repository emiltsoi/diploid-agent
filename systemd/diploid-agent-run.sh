#!/usr/bin/env bash
# diploid-agent-run.sh — run BOTH harness processes under one systemd unit.
#
# Starts the Telegram poller and the FastAPI ingress. If either process
# exits, the script kills the other and exits, so systemd can restart the
# whole pair (Restart=on-failure in diploid-agent.service).
set -euo pipefail

# Resolve to the project root (this script lives in systemd/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

# Unset Windsurf IDE markers so that a standalone `devin acp` can use the
# credentials file / WINDSURF_API_KEY instead of waiting for an IDE host.
unset ACP_BACKEND WINDSURF_IDE_TYPE WINDSURF_EXT_HOST_PID

# Load environment (secrets) if the unit did not already source them.
if [ -f "config/secrets.env" ]; then
    set -a
    # shellcheck source=/dev/null
    . config/secrets.env
    set +a
fi

# Start the Telegram poller in the background.
.venv/bin/python -m diploid_agent.telegram_poll \
  --config config/harness.yaml &
POLLER_PID=$!

# Start the FastAPI ingress in the background.
.venv/bin/python -m diploid_agent.telegram_ingress \
  --config config/harness.yaml &
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
# that child's exit code so systemd restarts the pair on failure.
set +e
wait -n
EXIT_CODE=$?
set -e
exit $EXIT_CODE
