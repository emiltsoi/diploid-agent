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

# Make the project venv the first python on PATH so that `devin acp` and the
# stdio MCP servers it spawns resolve `python` to the correct interpreter.
export PATH="$PROJECT_DIR/.venv/bin:$PATH"

# Build a PYTHONPATH that contains the project sources and any persona plugin
# directories declared in the harness config. `devin acp` inherits this env and
# passes it to each MCP server it starts.
PYTHONPATH_BASE="$PROJECT_DIR/src"
if [ -f "$CONFIG" ]; then
  PLUGIN_DIRS=$($PROJECT_DIR/.venv/bin/python - "$CONFIG" "$PROJECT_DIR" <<'PY'
import os, sys, yaml
cfg_file = sys.argv[1]
project = sys.argv[2]
try:
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
except Exception:
    cfg = {}
plugin_paths = cfg.get('harness', {}).get('plugin_paths', [])
out = [project + '/src']
for p in plugin_paths:
    p = os.path.expanduser(p)
    if not os.path.isabs(p):
        p = os.path.join(project, p)
    if p not in out:
        out.append(p)
print(':'.join(out))
PY
)
  if [ -n "$PLUGIN_DIRS" ]; then
    PYTHONPATH_BASE="$PLUGIN_DIRS"
  fi
fi
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$PYTHONPATH_BASE:$PYTHONPATH"
else
  export PYTHONPATH="$PYTHONPATH_BASE"
fi

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
