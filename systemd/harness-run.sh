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

# Secrets: under systemd they arrive via the unit's EnvironmentFile. Under
# launchd there is no EnvironmentFile, so source any env files listed in
# DIPLOID_ENV_FILES (colon-separated) — or, if unset, the conventional
# config/secrets*.env next to the harness config — but only for keys not
# already present in the environment. Sourcing is skipped entirely when the
# files are absent (e.g. a test checkout).
CONFIG="${1:-config/harness.yaml}"
PORT="${2:-4003}"

_load_env_file() {
  local f="$1"
  [ -f "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue;; esac
    line="${line#export }"
    key="${line%%=*}"
    [ "$key" = "$line" ] && continue
    if [ -z "${!key+x}" ]; then
      val="${line#*=}"
      val="${val%\"}"; val="${val#\"}"
      val="${val%\'}"; val="${val#\'}"
      export "$key=$val"
    fi
  done < "$f"
}
if [ -n "${DIPLOID_ENV_FILES:-}" ]; then
  IFS=':' read -rA _env_files <<< "$DIPLOID_ENV_FILES" 2>/dev/null || IFS=':' read -ra _env_files <<< "$DIPLOID_ENV_FILES"
  for _f in "${_env_files[@]}"; do _load_env_file "$_f"; done
else
  for _f in "$PROJECT_DIR/config/secrets.env" \
            "$PROJECT_DIR/config/secrets-$(basename "$CONFIG" .yaml | sed 's/^harness-//;s/^harness$/aurelia/').env"; do
    _load_env_file "$_f"
  done
fi

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

# If the script is stopped, forward TERM to both subprocesses and wait for them.
# Under KillMode=mixed systemd signals only this wrapper first; the ingress then
# gets a chance to drain active turns before the cgroup is SIGKILLed.
cleanup() {
  kill -TERM "$POLLER_PID" "$INGRESS_PID" 2>/dev/null || true
  for pid in "$POLLER_PID" "$INGRESS_PID"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# Wait for either subprocess to exit. As soon as one dies, the script exits with
# that subprocess's exit code so systemd can restart the pair on failure.
set +e
wait -n
EXIT_CODE=$?
set -e
exit $EXIT_CODE
