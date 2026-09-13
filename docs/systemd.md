# Running under systemd

The harness has two long-running processes:

1. `telegram_ingress` — FastAPI server.
2. `telegram_poll` — Telegram long-polling bot.

They are started together by `systemd/harness-run.sh`. If either subprocess exits,
the script exits and systemd restarts the pair.

## Run script

`systemd/harness-run.sh <harness-yaml> <listen-port>` (defaults
`config/harness.yaml` and `4003`):

- Resolves the project root from the script's own path.
- Unsets Windsurf IDE markers so `devin acp` uses the credentials file or
  `WINDSURF_API_KEY` instead of waiting for an IDE host.
- Puts `.venv/bin` first on `PATH` and builds a `PYTHONPATH` containing the
  project sources plus any persona `plugin_paths` declared in the config.
- Does **not** source `config/secrets.env` — secrets are loaded by the unit's
  `EnvironmentFile` (or sourced manually before a manual run).
- Starts the poller and the ingress as background subprocesses and forwards
  `TERM` to both on stop, so the ingress can drain active turns under
  `KillMode=mixed`.
- Waits for either subprocess to exit, then kills the other and returns the failing
  subprocess's exit code.

This lets `Restart=on-failure` restart the whole unit as a pair.

## Service file

Copy and edit the example:

```bash
cp systemd/diploid-agent.service.example systemd/diploid-agent.service
# replace /home/USER paths
```

`systemd/diploid-agent.service`:

```ini
[Unit]
Description=ACP fleet harness — ingress + telegram poller
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/USER/diploid-agent
ExecStart=/home/USER/diploid-agent/systemd/harness-run.sh config/harness.yaml 4003
Restart=on-failure
RestartSec=10
EnvironmentFile=-/home/USER/diploid-agent/config/secrets.env

[Install]
WantedBy=default.target
```

## User unit

```bash
systemctl --user enable --now "$(pwd)/systemd/diploid-agent.service"
```

For a user unit, the `User=` line must be omitted (it is not in the example).
For a system unit, add `User=<username>`.

## Authentication note

The default engine (`devin acp`) needs either:

1. `WINDSURF_API_KEY` or `ACP_API_KEY` in `config/secrets.env`, or
2. `~/.local/share/devin/credentials.toml` from a previous `devin auth login`.

Other engines may use a different key or credential source.

The systemd unit runs as your user and loads `config/secrets.env` via
`EnvironmentFile`. If you sign in through Devin Desktop or `devin auth login` on
the same account, the credentials file is found automatically.

## Graceful self-restart

The agent (or a user) can request a service restart through the harness instead
of killing the unit directly:

- Telegram: `/graceful-restart [service]`
- HTTP: `POST /graceful-restart` with `{"chat_id": "...", "service": "..."}`
- ACP subprocess (in `permission_mode: dangerous`): the subprocess can run
  `systemctl --user restart <service>` and the harness intercepts it.

In all cases the harness sends an acknowledgement, then schedules the actual
restart with `systemd-run --user --on-active=5s`. This gives the HTTP/Telegram
response time to be delivered before the service goes down.

When the service comes back up, `AgentRuntime.start` sends a direct
`System: service was restarted.` message to every chat whose latest session was
updated in the last 24 hours, so users know the bot is back online. It also drops
stale `auto_continue` wakes from the previous process to avoid immediately
re-firing an old continuation.
