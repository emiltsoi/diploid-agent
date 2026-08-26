# Running under systemd

The harness has two long-running processes:

1. `telegram_ingress` — FastAPI server.
2. `telegram_poll` — Telegram long-polling bot.

They are started together by `systemd/diploid-agent-run.sh`. If either child exits,
the script exits and systemd restarts the pair.

## Run script

`systemd/diploid-agent-run.sh`:

- Resolves the project root from the script's own path.
- Loads `config/secrets.env` if present.
- Starts the poller and the ingress as background children.
- Waits for either child to exit, then kills the other and returns the failing
  child's exit code.

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
ExecStart=/home/USER/diploid-agent/systemd/diploid-agent-run.sh
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
