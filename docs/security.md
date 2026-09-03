# Security notes

## Secrets

- `config/secrets.env` is **gitignored**. It holds `TELEGRAM_BOT_TOKEN` and
  optional ACP API keys.
- `config/harness.yaml` is **gitignored** because it contains local paths.
- Only `.example` templates are tracked.
- The Telegram token is never printed in logs.
- `httpx` and `httpcore` log levels are raised to `WARNING` so raw request URLs
  are not logged.
- The poller uses `POST` form data to Telegram, not query strings, so the token
  does not appear in access logs.

## ACP authentication

The default engine spawns `devin acp`. It needs either:

1. `WINDSURF_API_KEY` or `ACP_API_KEY` set in `config/secrets.env`, or
2. the credentials file written by `devin auth login` or Devin Desktop:
   `~/.local/share/devin/credentials.toml`.

Other engines may use a different API key or credential source.

The systemd user unit loads `config/secrets.env`. If you run the unit under the
same user that is signed in to Devin Desktop, the credentials file is found
automatically and the harness works without putting the token in `secrets.env`.

## Model self-identification

The agent cannot reliably report the model it is running. The harness prefixes
a system-generated `Now running on model \`<model>\`.` line instead of asking
the agent to guess. Use `/status` as the source of truth.

## Persona files

Persona files (`SOUL.md`, `AGENTS.md`, `MEMORY.md`, etc.) live under
`personas/<persona>` in the repository by default. The harness reads them but
does not write to them except via the explicit `/promote` command.

## HTTP/Telegram ingress

The FastAPI ingress is intended to run on a trusted or private network
(`127.0.0.1` by default). If you expose it externally, set `HARNESS_API_KEY` in
`config/secrets.env` (or the environment) and include it in the `X-API-Key`
header on every `POST` request and on `GET` requests to the live runtime config
endpoints (`/task/config`, `/waker/config`, `/timer/config`,
`/notifications/config`). Other `GET` endpoints and the Telegram `/webhook`
remain unauthenticated so that Telegram updates and health checks still work.

## Sessions and runtime state

The `sessions/` directory and `sessions.jsonl` contain conversation data. They
are gitignored and should be treated as runtime state.

## Hindsight

The Hindsight backend is unauthenticated by default in the current environment
(`http://localhost:8888`). If an API key is needed later, set
`harness.memory.hindsight.api_key`; the backend sends it in the `authorization`
header.

Hindsight is treated as eventually consistent: failed retains are spooled locally
and retried; failed recalls fall back to local transcript search.

## ACP subprocess sandbox and self-restart

When `permission_mode` is `dangerous` (ACP `bypass`), the ACP subprocess can auto-approve
tool calls including shell execution. This would let the agent run
`systemctl --user restart <service>` and kill its own harness. To prevent that:

- `AcpClient` creates an isolated home for the subprocess with a private `XDG_RUNTIME_DIR`
  and unsets `DBUS_SESSION_BUS_ADDRESS`, so the subprocess cannot reach the user's systemd
  manager directly.
- A fake `systemctl` wrapper (plus `reboot`, `poweroff`, `shutdown`, and `halt`)
  is placed first on the subprocess's `PATH`. It forwards restart requests to the harness
  over a private Unix socket.
- The harness schedules the actual restart with `systemd-run --on-active=<delay>`
  after sending its reply, so the restart is graceful and the user sees a message
  before the service goes down.
- Restart requests are rate-limited and suppress `auto_continue` wakes so a single
  "restart now" thought cannot loop the service.
- Long-running background work is started through `harness_subagent` (the
  `diploid-harness` MCP tool), which runs in a fresh AcpEngine via the TaskEngine.
  The parent turn can be stopped while the subagent continues, and the harness
  starts a new turn with the result.
