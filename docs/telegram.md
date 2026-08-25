# Telegram integration

## Setup

1. Create a bot with [BotFather](https://t.me/BotFather) and copy the token.
2. Add it to `config/secrets.env`:

   ```bash
   TELEGRAM_BOT_TOKEN=...
   ```

3. Start the service. `systemd/acp-fleet-harness-run.sh` starts both the ingress and
the long-polling bot.

The token is never logged. `httpx` request logging is suppressed to avoid
leaking the token, and the poller uses `POST` form data instead of query strings.

## How the poller works

`telegram_poll.py` long-polls Telegram's `getUpdates` endpoint. For each
message:

1. Skip messages sent by the bot itself.
2. Check for bot commands.
3. For normal messages, `POST /chat`.
4. For commands, `GET` or `POST` the relevant endpoint.
5. Stream the reply to Telegram by editing a placeholder message in place, then
   replace the placeholder with the final reply (splitting into multiple
   Telegram messages if it exceeds 4096 characters). The first bot message is
   sent as a Telegram reply to the user's message.

The poller is intentionally simple: all business logic lives in the FastAPI
ingress, so a `/webhook` endpoint could replace the poller without changing
behavior.

## Replying to messages

If you reply to any earlier message in the Telegram chat, the poller extracts
the text of the message you replied to and injects it into the next prompt with
a clear label:

```
[In reply to the assistant's earlier message:]
<quoted text>

[Your new message:]
<your reply>
```

If the quoted message is longer than `harness.memory.max_reply_quote_chars`
(default 2048 characters), it is trimmed and a `[... N characters truncated ...]`
marker is added.

The first bot message in a turn is also sent as a Telegram `reply` to your
message, so the conversation thread is visible.

## Commands

| Command | Action |
|---|---|
| `/status` | Show current model, session id, and working directory. |
| `/metrics` | Show token usage and latency for this chat. |
| `/mcp list` | List configured MCP servers and enabled state. |
| `/mcp enable <name>` | Enable an MCP server for this chat. |
| `/mcp disable <name>` | Disable an MCP server for this chat. |
| `/skill list` | List available skills and enabled state. |
| `/skill enable <name>` | Enable a skill for this chat. |
| `/skill disable <name>` | Disable a skill for this chat. |
| `/skill create <name> <markdown>` | Create a chat-scoped skill. |
| `/state <plugin> <event> [args...]` | Dispatch a state event to a plugin (e.g. `/state curriculum add_word hola hello`). |
| `/memory` | Show per-chat memory. |
| `/models` | List ACP model names. |
| `/model <name>` | Switch this chat to a new model. |
| `/new` | Start a fresh Devin session for this chat while keeping chat memory. |
| `/stop` | Cancel the current turn and return a partial reply. |
| `/continue` | Resume the previous turn after a partial reply or timeout. |
| `/stream_thoughts on\|off` | Toggle the optional real-time thought stream. |
| `/sessions` | List numbered sessions for this chat. |
| `/resume <n>` | Resume session `n` as the active session. |
| `/branch <n>` | Branch from session `n` and make it the active session. |
| `/summarize` | Trigger file-backend summarization. |
| `/recall <query>` | Search memory for relevant context. |
| `/promote <fact>` | Append a fact to the persona's global memory. |
| `/help` | Show the list of Telegram slash commands. |
| `/config <section> <key>=<value> [key=value...]` | Update live runtime config (task, waker, timer, notifications) without restarting. |

Anything else is treated as a normal chat message.

### Live runtime configuration

You can adjust the harness's live runtime configuration directly from Telegram without restarting the service:

```
/config <section> <key>=<value> [key=value...]
```

`<section>` is one of `task`, `waker`, `timer`, or `notifications`. The poller parses each `key=value` pair and POSTs it to the corresponding `/config` endpoint on the ingress. For example:

```
/config task workers=5
/config notifications webhook_url=https://example.com/notify
```

Invalid values are rejected with an error reply. Changes are persisted to `runtime-overrides.yaml` in the project root so they survive a harness restart.

For per-task ACP model switching, include `acp_model` (or `model`) in ACP tasks when using `/plan`, `!plan`, or `Plan:` triggers. Per-task values override the `task` runtime default.

## Resuming after a timeout

When a turn hits `devin.soft_timeout` (600 s by default) or is cancelled with
`/stop`, the harness returns a partial reply with a notice like:

> Reply `Continue` to keep going, or tell me what to change.

You can reply with the literal word **Continue** (or any configured
`continuation_triggers`), use the `/continue` command, or just send your next
message. If the ACP process was stuck, the harness will restart it and
rehydrate a fresh session from the durable transcript.


## Switching to a webhook

`telegram_ingress.py` exposes `POST /webhook` for Telegram push updates. If you
want to use it instead of polling:

1. Set the bot webhook URL to `https://your-host/webhook`.
2. Stop the poller.
3. Configure your reverse proxy to forward to `127.0.0.1:4003`.

The `/webhook` handler is minimal; it returns `{"ok": True, "reply": ...}`.

## Orphaned placeholders

The poller sends a `...` placeholder at the start of each turn and edits it in
place as the reply streams. If the poller process is killed before the turn
finishes, the placeholder is left in the chat. On startup the poller reads the
per-chat placeholder state files in `sessions/.poller-placeholders/` and either
updates the orphaned placeholder to a "Service restarted" notice or deletes it.
