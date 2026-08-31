# Telegram integration

## Setup

1. Create a bot with [BotFather](https://t.me/BotFather) and copy the token.
2. Add it to `config/secrets.env`:

   ```bash
   TELEGRAM_BOT_TOKEN=...
   ```

3. Start the service. `systemd/diploid-agent-run.sh` starts both the ingress and
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
5. Stream the reply to Telegram by editing a placeholder message in place. When the
   streamed text pauses after a complete sentence and `intermediate_messages` is
   enabled, the current placeholder is committed as a sent message and a fresh
   placeholder is started below it. At the end the last placeholder is replaced
   with the final reply (splitting into multiple Telegram messages if it exceeds
   4096 characters). The first bot message is sent as a Telegram reply to the
   user's message.

The poller is intentionally simple: all business logic lives in the FastAPI
ingress, so a `/webhook` endpoint could replace the poller without changing
behavior.

## Intermediate messages

When the model pauses while writing a reply — typically while it is running a
tool — the placeholder text can end up combining the pre-tool statement and the
post-tool result into one message. This is confusing in a chat UI.

With `harness.telegram.intermediate_messages: true` (default), the poller
watches the streamed text:

- If the text is idle for at least `intermediate_idle` seconds, the
  uncommitted tail is at least `intermediate_min_chars` long, and it ends on a
  sentence or paragraph boundary (`.`, `!`, `?`, or a newline), the current
  placeholder is committed as a real message.
- A new `...` placeholder is sent below it.
- The rest of the reply streams into the new placeholder.
- At the end, the final reply is sliced to remove the already-committed prefix,
  so the user does not see the same text twice.

## Message format

`harness.telegram.message_format` controls how the final assistant reply is sent to Telegram:

- `plain` (default): sends the raw markdown text without formatting.
- `markdown_v2`: converts standard markdown to Telegram MarkdownV2 so `**bold**`, `*italic*`, `` `code` ``, ```` ``` ```` blocks, `[links](url)`, `~~strikethrough~~`, `||spoiler||`, `# headings`, and `>` blockquotes render correctly.

Tables are rewritten to bold headings with bullet groups because Telegram MarkdownV2 has no table syntax. Citation tags like `<ref_file file="..." />` and `<ref_snippet file="..." lines="..." />` are replaced with short source notes.

Streaming edits and intermediate placeholder commits remain plain text to avoid parsing incomplete markdown. If a formatted chunk fails Telegram parsing, the poller falls back to plain text for that chunk.

You can change the format live with:

```
/config telegram message_format=markdown_v2
```

## Asking the user to choose

When the assistant needs the user to pick from a list of options, it can include a fenced `ask` block in its reply:

````
Which file should I edit?

```ask
{"question": "Which file should I edit?", "options": ["a.py", "b.py", "c.py"]}
```
````

The poller strips the block, sends the question as a Telegram message, and attaches a one-time reply keyboard with the options. When you tap a button, the poller rewrites your choice as an explicit answer and sends it to the harness.

For a simple approval dialog:

````
Should I continue?

```ask
{"question": "Should I continue?", "options": ["Approve", "Decline", "Cancel"]}
```
````

If the list needs an open-ended escape option, include `"Other (please specify)"` as the final option:

````
Where should the rule live?

```ask
{"question": "Where should the rule live?", "options": ["Shared AGENTS.md", "Private AGENTS.md", "Both", "Other (please specify)"]}
```
````

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
| `/status` | Show current model, session id, working directory, and context-window usage. |
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
| `/restart` | Kill the ACP child and start a fresh transport. |
| `/graceful-restart [service]` | Schedule a graceful `systemd-run` restart of the named service. Defaults to the current persona's `.service` unit. |
| `/subagent <prompt>` | Start a background ACP subagent. The harness continues the chat with the result when it finishes. |
| `/continue` | Resume the previous turn after a partial reply or timeout. |
| `/stream_thoughts on\|off` | Toggle the optional real-time thought stream. |
| `/sessions` | List numbered sessions for this chat. |
| `/resume <n>` | Resume session `n` as the active session. |
| `/branch <n>` | Branch from session `n` and make it the active session. |
| `/summarize` | Trigger file-backend summarization. |
| `/recall <query>` | Search memory for relevant context. |
| `/promote <fact>` | Append a fact to the persona's global memory. |
| `/help` | Show the list of Telegram slash commands. |
| `/config <section> <key>=<value> [key=value...]` | Update live runtime config (task, waker, timer, notifications, telegram) without restarting. |

Anything else is treated as a normal chat message.

The Telegram poller uses the same `transport/command_handler.py` dispatch layer
as the HTTP endpoints. When the poller is connected directly to a `RuntimeAPI`
it calls the runtime methods; when it is configured with a `harness_url` it
calls the matching HTTP endpoints. The formatting and command-specific parsing
live in `transport/telegram.py`.

### Live runtime configuration

You can adjust the harness's live runtime configuration directly from Telegram without restarting the service:

```
/config <section> <key>=<value> [key=value...]
```

`<section>` is one of `task`, `waker`, `timer`, `notifications`, or `telegram`. The poller parses each `key=value` pair and POSTs it to the corresponding `/config` endpoint on the ingress. For example:

```
/config task workers=5
/config notifications webhook_url=https://example.com/notify
/config telegram intermediate_idle=3.0
/config telegram intermediate_messages=false
/config telegram message_format=markdown_v2
```

For the `telegram` section, the following keys may be updated live:

| Key | Type | Description |
|---|---|---|
| `intermediate_messages` | `true` / `false` | Whether to commit intermediate replies as separate messages. |
| `intermediate_idle` | seconds | How long the streamed text must be idle before an intermediate chunk is committed. |
| `intermediate_min_chars` | integer | Minimum length of the uncommitted tail before it can become its own message. |
| `stream_thoughts` | `true` / `false` | Toggle the real-time thought stream. |
| `stream_chunk_interval` | seconds | Reserved; currently unused. |
| `message_format` | `plain` / `markdown_v2` | How the final reply is formatted. |

Because the Telegram poller is a separate process, live `telegram` config changes only take effect after the poller restarts. Other sections (`task`, `waker`, `timer`, `notifications`) take effect immediately on the running harness.

Invalid values are rejected with an error reply. Changes are persisted to `runtime-overrides.yaml` in the project root so they survive a harness restart.

For per-task ACP model switching, include `acp_model` (or `model`) in ACP tasks when using `/plan`, `!plan`, or `Plan:` triggers. Per-task values override the `task` runtime default.

## Resuming after a timeout

When a turn hits `engine.soft_timeout` (600 s by default) or is cancelled with
`/stop`, the harness returns a partial reply with a notice like:

> Reply `Continue` to keep going, or tell me what to change.

You can reply with the literal word **Continue** (or any configured
`continuation_triggers`), use the `/continue` command, or just send your next
message. If the ACP session was stale, the harness reuses the existing transport
and rehydrates a fresh session from the durable transcript. If the transport
itself was stuck, it restarts the ACP process and then rehydrates.


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
