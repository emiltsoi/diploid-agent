# Telegram integration

## Setup

1. Create a bot with [BotFather](https://t.me/BotFather) and copy the token.
2. Add it to `config/secrets.env`:

   ```bash
   TELEGRAM_BOT_TOKEN=...
   ```

3. Start the service. `systemd/harness-run.sh` starts both the ingress and
the long-polling bot.

The token is never logged. `httpx` request logging is suppressed to avoid
leaking the token, and the poller uses `POST` form data instead of query strings.

## How the poller works

`transport/telegram/poller.py` long-polls Telegram's `getUpdates` endpoint. For each
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

## Queued messages and outbox delivery

When a user sends a message while the chat is already running a turn, the
harness queues it as a high-priority `user_request` wake. The user immediately
sees:

```
I'll get back to you in a moment.
System: This chat is busy; your message was queued.
```

When the current turn finishes, the queued message is processed. If
`harness.notifications.outbox_delivery` is `true` (the default in
`runtime-overrides.yaml` for the shipped personas), the final `ChatResult` is
pushed to the harness's outbox. A single global `DeliveryWorker` in the
Telegram poller consumes it via `GET /outbox` and delivers the result to the
chat stored in the popped item. This lets background turns, subagent
completions, mesh wake replies, and wake continuations reach the user even
when the harness process has no runtime-side notifier.

While the turn is running, the `TurnWorker` edits the placeholder message every
few seconds. If the model has produced no visible reply text yet, the
placeholder is updated with a liveness suffix such as `(still working, 1m
30s)`. For wake-driven (outbox) turns, the harness now also pushes a
`⏳ Still thinking... (1m 30s)` result to the outbox after 30 seconds, and every
90 seconds after that, so the user knows the agent is alive.

The same outbox path is used for the optional mesh Telegram float. When
`harness.notifications.mesh_telegram_float` is `true`, the harness inserts a
system message such as `System: [mesh] aurelia → vesper: pong` into the outbox
after the agent calls `mesh_send` successfully.

## Service restart notice

When the systemd service restarts, the harness sends a direct message to each
recently active chat:

```
System: service was restarted. You can resume the conversation at any time.
```

This is sent directly through the Telegram Bot API (not the outbox), because the
poller and its global `DeliveryWorker` may not be connected to the harness yet
immediately after the harness starts.

## Stale-wake cleanup

On startup the harness drops `auto_continue` wakes that were created before the
current process started. This prevents a crash-restart loop from immediately
re-firing a stale `Continue` turn. Queued user messages (`user_request` wakes),
`dispatch`, `plan_task_update`, and `plan_completed` wakes are kept; the
conversation and session state used for resume are not touched.

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
- A new `...` placeholder is sent below it. Only the *uncommitted tail* of
  the stream is shown there — text already committed to an earlier message is
  never repeated (the liveness heartbeat works on the tail too).
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

## Streamed thoughts

`/stream_thoughts on` enables a second `Thinking...` placeholder, sent before
the reply placeholder, that streams `agent_thought` updates while the turn
runs. When the turn finishes, the live placeholders are deleted and the full
accumulated thought is re-sent as one or more `Thinking:` messages (split at
the 4096-character limit) above the final reply, so long reasoning traces are
not lost or interleaved with the answer.

## Asking the user to choose

When the assistant needs the user to pick from a list of options, it can include a fenced `ask` block in its reply:

````
Which file should I edit?

```ask
{"question": "Which file should I edit?", "options": ["a.py", "b.py", "c.py"]}
```
````

The poller strips the block, sends the question as a Telegram message, and attaches an inline keyboard with the options. When you tap a button, Telegram sends a callback query; the poller rewrites your choice as an explicit answer and sends it to the harness, so no user-facing message is created.

For a simple approval dialog:

````
Should I continue?

```ask
{"question": "Should I continue?", "options": ["Approve", "Decline"]}
```
````

Every ask block has a default cancel button, so the user can dismiss the prompt without waking the assistant and **no** turn is sent to the harness. When the user presses it, the question is edited to `Cancelled.` and the keyboard is removed.

Do not include `"Other (please specify)"` as an option. If the options are not exhaustive, the cancel button is the escape hatch. If you truly need a custom open-ended answer, ask the user directly in a follow-up after they cancel, or set `cancellable: false` and make "Other" a regular option.

If the user should be able to cancel the prompt without sending a turn, the default is already on. Set `cancellable: false` to make a forced-choice prompt with no cancel button, or provide an optional `cancel_label` (default `"Cancel"`):

````
Should I continue? (forced choice)

```ask
{"question": "Should I continue?", "options": ["Yes", "No"], "cancellable": false}
```
````

````
Should I continue? (custom cancel label)

```ask
{"question": "Should I continue?", "options": ["Yes", "No"], "cancel_label": "Never mind"}
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

## Attachments

When a message carries a file — a photo, document, voice note, video, sticker,
or animation — the poller downloads it through the Bot API (`getFile` plus the
file endpoint) and saves it under the chat's ACP workspace:

```
<sessions_root>/<chat_id>/inbox/<message_id>-<name>
```

The agent then sees the caption (if any) plus one annotation per file:

```
[attachment saved: inbox/231-holiday.jpg (photo, image/jpeg)]
```

so it can `read` the file like anything else in its workspace. Because the
inbox lives under the session directory, a `session:` cron `file` trigger can
also watch it — an arriving photo can wake a job without spending a turn.

Details:

- The `photo` field is a size ladder; only the largest variant is downloaded.
- A captionless attachment still reaches the agent — the annotation *is* the
  message text.
- Filenames are sanitized to a single safe segment and prefixed with the
  message id, so `../../etc/passwd` lands as `passwd` inside the inbox.
- `harness.telegram.attachments_max_bytes` (default 20 MB, the Bot API's
  `getFile` ceiling) is enforced on both the declared size and the streamed
  body; a partial download is removed.
- A download that fails or is skipped is annotated as
  `[attachment could not be saved: ...]` rather than dropping the message.
- `harness.telegram.attachments_enabled: false` restores the old behavior:
  media is ignored entirely. `attachments_dirname` renames the subfolder.

Downloads happen on the turn worker, not the poll loop, so a large file cannot
stall `getUpdates` for other chats.

### Transcription (STT)

Voice notes, audio files, and video notes can be transcribed at ingest, right
after the download:

```yaml
telegram:
  stt_provider: faster-whisper   # none | faster-whisper | command
  stt_model: base                # whisper size for faster-whisper
  stt_command: ""                # command provider: invoked as `cmd <file>`,
                                 # stdout becomes the transcript
```

The transcript is appended to the message annotation:

```
[attachment saved: inbox/2373-voice.oga (voice, audio/ogg)]
[transcript: "love, check the evening job"]
```

- `none` (default) skips transcription entirely.
- `faster-whisper` requires the package in the poller's Python env (it is not
  a hard dependency). One `WhisperModel` per `stt_model` size is loaded and
  cached; CPU int8 inference is enough for message-length notes.
- `command` runs `stt_command <file>` with a 60s timeout — the escape hatch
  for whisper.cpp, a host-side speech bridge, or anything else that prints a
  transcript.
- A provider failure annotates `[transcript unavailable]` rather than
  dropping the message; the audio file is kept either way.

### Speaking aloud (TTS)

A fenced `say` block in a reply is synthesized and sent as a Telegram voice
note (or a plain audio file for non-ogg output):

````
Here is the text version.

```say
Good night, love. The watch is standing.
```
````

```yaml
telegram:
  tts_provider: piper        # none | piper | command
  tts_model_path: ~/.devin/personas/vesper/voice/en_US-hfc_female-medium.onnx
  tts_command: ""            # command provider: text on stdin → audio bytes on stdout
  tts_max_chars: 800         # longer say blocks fall back to text
```

- The `say` block is always stripped from the text. If TTS is off, over the
  char cap, or synthesis/upload fails, the content is sent as a
  `[say] ...` text line — authored words are never dropped.
- `piper` requires `piper-tts` in the poller env plus a voice `.onnx` (with its
  `.onnx.json`) at `tts_model_path` (`~` is expanded). Wav output is converted
  to ogg/opus with `ffmpeg`, which must be on `PATH`. One `PiperVoice` per
  model path is cached.
- `command` reads the text on stdin and must emit audio on stdout; `OggS`
  output is sent via `sendVoice`, anything else via `sendAudio`. This is the
  escape hatch for a host-side speech bridge (e.g. macOS `say`/`afconvert`
  from a Linux guest).
- A say-only reply deletes the streaming placeholder instead of leaving a
  dangling `...` bubble.

### Sending files

A fenced `file` block in a reply uploads a file from the chat workspace back to
Telegram — the outbound half of attachments:

````
Here is the report I promised.

```file
outbox/september-notes.pdf
A short summary for September.
```
````

- First line is the path — relative to the chat workspace
  (`<sessions_root>/<chat_id>/`), or absolute underneath it. Any lines below
  become the caption (truncated to 1024 chars). Multiple blocks send multiple
  files, in order.
- The method is chosen by extension: images → `sendPhoto`, `.gif` →
  `sendAnimation`, video → `sendVideo`, everything else → `sendDocument`.
- The path is resolved post-`resolve()` and must stay inside the chat
  workspace; escapes (`../`, absolute paths outside) are refused.
- The block is always stripped from the text. If the file is missing, escapes,
  exceeds `attachments_max_bytes`, or the upload fails, the user still gets a
  `[file] <path>` message with the caption — nothing is silently dropped.
- A reply made only of `file` blocks deletes the streaming placeholder like a
  say-only reply does.

## Commands

At startup the poller pushes these to Telegram via `setMyCommands`, so the
client's "/" menu lists them with descriptions (Telegram menu names allow
only lowercase letters, digits, and underscores — hence `/graceful_restart`).
Disable with `harness.telegram.bot_menu: false`.

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
| `/plugin list` | List configured state plugins and enabled state. |
| `/plugin enable <name>` | Enable a plugin for this chat. |
| `/plugin disable <name>` | Disable a plugin for this chat. |
| `/plugin reload <name>` | Hot-swap the plugin's code without a restart. |
| `/memory` | Show per-chat memory. |
| `/models` | List ACP model names. |
| `/model [--in-place] <name>` | Switch this chat to a new model. `--in-place` changes the model on the live session instead of starting a new one. |
| `/new` | Start a fresh Devin session for this chat while keeping chat memory. |
| `/stop` | Cancel the current turn and return a partial reply. |
| `/restart` | Kill the ACP subprocess and start a fresh transport. |
| `/graceful_restart [service]` | Schedule a graceful `systemd-run` restart of the named service. If `service` is omitted, the current persona's `.service` unit is restarted. |
| `/subagent <prompt>` | Start a background ACP subagent. The harness continues the chat with the result when it finishes. |
| `/subagents` | List background subagents for this chat. |
| `/continue` | Resume the previous turn after a partial reply or timeout. |
| `/stream_thoughts on\|off` | Toggle the optional real-time thought stream. |
| `/sessions` | List numbered sessions for this chat. |
| `/resume <n>` | Resume session `n` as the active session. |
| `/branch <n>` | Branch from session `n` and make it the active session. |
| `/summarize` | Trigger file-backend summarization. |
| `/recall <query>` | Search memory for relevant context. |
| `/promote <fact>` | Append a fact to this chat's promoted memory pocket. |
| `/help` | Show the list of Telegram slash commands. |
| `/config <section> <key>=<value> [key=value...]` | Update live runtime config (task, waker, timer, notifications, telegram) without restarting. |

Anything else is treated as a normal chat message.

The Telegram poller uses the same `transport/command_handler.py` dispatch layer
as the HTTP endpoints. When the poller is connected directly to a `RuntimeAPI`
it calls the runtime methods; when it is configured with a `harness_url` it
calls the matching HTTP endpoints. The formatting and command-specific parsing
live in `transport/telegram/poller.py`, `commands.py`, `sender.py`, `state.py`, and `formatting.py`.

### Live runtime configuration

You can adjust the harness's live runtime configuration directly from Telegram without restarting the service:

```
/config <section> <key>=<value> [key=value...]
```

`<section>` is one of `task`, `waker`, `timer`, `notifications`, or `telegram`. The poller parses each `key=value` pair and applies it against the runtime. In the single-process deployment this happens in-process; when the poller talks to a remote harness it POSTs the fields to the corresponding `/{section}/config` endpoint. For example:

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

In the single-process deployment `telegram` changes take effect immediately —
the poller re-reads the runtime config on every message. In the two-process
deployment the command is routed through `PATCH /config` (there is no
`/telegram/config` route) and persisted to `runtime-overrides.yaml`, but the
fields are poller-side rendering settings, so a remote poller only picks them
up on restart. The other sections (`task`, `waker`, `timer`,
`notifications`) take effect immediately on the running harness in either
mode.

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
