# Design decisions

## Why ACP

ACP (Agent Communication Protocol) gives the harness real-time streaming,
mode/model steering via `session/set_config_option`, and a single long-lived
process per chat. The default engine is the Devin CLI's `devin acp` backend, but
the transport is generic: any binary that speaks ACP v1 JSON-RPC over stdio can
be configured under `engine` with custom `bin` and `start_args`.

The Devin engine uses the same Devin Desktop or `devin auth login` credentials
as the CLI; it runs headlessly when `WINDSURF_API_KEY` or
`~/.local/share/devin/credentials.toml` is available. Other engines may use
`ACP_API_KEY` or their own credential source.

## Why one Hindsight bank per persona

The Hindsight contract and fleet convention use one bank per agent/persona
(`example-persona`, `example-agent-0`, etc.), with chat separation via tags rather
than separate banks. This lets Hindsight consolidate facts across all chats of
that persona, which is better for entity extraction and cross-conversation
knowledge.

## Why the agent is not asked to name its model

LLMs cannot reliably self-identify. If the prompt says "You are on model X",
the agent will echo it even if the actual serving model is different. The
harness therefore generates the "Now running on model X" text itself and uses
`/status` as the source of truth.

## Why the poller and ingress are separate processes

- The FastAPI ingress can be used by Telegram polling, Telegram webhooks, or
direct HTTP callers.
- The poller is a lightweight, replaceable client. A webhook can replace it
without code changes.
- `systemd/diploid-agent-run.sh` runs both under one unit and restarts them as a
pair if either dies.

## Why spool on Hindsight retain failures

Hindsight is a local NAS Docker container. It can restart, run out of shared
memory, or be temporarily unreachable. The harness must not drop turns or block
the conversation. Local spooling + lazy replay gives durable, non-blocking
retention.

## Why `n_turns_summarization` only applies to the file backend

Hindsight performs its own consolidation and observation generation. Sending
client-side summaries to Hindsight would fight the extractor. Therefore, the
file backend summarization loop is disabled when `backend: hindsight`.

## Why the prompt is split into `first` vs `follow-up`

- **First prompt** (new session / model switch): full persona + memory + user
  message. This is the heavy context load.
- **Follow-up prompt** (resume): short identity anchor + user message. The ACP
  session already holds prior context.

This keeps follow-up messages small while still allowing complete context
re-injection on session boundaries.

## Why reply-to quotes are capped

A Telegram message can be up to 4096 characters. Quoting the full original
message plus the user's reply can push the prompt over a useful context budget.
The harness caps the quoted text at `max_reply_quote_chars` (default 2048),
trims it on a section boundary, and inserts a truncation marker. This gives the
agent enough context to understand the reference while leaving room for the
user's new message, the persona, and the chat memory.

## Why max memory is in characters, not tokens

The harness does not know which tokenizer the active model uses. Characters is a
rough but backend-agnostic cap. For Hindsight, `max_recall_tokens` is still sent
to the server because Hindsight itself budgets in tokens.

## Why `chat_memory_exceeded` is split from prompt truncation

The session record's `chat_memory_exceeded` flag used to be set whenever the
prompt's chat memory block was trimmed. That conflated two different events:

- The on-disk `MEMORY.md` file has grown beyond `max_chat_memory_chars`.
- The *prompt context* (short-term transcript + recalled content) had to be
trimmed to fit the model's context window.

The record now tracks only the file-size event. Prompt truncation is still
reported in the `## System notice`, but it is no longer treated as a persistent
"file exceeded" state. This keeps the Telegram transition warning accurate when
the memory file crosses the cap.

## Why short-term context can be summarized on demand

The last `short_term_turns` of raw conversation are the most valuable context,
but in a long conversation they can be the largest part of the prompt. The
`smart` short-term strategy keeps a configurable minimum number of raw turns and
summarizes the older short-term turns into a compact bullet list. This protects
the prompt budget without losing the most recent turns and without requiring the
agent to manually prune the transcript.

## Why MCP servers are configured in the harness, not in the ACP wrapper

The ACP `session/new` call accepts a `mcpServers` list, but which servers are
active is a chat-level policy, not a transport detail. The harness keeps the
configured server definitions and the per-chat enabled set, and injects the right
list at session creation time. This lets Telegram and HTTP callers enable or
disable servers per chat without restarting the engine process.

## Why stale-session rehydration rebuilds the full prompt

When an ACP session becomes stale (`_is_stale_session_error`), the harness falls
back to `_build_first_prompt` and starts a fresh session. The new first prompt
re-injects the persona, current memory, recalled transcript context, and the
optional continuation anchor. This is a full context reload, not an incremental
re-injection. It is a known cost of the stale-session fallback and is
intentionally chosen over trying to resume a broken ACP session.

## Why skills are files synced into the chat working directory

The active ACP process discovers skills from `cwd/.devin/skills/*/SKILL.md` at
session start.
The harness therefore owns skill *selection*: it reads persona, shared, and
chat-scoped skill files, computes the enabled set for the chat, and copies the
effective skills into the session's `cwd` before `session/new`. This keeps the
skill store in the filesystem, versionable, and persona-specific while still
allowing per-chat overrides and runtime creation.

The harness uses a compact skill index in the prompt.  Full skill content is
no longer injected there; only skills whose triggers match the user message
(or slash commands like `/{skill-name}`) are copied into the session's
`.devin/skills/` and loaded by the ACP process.  When the active skill set
changes mid-conversation the harness starts a fresh ACP session so the new
skills are discovered.
