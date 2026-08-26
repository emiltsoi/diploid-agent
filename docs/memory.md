# Memory loop and Hindsight

The harness has two memory scopes:

1. **Per-chat memory** — turns and summaries tied to one `chat_id`.
2. **Persona memory** — global facts promoted across all chats of a persona.

The active per-chat memory backend is chosen by `harness.memory.backend`:
`file` or `hindsight`.

## Core principle: the agent has agency

The harness does **not** mechanically prune memory files. It only enforces a
**context budget** on what is loaded into the first turn of a session. If a
memory file is larger than the budget, a `## System notice` is injected into the
prompt and a system message is sent to the user. The agent then decides whether
to read the file and rewrite it, using its own file tools.

## Shared behavior

### Transcript

Every user/assistant pair is written to:

```
sessions/<chat_id>/chat_transcript.jsonl
```

as two JSON lines. This file is the durable chat ledger and is always
written, regardless of backend. It is protected from `/new`, `/resume`, and
`/branch` so the conversation history survives ACP session changes.

### First turn / model switch

When a new Devin session starts, the prompt is built as:

```
<persona identity files>

## System notice        (only if a memory file is larger than its load cap)
...

## Current memory (MEMORY.md)
<first N chars of persona MEMORY.md>

## Chat memory
<short-term transcript + first N chars of recalled long-term memory>

<new user message>
```

Existing sessions do **not** re-inject the memory block; the existing Devin
session already holds it.

### System notice

If either `## Current memory` or `## Chat memory` was truncated, the first turn
contains:

```markdown
## System notice

The following memory files are larger than the context budget and were
partially loaded. Older content is still saved on disk but is not shown in this
prompt.

- Persona memory (.../MEMORY.md) was trimmed to {loaded} of {total} characters
  (limit: {limit}).
- Chat memory (.../MEMORY.md) was trimmed to {loaded} of {total} characters
  (limit: {limit}).

You may use your file tools to read the full files and, if you choose, rewrite
them. Preserve user-promoted facts unless the user explicitly says otherwise. If
you edit a memory file, tell the user what changed.
```

A shorter version is also sent to the user in Telegram:

```
System: Chat/persona memory was trimmed to fit the context budget. Older content
is still saved. You can ask me to read and prune it.
```

### In-session transitions

If a memory file was under its cap and then exceeds it during a session — for
example because `/summarize` wrote a new summary block or `/promote` appended a
persona fact — a single system message is sent at that point. No more warnings
are sent until the file is pruned back below the cap and then grows again.

## File backend

Configuration:

```yaml
harness:
  memory:
    backend: file
    n_turns_summarization: 10      # null to disable
    max_chat_memory_chars: 16384
    max_persona_memory_chars: 16384
    max_reply_quote_chars: 2048    # caps quoted reply-to text from Telegram
    short_term_strategy: smart     # raw | smart
    short_term_turns: 10           # max raw turns to load
    min_short_term_turns: 2        # always keep this many raw turns
    max_short_term_chars: 6144     # cap the short-term block
    include_short_term: true
    short_term_summary_cache_days: 7  # stale .cache/*.md entries are removed after this many days
```

The active `config/harness.yaml` (and `config/harness.yaml.example`) use
`max_chat_memory_chars: 16384` and `max_short_term_chars: 6144` to give long
conversations more room. The `MemoryConfig` defaults are lower (`8192` and
`4096`). You can tune them live with `/config memory max_chat_memory_chars=...`
or by editing `config/harness.yaml` and restarting.

### Summarization

If `n_turns_summarization` is set, every N turns the harness asks the Devin
agent to summarize the last N turns. The resulting summary is appended to:

```
sessions/<chat_id>/chat_MEMORY.md
```

The file is never pruned automatically. It can grow as large as the agent or the
user allows.

### Load cap

The first `max_chat_memory_chars` of the file are loaded. The load order is
**first N**: after multiple pruning cycles, the oldest, most distilled facts are
considered the most important. The short-term transcript is always appended
after the long-term recall so the most recent conversation is still visible.

### Short-term context

When `short_term_strategy` is `raw`, the last `short_term_turns` user/assistant
pairs are included verbatim. This can push the prompt over `max_chat_memory_chars`
if the recent turns are long.

When `short_term_strategy` is `smart` (recommended), the block is capped at
`max_short_term_chars`:

- If the raw short-term window fits, it is included verbatim.
- If it does not fit, the most recent `min_short_term_turns` are kept raw and
  the older short-term turns are summarized into a compact bullet list.
- If even the minimum raw turns exceed the cap, the raw block is truncated and
  a marker is added.

This keeps the most recent conversation intact while preventing a long
short-term transcript from squeezing out the long-term recall.

### Reply-to quote cap

When a Telegram message is a reply to an earlier message, the quoted text is
injected into the prompt with a clear label. The quote is capped at
`max_reply_quote_chars` (default 2048) and a truncation marker is added if it
exceeds that limit, so a long quoted message cannot bloat the prompt.

### Recall

`FileMemoryBackend.recall` performs a simple keyword search over
`chat_transcript.jsonl` and `chat_MEMORY.md`, returning the best matches. The
result is then capped by `MemoryManager` so it does not push the short-term
context out of the budget.

## Hindsight backend

Configuration:

```yaml
harness:
  memory:
    backend: hindsight
    max_chat_memory_chars: 16384
    short_term_strategy: smart
    short_term_turns: 10
    min_short_term_turns: 2
    max_short_term_chars: 6144
    include_short_term: true
    hindsight:
      base_url: http://localhost:8888
      bank: example
      api_key: null
      timeout: 30.0
      max_recall_tokens: 1500
      recall_min_scores:
        semantic: 0.25
        reranker: 0.5
      prefer_observations: true
      async_writes: true
      fallback_to_file: true
```

### Bank design

One bank per persona (`example`), with chat separation via the
`chat:<chat_id>` tag.

### Retain

Every turn is written to:

```
POST /v1/default/banks/<bank>/memories
```

with `tags: ["turn", "chat:<chat_id>", "session:<session_number>", "persona:<persona_name>"]`. Failed writes are spooled locally and
retried. The server auto-consolidates facts and observations; the harness does
not pre-summarize.

### Recall

On a new session, the harness calls:

```
POST /v1/default/banks/<bank>/memories/recall
```

with `query`, `tags`, `max_tokens`, `types: ["world", "experience", "observation"]`,
and `prefer_observations`. The result is capped to the remaining chat memory
budget (after the short-term transcript is reserved) and the agent is warned if
the cap was hit.

Hindsight is the long-term memory layer. It is not pruned by the harness.

## Persona memory promotion

`/promote <fact>` appends a bullet to:

```
personas/<persona>/MEMORY.md
```

When the backend is Hindsight, the fact is also retained as a `memory` item
with `persona` and `promoted` tags, so it can be recalled across chats.

There is no automatic pruning. The first `max_persona_memory_chars` are loaded
into the first turn of each session, and a `## System notice` is injected if
the file exceeds that cap.

## Commands

| Telegram | HTTP | Effect |
|---|---|---|
| `/memory` | `GET /memory/{chat_id}` | Show per-chat memory. |
| `/summarize` | `POST /summarize/{chat_id}` | Trigger file-backend summarization. |
| `/recall <query>` | `POST /recall` | Search the memory backend. |
| `/promote <fact>` | `POST /promote` | Append a fact to the persona's global memory. |
| — | `POST /retain` | Retain an observation in the active memory backend. |

There is **no `/prune` command**. The agent handles pruning itself using file
tools when it sees a `## System notice`.

## Failure modes

| Failure | Behavior |
|---|---|
| Hindsight unreachable on retain | Turn is spooled locally; retry on next turn. |
| Hindsight unreachable on recall | Fall back to local file keyword search if enabled. |
| Memory file exceeds load cap | `## System notice` in prompt + Telegram system message; no auto-prune. |
| Summarization fails | Warning logged; no `MEMORY.md` update. |

## Agent-facing memory tools

The `diploid-memory` MCP server gives the agent three memory tools:

- `memory_recall(query, tags, max_tokens)` — search prior turns and retained facts.
- `memory_retain(content, tags, context)` — save an observation to the chat ledger.
- `memory_promote(fact)` — promote a fact to the persona's `MEMORY.md` and Hindsight.

The `memory` shared skill triggers these tools when the user says things like "remember that", "what did we", or "promote to memory".

The MCP server is configured in `harness.mcp.servers` and uses the `{harness_url}` placeholder so each ACP session talks to the harness it was launched from.
