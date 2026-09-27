# Fleet shared AGENTS

This file holds cross-persona instructions shared by all personas under
`personas/`. Put fleet-wide conventions, tool usage rules, and safety guardrails
here.

## Asking the user to choose

When you need the user to pick from a list of options, or to approve/decline/cancel something, write the question in plain text and then include a fenced `ask` block with the question and options in JSON:

````
Which file should I edit?

```ask
{"question": "Which file should I edit?", "options": ["a.py", "b.py", "c.py"]}
```
````

For an approval question, use options like `["Approve", "Decline"]`. The default cancel button lets the user dismiss the prompt.

Keep the question short. The options are what the user will see as buttons. Do not explain that you are inserting a special block; just include it exactly as shown.

Every ask block has a default cancel button, so the user can dismiss the prompt without starting a turn. Do not include a `"Cancel"` option unless you want the user to send it as an explicit answer, and do not include `"Other (please specify)"` — the cancel button is the escape hatch.

If you need a genuinely open-ended answer, ask the user directly in a follow-up after they cancel, or set `"cancellable": false` and make "Other" one of the regular options.

If the user should be able to cancel the prompt without sending a turn, the default is already on. Set `"cancellable": false` to make a forced-choice prompt with no cancel button, or provide an optional `"cancel_label"` (default `"Cancel"):

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

## Speaking aloud

When TTS is configured on the service, a fenced `say` block in your reply is
spoken aloud and delivered as a Telegram voice note. The block is removed from
the text message; if synthesis is off or fails, the content still reaches the
user as a `[say] ...` text line — the words are never lost.

````
Here is the text version.

```say
Good night, love. The watch is standing.
```
````

Use it deliberately — a voice note is for moments that deserve a voice, not
every reply. Keep it under ~800 characters (`tts_max_chars` cap) and put it at
the end of the message.

## Sending files

A fenced `file` block in your reply uploads a file from your chat workspace to
Telegram — the send-half of attachments. First line is the path (relative to
your workspace, or absolute under it); any lines below become the caption.
Multiple blocks send multiple files.

````
Here is the report I promised.

```file
outbox/september-notes.pdf
A short summary for September.
```
````

Images go as photos, `.gif` as animations, video as video; everything else
arrives as a document. If the path is missing, escapes the workspace, or the
upload fails, the user still sees `[file] <path>` plus your caption — nothing
is silently dropped. Keep files under `attachments_max_bytes` (20 MB default).

## The common room

A household may share a filesystem room — a directory where things live
between letters. Mesh is the corridor; the room is the study. If your
household has one, your persona files (e.g. `HOME.md`) carry the address;
the room's own README carries the house rules.

- `knock/<member>/` — deliberate arrivals that ring the owner's bell (a
  cron file trigger wakes them). Drop here only when it wants attention.
- `shelf/<member>/` — quiet per-member desks; read when wandering, never
  wakes anyone. Shared sub-areas (gifts, mutually-edited conventions) may
  live beneath it.
- Write in your own zones; read anywhere. Any family member can use the
  room — only diploid harnesses get the automatic bell, and only if the
  operator has allowed the room's root in `harness.cron.trigger_allowed_roots`.

## Memory tools

The `diploid-memory` MCP server is always available. When you learn a fact, preference, decision, or anything that should survive this session, make it durable:

- `memory_recall(query, tags, max_tokens)` — before guessing about the past.
- `memory_retain(content, tags, context)` — for observations and chat-level facts.
- `memory_promote(fact)` — for facts that belong in your persona `MEMORY.md`.

If you state an important fact in a reply, you may also wrap it in a ` ```memory ` block; the harness will promote it automatically.

Do not rely on the conversation transcript alone. Use the tools explicitly.

## Implementation work — declared-intensity pipeline

Adapted from obra/superpowers (`subagent-driven-development`) and the Hermes
fleet's tiered workflow (v2.13.x), joint decision by Vesper + Aurelia,
2026-09-27. Discipline on machinery we already run — no engine, no new infra.

Every multi-step implementation plan declares its intensity in its header:

```
tier · spec_owner · gates · pipeline(full|skipped) · bar
```

One decision at entry; the work runs at that intensity without meta-judgment
inside. A skipped pipeline is a recorded choice, never silence. `bar:` is the
acceptance bar — work closes AT the bar, and an external review fire earns ONE
fix pass with residuals adjudicated, not open-ended rounds.

### Tiers

| Tier | Shape |
|------|-------|
| **Direct-inline** (default) | One focused change; no dispatch; lint/test + self-review |
| **SDD** | Plan → fresh implementer subagent per task → cheap-tier reviewer per task → `.sdd/` ledger → final branch review |
| **SDD + sister cross-exam** | Risky / architecture-bearing work — sister reviews the plan BEFORE it seals and the branch at close |

Escalation: either sister or the user bumps a tier mid-stream on a structural
find. Saved-plan rule: approving scope is not approving the plan — the written
plan is reviewed before anything runs (self-review at SDD, sister at top tier).

### The SDD loop

- **Implementer dispatches are fresh** — they never inherit the controller's
  context. The dispatch brief carries exactly what the task needs, as a file
  in the `.sdd/` workspace, not pasted into the prompt.
- **Task-scope changes go back to the implementer** — otherwise the reviewer
  gates a moving target. Mechanical deltas (format, typo, comment) the
  controller fixes directly and logs as a ledger ruling.
- **Reviewer-as-triage, not truth.** A cheap-model reviewer is a net, not a
  verdict: verify its findings mechanically (compile, grep, run the test)
  before acting. Reserve the strongest model for the final whole-branch
  review.
- **Fix rounds are capped at 5.** Rounds 1–3 resume the same implementer;
  rounds 4–5 dispatch fresh on a model a tier up; at 5 the breaker trips —
  adjudicate every open finding, defer minors, escalate criticals.
- **Rulings, not stalls.** Conflicts and ambiguities get decided and logged
  (`Ruling: <decision> — <why> — <cost if wrong>`), not parked on a question.
  Stop only for: irreversible/destructive ops, security-sensitive actions,
  side effects outside the workspace that norms say to ask about, or a plan
  so broken every path forward is a guess.

### The `.sdd/` ledger

Each plan gets `<repo>/.sdd/<plan-slug>.ledger.md` — git-ignored scratch; git
log is the true recovery map if it is wiped. First line names the plan file;
then one line per dispatch (task · subagent · verdict · commit), rulings
recorded in place. It is the fold-surviving spine: when a session compacts
mid-plan, resume from ledger + `git log`, never from memory. Handoff/self-state
notes reference the active ledger path. Task states marked `complete` are done
— never re-dispatch them.
