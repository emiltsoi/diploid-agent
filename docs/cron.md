# Cron — declarative scheduled and triggered jobs

Recurring (and one-shot) jobs defined in config files, run by a background
`CronService` — without spending a conversational turn. Complements the
[wake queue](wake.md): wakes are one-shot events that start a chat turn;
cron jobs are declared schedules — or declared event triggers — that run
as tasks or phantoms.

## Files

- **Global**: `config/crons.yaml` (or `harness.cron.global_file`) —
  operator-owned jobs, ungated.
- **Persona**: `<persona profile_root>/crons.yaml` — agent-authored jobs,
  gated by the authorship `cron_enabled` toggle and per-persona budgets.

Both files are watched by mtime every tick (default 5s) and re-parsed on
change. A parse error keeps the file's last-good job set and surfaces a
warning — the tick never crashes on a bad edit.

## Job schema

```yaml
jobs:
  - id: memory-tidy            # slug, unique across merged files
    enabled: true
    schedule:
      cron: "0 */6 * * *"      # 5-field cron (croniter), OR
      # every_seconds: 21600   # plain interval, OR
      # at_daily: "06:30"      # named local wall-clock time
    # — OR —
    trigger:
      type: file               # file | body
      path: "session:jobs/inbox.flag"   # file only — confined path
      # field: "energy"        # body only — chat_body_state.json key
      # op: ">"                # body only — > >= < <= == !=
      # value: 25              # body only — number, bool, or string
      # cooldown_seconds: 900  # default: min_interval_seconds
    call:
      type: llm                # llm | script
      persona: vesper          # llm only — whose files the phantom loads
      prompt: "…"              # llm only
      model: null              # llm only — optional override
      command: "scripts/x.sh"  # script only
      cwd: "~"                 # default: persona dir
      timeout_seconds: 300
    delivery: silent           # silent | digest | turn
    chat_id: "7945905361"      # owning chat; defaults to mesh fallback
    overlap: skip              # skip | queue
    catchup: once              # once | skip — fire once on boot if missed
    max_consecutive_failures: 3  # then auto-disable + forced digest entry
```

Exactly one **driver** is required — `schedule` or `trigger`, never both.
Exactly one schedule field is required inside `schedule:`. Jobs whose
interval is below `min_interval_seconds` (including cron expressions that
fire too often, or a `cooldown_seconds` under the floor) are dropped with
a warning. Duplicate ids across files conflict — the first-seen wins and
a warning names both files.

## Triggers

A `trigger:` job is event-driven rather than time-driven. It shares the
same registry, state file, overlap policy, failure/auto-disable handling,
`run_now` door, and all three delivery modes as scheduled jobs — only the
firing condition differs.

- **`file`** watches a path's mtime. The first observation *adopts* the
  current mtime without firing — a pre-existing file is not a change. A
  later change fires once if the cooldown has elapsed; a change observed
  inside the window stays pending and collapses a burst into one fire.
  A missing file is simply "no observation" — deletion is not an edge,
  and recreation counts as the next change.
- **`body`** reads the owning chat's `chat_body_state.json` and evaluates
  `field op value`. It fires only on a **false→true edge**: a held-true
  condition does not refire, and the condition clearing re-arms it. An
  edge inside the cooldown window stays pending and fires once the window
  elapses if the condition still holds. Missing file, missing field,
  malformed JSON, and incompatible value types all evaluate false —
  safely, without crashing the tick.

`cooldown_seconds` bounds refires (default `min_interval_seconds`);
explicit values below `min_interval_seconds` drop the job, because a
trigger under the floor is a loop wearing a watch. The cooldown gates
edge *consumption*, not just firing — an edge observed inside the window
is left pending, so a queued re-fire (`overlap: queue`) can never chain
fire-on-completion in defiance of the floor.

### File trigger path confinement

Persona-authored jobs may not watch arbitrary paths:

- `session:<rel>` resolves under the owning chat's session directory.
- Other relative paths resolve under the persona's `profile_root`.
- Absolute paths must land under one of the allowed roots.
- Operator-global files may additionally reach under `$HOME`.
- `harness.cron.trigger_allowed_roots` (list of paths) opens extra roots
  to **both** persona and global jobs — the operator's door for shared
  spaces outside the persona/session confinement, e.g. a common-room
  directory on a shared mount:

  ```yaml
  harness:
    cron:
      trigger_allowed_roots:
        - /nas/emiltsoi/Agents/vault
  ```

The resolved path is checked after `.resolve()` — symlinks and `..`
cannot escape the roots. A job whose path escapes is dropped with a
warning at merge time.

### Trigger state

Trigger observations persist on the job's `cron_state.jsonl` row
(`trigger_seen_mtime`, `trigger_fired_at`, `trigger_held`) so cooldowns
and edge state survive restarts. A hot edit to the `trigger:` block
re-bootstraps the state — the first observation of the new spec adopts
without firing — but `trigger_fired_at` deliberately survives the reset,
so an edit cannot buy a fire inside the previous cooldown window.
Converting a job between `schedule:` and `trigger:` is a hot edit like
any other. `POST /cron/<id>/run` fires a trigger job without consuming
its cooldown or edge state.

## Call types

- **`script`** — subprocess via the task engine's shell path
  (`shlex`-parsed, `shell=False`), cwd defaulting to the persona dir.
  Execution timeout follows `harness.task.shell_timeout`;
  `timeout_seconds` is validated against `max_script_timeout_seconds` at
  load.
- **`llm`** — a *phantom*: a fresh isolated ACP child per run (the
  task engine's `_run_acp` substrate). The prompt composes the persona's
  SOUL + AGENTS, persona `MEMORY.md`, and the owning chat's promoted
  pocket — each char-capped — plus the job instruction and an output
  contract (`<chat>/cron/<job>.last.md`, ≤3-line reply). Phantoms get an
  empty MCP list: persona file tools only — no mesh send, no self-wake
  arming. Never touches the chat's session or transcript.

## Delivery

Every mode appends `sessions/<chat>/cron/<job>.log` and writes
`<job>.last` (`{job_id, delivery, status, finished_at, next_due_at,
consecutive_failures, summary}`). The digest plugin filters on
`delivery`: `silent` entries never render, except that an auto-`disabled`
job always surfaces — silence must not hide a broken job.

- **`silent`** — files only; nothing enters prompts.
- **`digest`** — files plus a `cron` prompt slot (the
  `diploid_plugins.cron` plugin, `prompt_slot="cron"`) renders bounded
  last-run lines: `job … ok 2h ago — summary`. Parse/budget warnings land
  in `cron/service.last` and render as `! service:` lines.
- **`turn`** — files plus a wake (`reason=cron:<id>`,
  `payload.user_message` = status + summary) through the normal wake path:
  the result opens a real turn and the agent responds with judgment.
  Turn deliveries share the self-wake budgets (`self_wake_max_pending`,
  `self_wake_min_interval_seconds`): a full pending queue or an exhausted
  `turn_delivery_max_per_day` cap degrades the delivery to the files —
  `delivery_result` in `.last` records `turn_suppressed: <why>` — while a
  recent arm merely defers the wake (`turn_deferred: <n>s`). The `.last`
  payload's `delivery_result` field records the outcome for every mode.

## Hot edits

Files reload on mtime mid-run, so the service defines who owns a run:

- **A run belongs to the spec that fired it.** The firing spec and chat are
  persisted on the state row at materialize time (`fired_spec`,
  `fired_chat_id`), so a delivery, prompt, or chat edit mid-run does not
  reroute a result that is already in flight — and the rule survives a
  restart through `_reconcile_running`.
- **A schedule edit reseeds the next fire.** Idle jobs reseed on the
  reload tick; a job edited while running reseeds from the new spec at
  finalize (the in-flight slot is left alone).
- **A deleted job still lands its result** under the firing spec, then its
  state row is dropped. A queued re-fire (`overlap: queue`) is not owed to
  a deleted job.
- `enabled: false` edits stop future fires; the in-flight run completes.

## Scheduling semantics

- `CronService` ticks on `harness.cron.tick_seconds` (default 5).
- `cron` and `at_daily` schedules are local wall-clock time (system-cron
  convention — croniter is fed a tz-aware local base, so DST shifts land
  on the named hour); `every_seconds` is plain elapsed time.
- `min_interval_seconds` is checked against the *minimum* gap across the
  next several fires of a cron expression, so a dense sub-pattern hiding
  behind a sparse upcoming gap is still caught.
- Brand-new jobs seed `next_due_at` forward — no boot storm.
- A job past due at service start honors `catchup`: `once` fires a single
  tagged run; `skip` advances the schedule.
- `overlap: skip` records a `skipped` status and advances; `queue` owes
  one fire when the current run finishes.
- A failure moves to the next slot — no retry storm. At
  `max_consecutive_failures` the job auto-disables and forces a digest
  entry regardless of delivery mode.

## Budgets (`harness.cron.*`)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | master switch for the service |
| `tick_seconds` | `5` | scheduler poll interval |
| `max_jobs_per_persona` | `8` | cap on persona-authored jobs |
| `max_jobs_global` | `16` | cap on operator jobs |
| `min_interval_seconds` | `300` | minimum fire interval per job |
| `max_llm_timeout_seconds` | `600` | phantom turn cap |
| `max_script_timeout_seconds` | `300` | script job validation ceiling |
| `turn_delivery_max_per_day` | `4` | daily cap on turn deliveries |
| `digest_max_jobs` | `8` | lines the digest slot renders |

Persona-authored files only load while the persona's authorship
`cron_enabled` toggle is on — the same posture as `self_wake_enabled`.

## Inspection

`GET /cron` returns the merged job list plus per-job state
(`next_due_at`, `last_status`, `consecutive_failures`, `running`,
`auto_disabled`, `turns_today`) and the current reload warnings. Trigger
jobs carry `trigger` (the spec) and `trigger_state` (`seen_mtime`,
`fired_at`, `held`); `schedule`/`next_due_at` are null for them.

`POST /cron/<id>/run` fires a job immediately — the operator door for
testing and recovery. It requires the API key like every mutating route,
ignores `enabled` and auto-`disabled` (a successful manual run re-enables
the job), refuses with `409` while a run is in flight, and does not
consume the schedule — `next_due_at` is left untouched.

The `harness_cron_list` `diploid-harness` MCP tool renders `GET /cron`
for the agent: job ids, schedules, next due, last status, `running`, and
`turns_today`. Manual firing stays operator-only — there is deliberately
no `harness_cron_run` tool.
