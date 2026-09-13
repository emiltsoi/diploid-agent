# Cron — declarative scheduled jobs

Recurring (and one-shot) jobs defined in config files, run by a background
`CronService` — without spending a conversational turn. Complements the
[wake queue](wake.md): wakes are one-shot events that start a chat turn;
cron jobs are declared schedules that run as tasks or phantoms.

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
    call:
      type: llm                # llm | script
      persona: vesper          # llm only — whose files the phantom loads
      prompt: "…"              # llm only
      model: null              # llm only — optional override
      command: "scripts/x.sh"  # script only
      cwd: "~"                 # default: persona dir
      timeout_seconds: 300
    delivery: silent           # silent | digest (turn arrives in Wave B)
    chat_id: "7945905361"      # owning chat; defaults to mesh fallback
    overlap: skip              # skip | queue
    catchup: once              # once | skip — fire once on boot if missed
    max_consecutive_failures: 3  # then auto-disable + forced digest entry
```

Exactly one schedule field is required. Jobs whose interval is below
`min_interval_seconds` (including cron expressions that fire too often)
are dropped with a warning. Duplicate ids across files conflict — the
first-seen wins and a warning names both files.

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
- **`turn`** — reserved for Wave B; rejected at validation today.

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
| `turn_delivery_max_per_day` | `4` | Wave B budget (reserved) |
| `digest_max_jobs` | `8` | lines the digest slot renders |

Persona-authored files only load while the persona's authorship
`cron_enabled` toggle is on — the same posture as `self_wake_enabled`.

## Inspection

`GET /cron` returns the merged job list plus per-job state
(`next_due_at`, `last_status`, `consecutive_failures`, `running`,
`auto_disabled`) and the current reload warnings.
