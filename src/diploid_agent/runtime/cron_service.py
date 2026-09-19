"""Declarative cron scheduler: config-file jobs materialized as tasks.

Rides its own daemon tick (``harness.cron.tick_seconds``), reloads
``crons.yaml`` files on mtime change, keeps per-job state in
``cron_state.jsonl``, and materializes due jobs directly into the
``TaskEngine`` through a standing ``__cron__`` plan per chat — status and
history come free, with no user-visible plan ceremony.

LLM jobs run as *phantoms*: a fresh isolated ACP child per run with the
persona's identity files, persona memory, and the owning chat's promoted
pocket composed into the prompt — never the chat session or transcript.
Phantoms get an empty MCP list: persona file tools only, no mesh send, no
self-wake arming (no recursive scheduling).
"""

from __future__ import annotations

import json
import logging
import operator
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from diploid_agent.config import (
    AuthorshipConfig,
    Config,
    CronJobSpec,
    PersonaConfig,
)
from diploid_agent.memory_promoted import PromotedMemory
from diploid_agent.models import WakeEvent
from diploid_agent.persona_composer import _trim_to_section, compose_persona
from diploid_agent.plan.models import Task, TaskStatus, TaskType
from diploid_agent.runtime.cron_state import CronJobState, CronStateStore
from diploid_agent.runtime.event_bus import Event
from diploid_agent.runtime.wake_queue import WAKE_BUDGET_REASON_PREFIXES

if TYPE_CHECKING:
    from diploid_agent.plan.manager import PlanManager
    from diploid_agent.runtime.event_bus import EventBus
    from diploid_agent.runtime.wake_queue import WakeQueue
    from diploid_agent.task.engine import TaskEngine

try:
    from croniter import croniter
except ImportError:  # pragma: no cover - dependency guard
    croniter = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CRON_PLAN_NAME = "__cron__"
CRON_WAKE_REASON_PREFIX = "cron:"
# Body-trigger comparison operators; the value column of
# chat_body_state.json is free-form, so TypeError means "no match".
_TRIGGER_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}
_SESSION_TRIGGER_PREFIX = "session:"
_BODY_STATE_FILENAME = "chat_body_state.json"
_PERSONA_BODY_STATE_FILENAME = "persona_body_state.json"


@dataclass
class _ResolvedJob:
    """A job spec resolved against this service's persona and budgets."""

    spec: CronJobSpec
    source_file: Path
    source_label: str  # "global" | "persona"
    chat_id: str
    persona: PersonaConfig | None  # resolved identity for llm calls
    persona_dir: Path | None  # cwd default for script + llm calls


@dataclass
class _CronFileWatch:
    path: Path
    label: str
    mtime: float | None = None
    jobs: list[CronJobSpec] = field(default_factory=list)
    error: str | None = None


class CronService:
    """Background thread that fires declarative cron jobs as tasks."""

    def __init__(
        self,
        *,
        config: Config,
        plan_manager: PlanManager,
        task_engine: TaskEngine,
        event_bus: EventBus,
        sessions_root: Path,
        wake_queue: WakeQueue | None = None,
    ) -> None:
        self._config = config
        self._plan_manager = plan_manager
        self._task_engine = task_engine
        self._event_bus = event_bus
        self._wake_queue = wake_queue
        self._sessions_root = Path(sessions_root).expanduser()
        cron_cfg = config.harness.cron
        self._state = CronStateStore(cron_cfg.state_path)
        self._files: list[_CronFileWatch] = []
        self._jobs: dict[str, _ResolvedJob] = {}
        self._warnings: list[str] = []
        self._cron_plans: dict[str, str] = {}  # chat_id -> plan_id
        self._task_to_job: dict[str, str] = {}  # task_id -> job_id
        self._catchup_pending: set[str] = set()
        self._finalize_lock = threading.Lock()
        self._booted = False
        self._thread: threading.Thread | None = None
        self._running = False
        self._watches: list[_CronFileWatch] = self._build_watches()
        self._reload(force=True)

    # ------------------------------------------------------------------ config

    def _build_watches(self) -> list[_CronFileWatch]:
        cron_cfg = self._config.harness.cron
        watches: list[_CronFileWatch] = []
        if cron_cfg.global_file is not None:
            watches.append(_CronFileWatch(path=cron_cfg.global_file, label="global"))
        persona = self._config.persona
        if persona is not None:
            watches.append(
                _CronFileWatch(
                    path=persona.profile_root / cron_cfg.persona_filename,
                    label="persona",
                )
            )
        return watches

    def _persona_cron_permitted(self) -> bool:
        """The authorship ``cron_enabled`` toggle gates persona-authored jobs."""
        for plugin in self._config.harness.plugins:
            if plugin.name == "authorship" and plugin.enabled:
                try:
                    auth = AuthorshipConfig.model_validate(plugin.config or {})
                except Exception:  # noqa: BLE001
                    return False
                return auth.cron_enabled
        return False

    def _resolve_persona(self, name: str | None) -> PersonaConfig | None:
        persona = self._config.persona
        if persona is None:
            return None
        if name in (None, "", persona.name):
            return persona
        fleet_root = persona.fleet_root or persona.profile_root.parent
        root = fleet_root / name
        if (root / "SOUL.md").exists():
            return PersonaConfig(
                name=name,
                profile_root=root,
                fleet_root=fleet_root,
                memory_filename=persona.memory_filename,
            )
        return None

    # ------------------------------------------------------------------ reload

    def _reload(self, force: bool = False) -> None:
        changed = False
        for watch in self._watches:
            try:
                mtime = watch.path.stat().st_mtime
            except OSError:
                mtime = None
            if not force and mtime == watch.mtime:
                continue
            watch.mtime = mtime
            changed = True
            if mtime is None:
                watch.jobs = []
                watch.error = None
                continue
            try:
                raw = yaml.safe_load(watch.path.read_text()) or {}
                parsed = [CronJobSpec.model_validate(j) for j in raw.get("jobs", [])]
            except Exception as exc:  # noqa: BLE001
                # Keep the last-good job set for this file.
                watch.error = f"{watch.path}: parse error ({exc}); last-good kept"
                logger.warning("Cron file %s failed to parse; keeping last-good", watch.path)
                continue
            watch.jobs = parsed
            watch.error = None
        if changed or force:
            self._merge_jobs()

    def _merge_jobs(self) -> None:
        cron_cfg = self._config.harness.cron
        warnings: list[str] = []
        merged: dict[str, _ResolvedJob] = {}
        now = time.time()
        persona_permitted = self._persona_cron_permitted()

        for watch in self._watches:
            if watch.error:
                warnings.append(watch.error)
            if watch.label == "persona" and not persona_permitted:
                if watch.jobs:
                    warnings.append(f"{watch.path}: ignored — authorship cron_enabled is off")
                continue
            cap = (
                cron_cfg.max_jobs_global
                if watch.label == "global"
                else cron_cfg.max_jobs_per_persona
            )
            kept = 0
            for spec in watch.jobs:
                if kept >= cap:
                    warnings.append(
                        f"{watch.path}: job {spec.id} dropped — over {watch.label} cap ({cap})"
                    )
                    continue
                if spec.id in merged:
                    warnings.append(
                        f"cron id conflict: {spec.id} in {watch.path} is shadowed "
                        f"by {merged[spec.id].source_file}"
                    )
                    continue
                resolved = self._resolve_job(spec, watch, warnings, now)
                if resolved is not None:
                    merged[spec.id] = resolved
                    kept += 1

        self._jobs = merged
        self._warnings = warnings
        self._state.remove_missing(set(merged))
        self._write_service_last()

    def _resolve_job(
        self,
        spec: CronJobSpec,
        watch: _CronFileWatch,
        warnings: list[str],
        now: float,
    ) -> _ResolvedJob | None:
        cron_cfg = self._config.harness.cron
        # Minimum-interval floor.
        if spec.schedule is not None:
            if spec.schedule.every_seconds is not None:
                if spec.schedule.every_seconds < cron_cfg.min_interval_seconds:
                    warnings.append(
                        f"job {spec.id}: every_seconds {spec.schedule.every_seconds:.0f} "
                        f"below min_interval {cron_cfg.min_interval_seconds:.0f} — dropped"
                    )
                    return None
            elif spec.schedule.cron is not None:
                gap = self._cron_gap(spec.schedule.cron, now)
                if gap is None:
                    warnings.append(f"job {spec.id}: invalid cron expression — dropped")
                    return None
                if gap < cron_cfg.min_interval_seconds:
                    warnings.append(
                        f"job {spec.id}: cron interval {gap:.0f}s below "
                        f"min_interval {cron_cfg.min_interval_seconds:.0f} — dropped"
                    )
                    return None
        elif spec.trigger is not None:
            # The trigger cooldown is the anti-loop floor for event-driven
            # jobs — below min_interval it is a loop wearing a watch.
            cooldown = spec.trigger.cooldown_seconds or cron_cfg.min_interval_seconds
            if cooldown < cron_cfg.min_interval_seconds:
                warnings.append(
                    f"job {spec.id}: trigger cooldown {cooldown:.0f}s below "
                    f"min_interval {cron_cfg.min_interval_seconds:.0f} — dropped"
                )
                return None
        # Resolve persona for llm calls and cwd defaults.
        persona = self._resolve_persona(spec.call.persona)
        if spec.call.type == "llm" and persona is None:
            warnings.append(f"job {spec.id}: persona {spec.call.persona!r} not found — dropped")
            return None
        persona_dir = persona.profile_root if persona is not None else None
        chat_id = spec.chat_id or self._config.harness.mesh.fallback_chat_id
        if not chat_id:
            warnings.append(f"job {spec.id}: no chat_id and no mesh fallback configured — dropped")
            return None
        resolved = _ResolvedJob(
            spec=spec,
            source_file=watch.path,
            source_label=watch.label,
            chat_id=chat_id,
            persona=persona,
            persona_dir=persona_dir,
        )
        if (
            spec.trigger is not None
            and spec.trigger.type == "file"
            and self._resolve_trigger_path(resolved, spec.trigger.path or "") is None
        ):
            warnings.append(
                f"job {spec.id}: trigger path {spec.trigger.path!r} escapes "
                "the allowed roots (persona dir / session dir) — dropped"
            )
            return None
        return resolved

    @staticmethod
    def _cron_gap(expr: str, now: float, samples: int = 5) -> float | None:
        """Minimum gap across the next several fires of a cron expression.

        Sampling only the first pair can miss a dense sub-pattern hiding
        behind a sparse upcoming gap, so we take the min over the next
        ``samples - 1`` inter-fire gaps.
        """
        if croniter is None:
            return None
        try:
            # A tz-aware local start keeps the expression in wall-clock
            # time — croniter treats float/naive input as UTC.
            it = croniter(expr, datetime.fromtimestamp(now).astimezone())
            prev = it.get_next(float)
            gaps: list[float] = []
            for _ in range(max(1, samples - 1)):
                cur = it.get_next(float)
                gaps.append(cur - prev)
                prev = cur
            return min(gaps)
        except (ValueError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------ service

    def start(self) -> None:
        if not self._config.harness.cron.enabled:
            return
        if self._running:
            return
        self._running = True
        self._event_bus.subscribe(self._on_event)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._event_bus.unsubscribe(self._on_event)
        except ValueError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._config.harness.cron.tick_seconds))
            self._thread = None

    @property
    def running(self) -> bool:
        return self._running

    def _run(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                logger.exception("Cron tick failed")
            time.sleep(self._config.harness.cron.tick_seconds)

    # -------------------------------------------------------------------- tick

    def _tick(self) -> None:
        if not self._config.harness.cron.enabled:
            return
        now = time.time()
        self._reload()
        if not self._booted:
            self._booted = True
            self._boot_pass(now)
        self._reconcile_running()
        for job_id, resolved in self._jobs.items():
            spec = resolved.spec
            if not spec.enabled:
                continue
            state = self._state.ensure(job_id, source_file=str(resolved.source_file))
            if state.disabled:
                continue
            if state.source_file != str(resolved.source_file):
                state.source_file = str(resolved.source_file)
            driver_key = self._driver_key(spec)
            if state.schedule_key != driver_key:
                if not state.schedule_key:
                    # First sight of this state row (or a row that predates
                    # the field): stamp the key without disturbing next_due
                    # or a pending catchup.
                    state.schedule_key = driver_key
                    self._state.update(state)
                elif state.running_task_id is None:
                    # Hot-edited driver on an idle job: reseed forward /
                    # re-bootstrap the trigger.
                    state.schedule_key = driver_key
                    if spec.trigger is not None:
                        self._reset_trigger(state)
                        self._state.update(state)
                        # Fall through: observe the new source this tick so
                        # the first sight adopts immediately.
                    else:
                        state.next_due_at = self._next_due(spec, now)
                        self._state.update(state)
                        self._catchup_pending.discard(job_id)
                        continue
                # else: drift while a run is in flight — leave the stale key
                # so _finalize reseeds from the new spec once it lands.
            if spec.trigger is not None:
                self._tick_trigger(resolved, state, now)
                continue
            if state.next_due_at is None:
                state.next_due_at = self._next_due(spec, now)
                self._state.update(state)
                continue
            if state.next_due_at > now:
                continue
            # Due.
            if state.running_task_id is not None:
                if spec.overlap == "queue":
                    state.queued_due = True
                else:
                    state.last_status = "skipped"
                self._state.update(state)
                continue
            self._materialize(resolved, state, now, catchup=job_id in self._catchup_pending)
            self._catchup_pending.discard(job_id)

    def _boot_pass(self, now: float) -> None:
        """Handle jobs whose due time passed while the service was down."""
        states = self._state.all()
        for job_id, resolved in self._jobs.items():
            state = states.get(job_id)
            if resolved.spec.trigger is not None:
                # Event-driven, not time-driven: nothing was "missed" while
                # down — the trigger re-observes its source on the next tick.
                continue
            if state is None or state.next_due_at is None:
                continue  # brand-new job — the tick seeds it forward
            if state.next_due_at >= now or state.disabled:
                continue
            if resolved.spec.catchup == "once":
                self._catchup_pending.add(job_id)  # tick fires it once, tagged
            else:
                state.next_due_at = self._next_due(resolved.spec, now)
                self._state.update(state)

    def _reconcile_running(self) -> None:
        """Re-attach to tasks still running, or finalize ones we missed."""
        for state in self._state.all().values():
            if state.running_task_id is None:
                continue
            task = self._plan_manager.get_task(state.running_plan_id or "", state.running_task_id)
            if task is None:
                state.running_task_id = None
                state.running_plan_id = None
                state.last_status = "failed"
                state.last_summary = "task record lost during restart"
                self._state.update(state)
                continue
            self._task_to_job[task.id] = state.job_id
            if task.status in (TaskStatus.DONE, TaskStatus.INCOMPLETE, TaskStatus.FAILED):
                self._finalize(state, task)

    # ------------------------------------------------------------ materialize

    def _next_due(self, spec: CronJobSpec, base: float) -> float:
        sched = spec.schedule
        assert sched is not None  # callers guard on spec.schedule
        if sched.every_seconds is not None:
            return base + sched.every_seconds
        if sched.at_daily is not None:
            hour, minute = (int(p) for p in sched.at_daily.split(":"))
            if croniter is not None:
                # croniter does wall-clock math, so 23h/25h DST days stay right.
                return croniter(
                    f"{minute} {hour} * * *",
                    datetime.fromtimestamp(base).astimezone(),
                ).get_next(float)
            lt = time.localtime(base)
            candidate = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hour, minute, 0, 0, 0, -1))
            if candidate <= base:
                candidate += 86400.0
            return candidate
        if sched.cron is not None and croniter is not None:
            return croniter(sched.cron, datetime.fromtimestamp(base).astimezone()).get_next(float)
        return base + 86400.0  # unreachable: validators require one field

    def _cron_plan_id(self, chat_id: str) -> str:
        plan_id = self._cron_plans.get(chat_id)
        if plan_id is not None and self._plan_manager.get_plan(plan_id) is not None:
            return plan_id
        for plan in self._plan_manager.list_plans(chat_id):
            if plan.name == CRON_PLAN_NAME:
                self._cron_plans[chat_id] = plan.id
                return plan.id
        plan = self._plan_manager.create_plan(
            name=CRON_PLAN_NAME,
            description="Standing plan for cron-scheduled tasks.",
            chat_id=chat_id,
        )
        self._cron_plans[chat_id] = plan.id
        return plan.id

    # ------------------------------------------------------------- triggers

    @staticmethod
    def _driver_key(spec: CronJobSpec) -> str:
        """Persisted identity of whatever drives the job — drift detection."""
        driver = spec.schedule if spec.schedule is not None else spec.trigger
        return driver.model_dump_json() if driver is not None else ""

    @staticmethod
    def _reset_trigger(state: CronJobState) -> None:
        """Re-bootstrap a trigger after its spec was hot-edited.

        ``trigger_fired_at`` deliberately survives: an edit may re-arm the
        observation but must not buy a fire inside the previous cooldown
        window.
        """
        state.trigger_seen_mtime = None
        state.trigger_held = False
        state.queued_due = False
        state.next_due_at = None  # a schedule→trigger conversion may leave one

    def _tick_trigger(self, resolved: _ResolvedJob, state: CronJobState, now: float) -> None:
        trig = resolved.spec.trigger
        if trig is None:
            return
        cooldown = trig.cooldown_seconds or self._config.harness.cron.min_interval_seconds
        if trig.type == "file":
            reason = self._eval_file_trigger(resolved, state, now, cooldown)
        else:
            reason = self._eval_body_trigger(resolved, state, now, cooldown)
        if reason is not None:
            self._materialize(resolved, state, now, reason=reason)

    def _eval_file_trigger(
        self,
        resolved: _ResolvedJob,
        state: CronJobState,
        now: float,
        cooldown: float,
    ) -> str | None:
        """Return a fire-reason when the watched file's mtime changed.

        First sight adopts the current mtime without firing — a pre-existing
        file is not a change. A change observed inside the cooldown window
        stays pending: ``trigger_seen_mtime`` is only consumed on fire (or
        on an overlap decision), so a burst collapses into one fire.
        """
        trig = resolved.spec.trigger
        assert trig is not None
        path = self._resolve_trigger_path(resolved, trig.path or "")
        if path is None:
            return None  # warned at merge time
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None  # missing file == no change observed
        if state.trigger_seen_mtime is None:
            state.trigger_seen_mtime = mtime
            self._state.update(state)
            return None
        if mtime == state.trigger_seen_mtime:
            return None
        # Cooldown gates *consumption*, not just firing: an edge inside the
        # window stays pending so a queued re-fire can never chain
        # fire-on-completion in defiance of the floor.
        if state.trigger_fired_at is not None and now - state.trigger_fired_at < cooldown:
            return None
        if state.running_task_id is not None:
            # Consume the edge once: skip marks it, queue owes a re-fire.
            state.trigger_seen_mtime = mtime
            if resolved.spec.overlap == "queue":
                state.queued_due = True
            else:
                state.last_status = "skipped"
            self._state.update(state)
            return None
        state.trigger_seen_mtime = mtime
        self._state.update(state)
        return f"file changed: {path}"

    def _eval_body_trigger(
        self,
        resolved: _ResolvedJob,
        state: CronJobState,
        now: float,
        cooldown: float,
    ) -> str | None:
        """Edge-fire when ``field op value`` in chat_body_state.json goes
        false→true. While the condition holds, no refire; when it clears,
        the trigger re-arms. An edge observed inside the cooldown window
        stays pending (``trigger_held`` is only consumed on a decision), so
        it still fires once the window elapses if the condition holds."""
        trig = resolved.spec.trigger
        assert trig is not None
        body_path = self._sessions_root / resolved.chat_id.replace("/", "_") / _BODY_STATE_FILENAME
        try:
            data = json.loads(body_path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        current = data.get(trig.field) if isinstance(data, dict) else None
        if current is None:
            # Persona-scope fields (energy, mood_tint, persona attention
            # fallback) live beside the chat dirs — one body, one record.
            persona_path = self._sessions_root / _PERSONA_BODY_STATE_FILENAME
            try:
                pdata = json.loads(persona_path.read_text())
            except (OSError, json.JSONDecodeError):
                pdata = {}
            current = pdata.get(trig.field) if isinstance(pdata, dict) else None
        # Richer body-state fields are {"value": ..., "set_at": ...} records —
        # compare on the value so `fatigue > 0.7` and `attention == "background"`
        # keep the existing false→true vocabulary.
        if isinstance(current, dict) and "value" in current:
            current = current["value"]
        held = self._compare(current, trig.op, trig.value)
        if not held:
            if state.trigger_held:
                state.trigger_held = False  # re-arm
                self._state.update(state)
            return None
        if state.trigger_held:
            return None
        # New edge. Cooldown gates consumption, not just firing — inside the
        # window the edge stays pending and is re-observed once both the run
        # and the window have passed (an edge whose condition clears first
        # simply evaporates).
        if state.trigger_fired_at is not None and now - state.trigger_fired_at < cooldown:
            return None
        if state.running_task_id is not None:
            state.trigger_held = True  # consume the edge once
            if resolved.spec.overlap == "queue":
                state.queued_due = True
            else:
                state.last_status = "skipped"
            self._state.update(state)
            return None
        state.trigger_held = True
        self._state.update(state)
        return f"body {trig.field} {trig.op} {trig.value!r} (now {current!r})"

    def _resolve_trigger_path(self, resolved: _ResolvedJob, raw: str) -> Path | None:
        """Resolve a file-trigger path, confined to the job's allowed roots.

        ``session:<rel>`` selects the owning chat's session dir; other
        relative paths resolve under the persona dir, and absolute paths
        must land under an allowed root. Operator-global files may also
        reach under ``$HOME`` — persona-authored jobs may not. The
        operator's ``trigger_allowed_roots`` opens extra roots (e.g. a
        shared common-room mount) to both kinds of job.
        """
        session_root = (self._sessions_root / resolved.chat_id.replace("/", "_")).resolve()
        roots = [session_root]
        if resolved.persona_dir is not None:
            roots.insert(0, resolved.persona_dir.resolve())
        if resolved.source_label == "global":
            roots.append(Path.home().resolve())
        for extra in self._config.harness.cron.trigger_allowed_roots:
            roots.append(extra.resolve())

        if raw.startswith(_SESSION_TRIGGER_PREFIX):
            candidate = (session_root / raw[len(_SESSION_TRIGGER_PREFIX) :]).resolve()
            allowed = [session_root]
        else:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                base = resolved.persona_dir or Path.cwd()
                candidate = (base / candidate).resolve()
            else:
                candidate = candidate.resolve()
            allowed = roots
        for root in allowed:
            if candidate == root or root in candidate.parents:
                return candidate
        return None

    @staticmethod
    def _compare(current: Any, op: str | None, value: Any) -> bool:
        if current is None or op is None:
            return False
        fn = _TRIGGER_OPS.get(op)
        if fn is None:
            return False
        try:
            return bool(fn(current, value))
        except TypeError:
            return False

    def run_now(self, job_id: str) -> dict[str, Any]:
        """Fire a job immediately — the operator door behind POST /cron/<id>/run.

        Manual runs ignore ``enabled`` and auto-``disabled`` (the operator is
        the override; a successful manual run also re-enables the job) but
        respect overlap — an in-flight run is a conflict. The schedule is not
        consumed: ``next_due_at`` is left alone.
        """
        resolved = self._jobs.get(job_id)
        if resolved is None:
            raise KeyError(job_id)
        state = self._state.ensure(job_id, source_file=str(resolved.source_file))
        if state.running_task_id is not None:
            raise RuntimeError(f"cron job {job_id} is already running")
        self._materialize(resolved, state, time.time(), manual=True)
        return {
            "job_id": job_id,
            "task_id": state.last_task_id,
            "status": "started" if state.running_task_id else "failed",
        }

    def _materialize(
        self,
        resolved: _ResolvedJob,
        state: CronJobState,
        now: float,
        *,
        catchup: bool = False,
        manual: bool = False,
        reason: str | None = None,
    ) -> None:
        spec = resolved.spec
        cron_cfg = self._config.harness.cron
        plan_id = self._cron_plan_id(resolved.chat_id)
        cwd = Path(spec.call.cwd).expanduser() if spec.call.cwd else resolved.persona_dir
        tags = f"{', catchup' if catchup else ''}{', manual' if manual else ''}"
        description = f"cron job {spec.id} ({resolved.source_label}{tags})"
        if reason:
            description += f" — {reason}"
        task = Task(
            name=f"cron:{spec.id}",
            description=description,
            chat_id=resolved.chat_id,
            cwd=cwd,
        )
        if spec.call.type == "script":
            task.type = TaskType.SHELL
            task.command = spec.call.command or ""
        else:
            task.type = TaskType.ACP
            task.prompt = self._compose_phantom(resolved, reason=reason)
            task.acp_model = spec.call.model
            timeout = spec.call.timeout_seconds or cron_cfg.max_llm_timeout_seconds
            task.acp_timeout = min(timeout, cron_cfg.max_llm_timeout_seconds)
            task.mcp_servers = []  # phantom: file tools only, no MCP surface

        added = self._plan_manager.add_task(plan_id, task)
        if added is None:
            logger.warning("Cron job %s: could not add task to plan %s", spec.id, plan_id)
            return
        state.running_task_id = added.id
        state.running_plan_id = plan_id
        state.last_task_id = added.id
        state.last_run_at = now
        state.queued_due = False
        # Hot-edit rule: the run belongs to the spec that fired it. Persist
        # both so finalize can deliver under the firing spec even if the
        # file changes (or the process restarts) mid-run.
        state.fired_spec = spec.model_dump_json()
        state.fired_chat_id = resolved.chat_id
        if not manual and spec.schedule is not None:
            state.next_due_at = self._next_due(spec, now)
        if not manual and spec.trigger is not None:
            # Cooldown anchors on the fire itself, so a queued re-fire that
            # lands after a run still gets a full cooldown before the next.
            state.trigger_fired_at = now
        if catchup:
            state.last_status = "catchup"
        self._state.update(state)
        self._task_to_job[added.id] = spec.id
        try:
            self._task_engine.start_task(plan_id, added.id)
        except Exception as exc:  # noqa: BLE001 — leave no orphaned READY task
            logger.warning("Cron job %s failed to start: %s", spec.id, exc)
            self._plan_manager.fail_task(plan_id, added.id, log=f"cron could not start: {exc}")
            state.running_task_id = None
            state.running_plan_id = None
            state.last_status = "failed"
            state.last_summary = f"could not start: {exc}"
            self._state.update(state)

    # ------------------------------------------------------------ completion

    def _on_event(self, event: Event) -> None:
        if event.type not in ("task.completed", "task.failed"):
            return
        task_id = event.payload.get("task_id")
        if not task_id:
            return
        job_id = self._task_to_job.pop(task_id, None)
        states = self._state.all()
        state = None
        if job_id is not None:
            state = states.get(job_id)
        if state is None:
            for candidate in states.values():
                if task_id in (candidate.running_task_id, candidate.last_task_id):
                    state = candidate
                    break
        if state is None:
            return  # not a cron task
        task = self._plan_manager.get_task(state.running_plan_id or "", task_id)
        if task is not None:
            self._finalize(state, task, event=event)

    def _finalize(
        self,
        state: CronJobState,
        task: Task,
        event: Event | None = None,
    ) -> None:
        resolved = self._jobs.get(state.job_id)
        # The event-bus thread and the tick's reconcile pass can both reach
        # here for the same finished task — re-read under the lock so only
        # the first finalizer claims it.
        with self._finalize_lock:
            fresh = self._state.get(state.job_id)
            if fresh is None or fresh.running_task_id != task.id:
                return
            state = fresh
            ok = task.status == TaskStatus.DONE and not task.timed_out
            if event is not None and event.type == "task.failed":
                ok = False
            summary = (task.result or task.log or "").strip().splitlines()
            state.last_summary = summary[0][:200] if summary else ""
            state.last_status = "ok" if ok else "failed"
            state.last_finished_at = time.time()
            state.running_task_id = None
            state.running_plan_id = None
            if ok:
                state.consecutive_failures = 0
                # Only a manual run can succeed on a disabled job, so a
                # success here is the operator's re-enable signal.
                state.disabled = False
            else:
                state.consecutive_failures += 1
                max_failures = resolved.spec.max_consecutive_failures if resolved is not None else 3
                if state.consecutive_failures >= max_failures:
                    state.disabled = True
                    state.last_status = "disabled"
                    state.last_summary = (
                        f"auto-disabled after {state.consecutive_failures} "
                        f"consecutive failures: {state.last_summary}"
                    )
            if resolved is not None:
                # Hot-edited driver while the run was in flight: adopt the
                # new schedule from the next fire / re-bootstrap the trigger.
                driver_key = self._driver_key(resolved.spec)
                if state.schedule_key != driver_key:
                    state.schedule_key = driver_key
                    if resolved.spec.trigger is not None:
                        self._reset_trigger(state)
                    else:
                        state.next_due_at = self._next_due(resolved.spec, state.last_finished_at)
            self._state.update(state)
        # Delivery belongs to the spec that fired the run, so a hot edit or
        # a mid-run removal still lands the result where the firing spec said.
        delivery_ctx = self._delivery_context(state, resolved)
        if delivery_ctx is not None:
            self._deliver(delivery_ctx, state, task)
        if resolved is not None and state.queued_due and not state.disabled:
            state.queued_due = False
            self._state.update(state)
            self._materialize(resolved, state, time.time())
        elif resolved is None:
            # The job was deleted from every config file mid-run: the result
            # was still delivered under the firing spec; drop the bookkeeping.
            self._state.drop(state.job_id)

    def _delivery_context(
        self,
        state: CronJobState,
        resolved: _ResolvedJob | None,
    ) -> _ResolvedJob | None:
        """Build the delivery view for a finished run: firing spec wins."""
        spec: CronJobSpec | None = None
        if state.fired_spec:
            try:
                spec = CronJobSpec.model_validate_json(state.fired_spec)
            except (ValueError, TypeError):
                spec = None
        if spec is None and resolved is not None:
            spec = resolved.spec
        chat_id = state.fired_chat_id or (resolved.chat_id if resolved else "")
        if spec is None or not chat_id:
            return None
        return _ResolvedJob(
            spec=spec,
            source_file=resolved.source_file if resolved else Path(state.source_file or "."),
            source_label=resolved.source_label if resolved else "",
            chat_id=chat_id,
            persona=resolved.persona if resolved else None,
            persona_dir=resolved.persona_dir if resolved else None,
        )

    # ------------------------------------------------------------- delivery

    def _results_dir(self, chat_id: str) -> Path:
        safe = chat_id.replace("/", "_")
        return self._sessions_root / safe / self._config.harness.cron.results_dirname

    def _deliver(self, resolved: _ResolvedJob, state: CronJobState, task: Task) -> None:
        """Every mode writes the result files; the digest slot reads them.

        ``turn`` additionally enqueues a wake that opens a real turn; when the
        shared self-wake budget or the daily cap refuses, the delivery degrades
        to the files (the digest slot still shows the result next turn).
        """
        outcome = resolved.spec.delivery
        if resolved.spec.delivery == "turn":
            outcome = self._deliver_turn(resolved, state, task)
        out_dir = self._results_dir(resolved.chat_id)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = (task.result or task.log or "").strip()
            log_line = json.dumps(
                {
                    "finished_at": state.last_finished_at,
                    "status": state.last_status,
                    "summary": summary[:500],
                    "task_id": task.id,
                },
                default=str,
            )
            with (out_dir / f"{resolved.spec.id}.log").open("a") as fh:
                fh.write(log_line + "\n")
            (out_dir / f"{resolved.spec.id}.last").write_text(
                json.dumps(
                    {
                        "job_id": resolved.spec.id,
                        "delivery": resolved.spec.delivery,
                        "delivery_result": outcome,
                        "status": state.last_status,
                        "finished_at": state.last_finished_at,
                        "next_due_at": state.next_due_at,
                        "consecutive_failures": state.consecutive_failures,
                        "summary": state.last_summary,
                    },
                    default=str,
                )
            )
        except OSError:
            logger.warning("Cron job %s: could not write result files", resolved.spec.id)

    def _deliver_turn(
        self,
        resolved: _ResolvedJob,
        state: CronJobState,
        task: Task,
    ) -> str:
        """Enqueue the result as a real turn; returns the delivery outcome.

        Shares the self-wake budgets so a turn-delivering job cannot widen the
        interrupt loop an agent could already open herself: pending-cap and
        daily-cap refusals degrade to file delivery (digest still shows it);
        a recent arm only defers the wake by the remaining interval.
        """
        if self._wake_queue is None:
            return "turn_suppressed: no wake queue"
        cron_cfg = self._config.harness.cron
        timer_cfg = self._config.harness.timer
        now = time.time()
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        turns_today = sum(s.turn_count for s in self._state.all().values() if s.turn_date == today)
        if turns_today >= cron_cfg.turn_delivery_max_per_day:
            return "turn_suppressed: daily cap reached"
        pending = [
            e
            for e in self._wake_queue.pending(chat_id=resolved.chat_id)
            if e.reason.startswith(WAKE_BUDGET_REASON_PREFIXES)
        ]
        if len(pending) >= timer_cfg.self_wake_max_pending:
            return "turn_suppressed: wake budget reached"
        scheduled_at = now
        interval = timer_cfg.self_wake_min_interval_seconds
        # Space fires, not arms: only a fire inside the interval window
        # collides with a now-delivery — an overdue event effectively fires
        # at the next waker tick, while an arm for far beyond the window
        # must not push a ready result out behind it.
        colliding = [max(e.scheduled_at, now) for e in pending if e.scheduled_at < now + interval]
        if colliding:
            scheduled_at = max(colliding) + interval
        summary = (task.result or task.log or "").strip()
        message = (
            f"[cron: {resolved.spec.id} finished — {state.last_status}]\n"
            f"{summary or '(no summary)'}\n\n"
            "A scheduled job delivered this result as a turn — respond with judgment."
        )
        self._wake_queue.enqueue(
            WakeEvent(
                id="",
                chat_id=resolved.chat_id,
                reason=f"{CRON_WAKE_REASON_PREFIX}{resolved.spec.id}",
                priority=1,
                scheduled_at=scheduled_at,
                payload={
                    "user_message": message,
                    "agent_reason": f"{CRON_WAKE_REASON_PREFIX}{resolved.spec.id}",
                    "notify": True,
                },
                silent=False,
                created_at=now,
                ready=True,
            )
        )
        if state.turn_date != today:
            state.turn_date = today
            state.turn_count = 0
        state.turn_count += 1
        self._state.update(state)
        if scheduled_at > now:
            return f"turn_deferred: {scheduled_at - now:.0f}s"
        return "turn"

    def _write_service_last(self) -> None:
        """Surface reload warnings where the digest slot can read them."""
        chat_id = self._config.harness.mesh.fallback_chat_id
        if not chat_id:
            return
        out_dir = self._results_dir(chat_id)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "service.last").write_text(
                json.dumps(
                    {"warnings": self._warnings, "updated_at": time.time()},
                    default=str,
                )
            )
        except OSError:
            pass

    # -------------------------------------------------------------- phantom

    def _compose_phantom(self, resolved: _ResolvedJob, reason: str | None = None) -> str:
        cron_cfg = self._config.harness.cron
        persona = resolved.persona or self._config.persona
        parts: list[str] = [f"# {persona.name if persona else 'agent'} — background job", ""]
        if persona is not None:
            try:
                identity = compose_persona(persona).text
                parts.append(_trim_to_section(identity, cron_cfg.phantom_persona_max_chars))
            except FileNotFoundError:
                parts.append(f"(persona files for {persona.name} unavailable)")
        if persona is not None:
            promoted = PromotedMemory(
                self._config.harness.memory,
                persona,
                self._sessions_root,
                resolved.chat_id,
                backend_fn=lambda: None,  # type: ignore[return-value]
            )
            memory = promoted.persona_memory(cron_cfg.phantom_memory_max_chars)
            if memory["text"]:
                parts += ["", "## Persona memory", "", memory["text"]]
            pocket = promoted.promoted_memory(cron_cfg.phantom_promoted_max_chars)
            if pocket["text"]:
                parts += ["", "## Promoted facts (user-curated)", "", pocket["text"]]
        out_dir = self._results_dir(resolved.chat_id)
        parts += [
            "",
            "## Job",
            "",
            *([f"Trigger: {reason}", ""] if reason else []),
            (resolved.spec.call.prompt or "").strip(),
            "",
            "## Output contract",
            "",
            f"Write durable findings to {out_dir / (resolved.spec.id + '.last.md')}.",
            "Reply with a <=3-line summary — that becomes the job result.",
        ]
        return "\n".join(parts)

    # -------------------------------------------------------------- snapshot

    def snapshot(self) -> dict[str, Any]:
        """Read-only view of merged jobs and their state for ``GET /cron``."""
        states = self._state.all()
        today = datetime.now().astimezone().strftime("%Y-%m-%d")
        jobs: list[dict[str, Any]] = []
        for job_id, resolved in sorted(self._jobs.items()):
            spec = resolved.spec
            state = states.get(job_id)
            jobs.append(
                {
                    "id": job_id,
                    "enabled": spec.enabled and not (state and state.disabled),
                    "source": resolved.source_label,
                    "source_file": str(resolved.source_file),
                    "chat_id": resolved.chat_id,
                    "delivery": spec.delivery,
                    "call_type": spec.call.type,
                    "schedule": (
                        spec.schedule.model_dump(exclude_none=True)
                        if spec.schedule is not None
                        else None
                    ),
                    "trigger": (
                        spec.trigger.model_dump(exclude_none=True)
                        if spec.trigger is not None
                        else None
                    ),
                    "trigger_state": (
                        {
                            "seen_mtime": state.trigger_seen_mtime,
                            "fired_at": state.trigger_fired_at,
                            "held": state.trigger_held,
                        }
                        if state is not None and spec.trigger is not None
                        else None
                    ),
                    "next_due_at": state.next_due_at if state else None,
                    "last_run_at": state.last_run_at if state else None,
                    "last_status": state.last_status if state else None,
                    "last_summary": state.last_summary if state else "",
                    "consecutive_failures": state.consecutive_failures if state else 0,
                    "running": bool(state and state.running_task_id),
                    "auto_disabled": bool(state and state.disabled),
                    "turns_today": (state.turn_count if state and state.turn_date == today else 0),
                }
            )
        return {
            "enabled": self._config.harness.cron.enabled,
            "jobs": jobs,
            "warnings": list(self._warnings),
        }
