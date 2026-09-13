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
import threading
import time
from dataclasses import dataclass, field
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
from diploid_agent.persona_composer import _trim_to_section, compose_persona
from diploid_agent.plan.models import Task, TaskStatus, TaskType
from diploid_agent.runtime.cron_state import CronJobState, CronStateStore
from diploid_agent.runtime.event_bus import Event

if TYPE_CHECKING:
    from diploid_agent.plan.manager import PlanManager
    from diploid_agent.runtime.event_bus import EventBus
    from diploid_agent.task.engine import TaskEngine

try:
    from croniter import croniter
except ImportError:  # pragma: no cover - dependency guard
    croniter = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CRON_PLAN_NAME = "__cron__"


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
    ) -> None:
        self._config = config
        self._plan_manager = plan_manager
        self._task_engine = task_engine
        self._event_bus = event_bus
        self._sessions_root = Path(sessions_root).expanduser()
        cron_cfg = config.harness.cron
        self._state = CronStateStore(cron_cfg.state_path)
        self._files: list[_CronFileWatch] = []
        self._jobs: dict[str, _ResolvedJob] = {}
        self._warnings: list[str] = []
        self._cron_plans: dict[str, str] = {}  # chat_id -> plan_id
        self._task_to_job: dict[str, str] = {}  # task_id -> job_id
        self._catchup_pending: set[str] = set()
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
            watches.append(
                _CronFileWatch(path=cron_cfg.global_file, label="global")
            )
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
                    warnings.append(
                        f"{watch.path}: ignored — authorship cron_enabled is off"
                    )
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
                        f"cron id conflict: {spec.id} in {watch.path} shadows "
                        f"{merged[spec.id].source_file}"
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
        # Resolve persona for llm calls and cwd defaults.
        persona = self._resolve_persona(spec.call.persona)
        if spec.call.type == "llm" and persona is None:
            warnings.append(
                f"job {spec.id}: persona {spec.call.persona!r} not found — dropped"
            )
            return None
        persona_dir = persona.profile_root if persona is not None else None
        chat_id = spec.chat_id or self._config.harness.mesh.fallback_chat_id
        return _ResolvedJob(
            spec=spec,
            source_file=watch.path,
            source_label=watch.label,
            chat_id=chat_id,
            persona=persona,
            persona_dir=persona_dir,
        )

    @staticmethod
    def _cron_gap(expr: str, now: float) -> float | None:
        """Gap between the next two fires of a cron expression."""
        if croniter is None:
            return None
        try:
            it = croniter(expr, now)
            first = it.get_next(float)
            return it.get_next(float) - first
        except (ValueError, KeyError):
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
            task = self._plan_manager.get_task(
                state.running_plan_id or "", state.running_task_id
            )
            if task is None:
                state.running_task_id = None
                state.running_plan_id = None
                state.last_status = "failed"
                state.last_summary = "task record lost during restart"
                self._state.update(state)
                continue
            self._task_to_job[task.id] = state.job_id
            if task.status in (TaskStatus.DONE, TaskStatus.FAILED):
                self._finalize(state, task)

    # ------------------------------------------------------------ materialize

    def _next_due(self, spec: CronJobSpec, base: float) -> float:
        sched = spec.schedule
        if sched.every_seconds is not None:
            return base + sched.every_seconds
        if sched.at_daily is not None:
            hour, minute = (int(p) for p in sched.at_daily.split(":"))
            lt = time.localtime(base)
            candidate = time.mktime(
                (lt.tm_year, lt.tm_mon, lt.tm_mday, hour, minute, 0, 0, 0, -1)
            )
            if candidate <= base:
                candidate += 86400.0
            return candidate
        if sched.cron is not None and croniter is not None:
            return croniter(sched.cron, base).get_next(float)
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

    def _materialize(
        self,
        resolved: _ResolvedJob,
        state: CronJobState,
        now: float,
        *,
        catchup: bool = False,
    ) -> None:
        spec = resolved.spec
        cron_cfg = self._config.harness.cron
        plan_id = self._cron_plan_id(resolved.chat_id)
        cwd = Path(spec.call.cwd).expanduser() if spec.call.cwd else resolved.persona_dir
        task = Task(
            name=f"cron:{spec.id}",
            description=f"cron job {spec.id} ({resolved.source_label})",
            chat_id=resolved.chat_id,
            cwd=cwd,
        )
        if spec.call.type == "script":
            task.type = TaskType.SHELL
            task.command = spec.call.command or ""
        else:
            task.type = TaskType.ACP
            task.prompt = self._compose_phantom(resolved)
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
        state.next_due_at = self._next_due(spec, now)
        if catchup:
            state.last_status = "catchup"
        self._state.update(state)
        self._task_to_job[added.id] = spec.id
        try:
            self._task_engine.start_task(plan_id, added.id)
        except ValueError as exc:
            logger.warning("Cron job %s failed to start: %s", spec.id, exc)
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
        else:
            state.consecutive_failures += 1
            max_failures = (
                resolved.spec.max_consecutive_failures if resolved is not None else 3
            )
            if state.consecutive_failures >= max_failures:
                state.disabled = True
                state.last_status = "disabled"
                state.last_summary = (
                    f"auto-disabled after {state.consecutive_failures} "
                    f"consecutive failures: {state.last_summary}"
                )
        if resolved is not None:
            self._deliver(resolved, state, task)
            if state.queued_due and not state.disabled:
                state.queued_due = False
                self._state.update(state)
                self._materialize(resolved, state, time.time())
                return
        self._state.update(state)

    # ------------------------------------------------------------- delivery

    def _results_dir(self, chat_id: str) -> Path:
        safe = chat_id.replace("/", "_")
        return self._sessions_root / safe / self._config.harness.cron.results_dirname

    def _deliver(self, resolved: _ResolvedJob, state: CronJobState, task: Task) -> None:
        """Every mode writes the result files; the digest slot reads them."""
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

    def _write_service_last(self) -> None:
        """Surface reload warnings where the digest slot can read them."""
        chat_id = self._config.harness.mesh.fallback_chat_id
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

    def _compose_phantom(self, resolved: _ResolvedJob) -> str:
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
                    "schedule": spec.schedule.model_dump(exclude_none=True),
                    "next_due_at": state.next_due_at if state else None,
                    "last_run_at": state.last_run_at if state else None,
                    "last_status": state.last_status if state else None,
                    "last_summary": state.last_summary if state else "",
                    "consecutive_failures": state.consecutive_failures if state else 0,
                    "running": bool(state and state.running_task_id),
                    "auto_disabled": bool(state and state.disabled),
                }
            )
        return {
            "enabled": self._config.harness.cron.enabled,
            "jobs": jobs,
            "warnings": list(self._warnings),
        }
