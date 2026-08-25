"""Planner plugin: turn a triggered user request into an executable plan."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from acp_fleet_harness.config import PluginConfig
from acp_fleet_harness.engine.base import TurnRequest
from acp_fleet_harness.models import ChatResult
from acp_fleet_harness.plan.models import Task, TaskType
from acp_fleet_harness.plugins.base import StatePlugin
from acp_fleet_harness.plugins.contexts import TurnStartContext
from acp_fleet_harness.runtime.plugin_runtime import PluginRuntime

logger = logging.getLogger(__name__)


class PlannerPlugin(StatePlugin):
    """A swappable planner that generates executable plans via the ACP engine."""

    def __init__(
        self,
        config: PluginConfig,
        chat_id: str,
        sessions_root: Any,
        runtime: PluginRuntime | None = None,
    ) -> None:
        super().__init__(config, chat_id, sessions_root, runtime=runtime)

    @property
    def _trigger_prefixes(self) -> list[str]:
        prefixes = self.config.config.get("trigger_prefix")
        if isinstance(prefixes, str):
            return [prefixes]
        if isinstance(prefixes, list):
            return [str(p) for p in prefixes]
        return ["!plan ", "Plan: "]

    @property
    def _max_tasks(self) -> int:
        return int(self.config.config.get("max_tasks", 20))

    @property
    def _auto_triage(self) -> bool:
        return bool(self.config.config.get("auto_triage", False))

    @property
    def _planning_model(self) -> str | None:
        return self.config.config.get("model") or (
            self._runtime.config.engine.model if self._runtime else None
        )

    @property
    def _chat_cwd(self) -> Path:
        return Path(self.sessions_root) / self.chat_id.replace("/", "_")

    def prompt_block(self, max_chars: int | None = None) -> str | None:
        return None

    def before_turn(
        self,
        context: TurnStartContext,
    ) -> TurnStartContext | ChatResult | None:
        """Generate and start a plan when triggered or when auto-triage is on."""
        if self._auto_triage:
            request = context.user_message
        else:
            request = self._extract_request(context.user_message)
            if request is None:
                return None

        if self._runtime is None:
            logger.warning("Planner %s has no runtime reference", self.name)
            return None

        try:
            plan_json = self._ask_for_plan(request)
            plan_data = self._parse_plan_json(plan_json)
        except Exception as exc:
            logger.exception("Planner %s failed to get a plan", self.name)
            return ChatResult(reply=f"I couldn't make a plan for that: {exc}")

        if plan_data is None or not plan_data.get("needs_plan"):
            return None

        try:
            tasks = self._build_tasks(plan_data.get("tasks", []))
        except Exception as exc:
            logger.exception("Planner %s failed to build tasks", self.name)
            return ChatResult(reply=f"I couldn't build the plan: {exc}")

        if not tasks:
            return None

        plan = self._runtime.plan_create(
            name=plan_data.get("plan_name") or request[:60],
            description=plan_data.get("description", request),
            chat_id=self.chat_id,
            tasks=tasks,
        )
        self._runtime.plan_task_start(plan.id)

        return ChatResult(reply=f"Plan started: {plan.name} ({len(tasks)} tasks).")

    def _extract_request(self, user_message: str) -> str | None:
        """Return the request text if the message is a planning trigger."""
        text = user_message.strip()

        if text.startswith("/plan"):
            remainder = text[5:].strip()
            return remainder if remainder else None

        for prefix in self._trigger_prefixes:
            if text.startswith(prefix):
                return text[len(prefix) :].strip()

        return None

    def _ask_for_plan(self, request: str) -> str:
        """Call the engine with a small planning prompt."""
        prompt = self._planning_prompt(request)
        turn_request = TurnRequest(
            prompt=prompt,
            cwd=self._chat_cwd,
            model=self._planning_model,
            mcp_servers=None,
            soft_timeout=None,
        )

        def _prompt() -> Any:
            return self._runtime.engine.prompt(turn_request)

        result = self._runtime.call_engine_unlocked(_prompt)
        return result.reply or ""

    def _planning_prompt(self, request: str) -> str:
        return (
            "You are a planning assistant. Given the user request, decide whether "
            "it needs an executable plan and, if so, return a compact JSON object "
            "with no surrounding prose.\n\n"
            "The JSON must look like this:\n"
            "{\n"
            '  "needs_plan": true,\n'
            '  "plan_name": "short name",\n'
            '  "description": "one line summary",\n'
            '  "tasks": [\n'
            '    {"name": "...", "type": "shell|noop|acp", "command": "...", '
            '"prompt": "...", "depends_on": [], "cwd": "...", "acp_model": "..."}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- needs_plan is false if the request is just chat, greetings, or simple Q&A.\n"
            "- type is 'shell' for commands, 'acp' for model calls (use 'prompt' for the ACP prompt, fallback to 'command'), 'noop' for placeholders.\n"
            "- For 'acp' tasks, 'acp_model' is the optional Devin model to use for that subagent (also accepted as 'model').\n"
            "- depends_on is a list of task names, or empty.\n"
            "- cwd is optional; omit it to use the session directory.\n"
            f"- Keep the plan small (<= {self._max_tasks} tasks).\n\n"
            f"User request: {request}\n\n"
            "Return only the JSON object."
        )

    def _parse_plan_json(self, text: str) -> dict[str, Any] | None:
        """Extract and parse the JSON plan from the model reply."""
        text = text.strip()
        if not text:
            return None

        # Try to find a fenced JSON block.
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            text = fenced.group(1).strip()
        else:
            # Otherwise take the first '{' ... '}' balanced pair.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]

        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Planner could not parse plan JSON: %s", exc)
            return None

    def _build_tasks(self, tasks_data: list[Any]) -> list[Task]:
        """Convert the raw task dicts into Task models."""
        tasks: list[Task] = []
        name_to_id: dict[str, str] = {}

        for i, data in enumerate(tasks_data[: self._max_tasks]):
            if not isinstance(data, dict):
                continue

            name = (data.get("name") or f"task-{i + 1}").strip()
            if not name:
                name = f"task-{i + 1}"

            type_str = data.get("type", "shell")
            try:
                task_type = TaskType(type_str)
            except ValueError:
                logger.warning("Unknown task type %r; defaulting to shell", type_str)
                task_type = TaskType.SHELL

            cwd = data.get("cwd")
            if cwd:
                cwd = Path(cwd).expanduser()

            task = Task(
                name=name,
                description=data.get("description", ""),
                type=task_type,
                command=data.get("command", ""),
                prompt=data.get("prompt"),
                acp_model=data.get("acp_model") or data.get("model"),
                depends_on=list(data.get("depends_on", [])),
                cwd=cwd,
                chat_id=self.chat_id,
            )
            name_to_id[name] = task.id
            tasks.append(task)

        # Resolve any dependency references that were given as task names.
        for task in tasks:
            resolved: list[str] = []
            for dep in task.depends_on:
                if dep in name_to_id:
                    resolved.append(name_to_id[dep])
                else:
                    resolved.append(dep)
            task.depends_on = resolved

        return tasks
