# State plugins

The harness supports pluggable, per-chat state. Each plugin can:

- Persist a JSON state file inside the chat session directory.
- Inject a prompt block into a named slot (`wake`, `self_state`, `body`, `mesh`, etc.).
- Expose an MCP server or a chat skill.
- Intercept, modify, or observe the conversation through **lifecycle hooks**.

The former `persona_state` slot has been split into three dedicated slots: `body` (sensation), `self_state` (private mood/resume note), and `mesh` (external protocol). Plugins that do not set `prompt_slot` now default to `self_state`.

## Lifecycle hooks

Hooks are grouped by the part of the conversation they affect. Each hook receives a typed `dataclass` with the data it is most likely to need.

### Hook types

There are three kinds of hook:

- **Gate** — can return `ChatResult` to short-circuit the operation. The first `ChatResult` wins and the chain stops.
- **Consult** — can return a modified context; cannot short-circuit.
- **Notify** — receives a context and returns `None`; for observation and side effects.

### Hook inventory

#### Turn flow

| Hook | Type | When it fires | Context |
|------|------|---------------|---------|
| `before_turn` | **Gate** | Start of `process` / `continue_turn` | `TurnStartContext` |
| `before_format_user_message` | Consult | Before quote handling and formatting | `UserMessageContext` |
| `before_build_prompt` | Consult | Before prompt assembly | `PromptBuildContext` |
| `after_prompt_built` | Consult | After any prompt is built | `PromptContext` |
| `after_first_prompt_built` | Consult | After the first prompt of a session is built | `PromptContext` |
| `before_engine_call` | **Gate** | Before the ACP engine is called | `EngineCallContext` |
| `after_engine_call` | Consult | After the engine returns | `EngineResultContext` |
| `before_record_turn` | Consult | Before the `SessionRecord` is updated | `RecordTurnContext` |
| `after_turn` | Notify | After the turn is recorded | `TurnInfo` |
| `on_turn_error` | Notify | If a turn raises before returning | `TurnErrorContext` |

#### Session lifecycle

| Hook | Type | When it fires | Context |
|------|------|---------------|---------|
| `before_session_archive` | Consult | Before the active session is archived | `SessionArchiveContext` |
| `before_session_clear` | Consult | Before the active directory is cleared | `SessionClearContext` |
| `before_session_start` | **Gate** | Before any new active session is created (`/new`, `/resume`, `/branch`, `/switch-model`) | `SessionStartContext` |
| `after_session_active` | Notify | After the active record is stored and skills/MCP synced | `SessionActiveContext` |

#### Dispatch

| Hook | Type | When it fires | Context |
|------|------|---------------|---------|
| `before_dispatch` | Consult | `dispatch()` creates a pending dispatch | `DispatchCreateContext` |
| `after_dispatch` | Notify | After the dispatch is registered | `DispatchCreateContext` |
| `before_dispatch_continue` | Consult | `continue_turn()` receives a completed dispatch | `DispatchContinueContext` |
| `after_dispatch_continue` | Notify | After the continuation turn completes | `DispatchCompleteContext` |

#### Memory, skills, and MCP

| Hook | Type | When it fires | Context |
|------|------|---------------|---------|
| `on_chat_memory_transition` | Consult | Chat memory crosses its character cap | `MemoryTransitionContext` |
| `on_persona_memory_transition` | Consult | Persona memory crosses its character cap | `MemoryTransitionContext` |
| `before_retain` | Consult | `retain()` is called | `RetainContext` |
| `after_retain` | Notify | After memory is retained | `RetainContext` |
| `before_promote` | Consult | `promote()` is called | `PromoteContext` |
| `after_promote` | Notify | After a fact is promoted to persona memory | `PromoteContext` |
| `before_skill_enabled` | Consult | `/skill enable` | `SkillCommandContext` |
| `after_skill_enabled` | Notify | After the record is updated | `SkillCommandContext` |
| `before_skill_disabled` | Consult | `/skill disable` | `SkillCommandContext` |
| `after_skill_disabled` | Notify | After the record is updated | `SkillCommandContext` |
| `before_mcp_enabled` | Consult | `/mcp enable` | `McpCommandContext` |
| `after_mcp_enabled` | Notify | After the record is updated | `McpCommandContext` |
| `before_mcp_disabled` | Consult | `/mcp disable` | `McpCommandContext` |
| `after_mcp_disabled` | Notify | After the record is updated | `McpCommandContext` |

#### Partial, dispatch, event, and idle

| Hook | Type | When it fires | Context |
|------|------|---------------|---------|
| `on_partial` | Notify | ACP streaming partial arrives | `PartialTurn` |
| `on_dispatch` | Notify | After a dispatch is registered | `Dispatch` |
| `on_event` | Notify | Generic state event fired | `event, payload` |
| `on_idle` | Notify | Future idle loop (not wired yet) | `IdleContext` |

#### Wake and shutdown

| Hook | Type | When it fires | Context |
|------|------|---------------|---------|
| `on_waking` | Notify | Before the first prompt is built | `WakeContext` |
| `on_turn_end` | Notify | After a turn is recorded | `TurnInfo` |
| `on_sleeping` | Notify | End of a turn, and again during `shutdown` with reason `shutdown` | `SleepContext` |
| `on_shutdown` | Notify | `AgentRuntime.shutdown()` (also invokes `on_sleeping`) | `ShutdownContext` |

`start()` and `stop()` are optional lifecycle hooks called when a plugin
instance is first created and when it is removed, globally disabled, or the
harness shuts down. Use them to allocate and release per-plugin resources.

### Chain and return rules

- Plugins run in `harness.plugins` order.
- The output of plugin `N` becomes the input of plugin `N+1`.
- Returning `None` means "no change".
- Exceptions are logged and the chain continues; one failing plugin does not break others.

### Memory transition notices

A plugin can override or suppress the default memory-budget notice:

- Set `context.notice` to replace the default message.
- Set `context.suppress_default = True` to silence it.

This is the mechanism for proposing summarization or other memory discipline behavior.

## Built-in plugins

The built-in state plugins now live in the [`diploid-plugins`](https://github.com/emiltsoi/diploid-plugins) package. Install them with `pip install diploid-agent[plugins]` (or `pip install diploid-plugins`).

### `continuity`

Tracks wake state: time since last turn, last stop reason, and any pending background dispatches.
By default it is placed in the `wake` slot and shown at session boundaries: the first prompt of a session and any rehydrated prompt (for example after ACP resume or stale-session rehydration).
State is stored in `sessions/<chat_id>/chat_wake_state.json`.

### `curriculum`

Tracks a language-learning target, unit, and vocabulary. The `curriculum` skill lets the user add words:

- `/state curriculum set_target_language Klingon`
- `/state curriculum set_unit Greetings`
- `/state curriculum add_word hola hello`

State is stored in `sessions/<chat_id>/chat_curriculum.json`. An MCP server `diploid-curriculum` is also exposed.

## Adding custom plugins

Drop a package directory in `~/.devin/plugins/<name>/` with an `__init__.py` that exports `Plugin`. The class should inherit from `StatePlugin` and implement the hooks it needs. List it in `harness.plugins` and add the search path to `harness.plugin_paths`.

### Example plugin

```python
from pathlib import Path
from diploid_agent.config import PluginConfig
from diploid_agent.models import ChatResult
from diploid_agent.plugins.base import StatePlugin
from diploid_agent.plugins.contexts import TurnStartContext


class MyPlugin(StatePlugin):
    def __init__(self, config: PluginConfig, chat_id: str, sessions_root: Path) -> None:
        super().__init__(config, chat_id, sessions_root)

    def before_turn(self, context: TurnStartContext) -> TurnStartContext | ChatResult | None:
        if context.user_message == "/help":
            return ChatResult(reply="Help topics: ...")
        context.user_message = f"[plugin] {context.user_message}"
        return context
```

Return `None` from a hook to leave the context unchanged, return a modified context to pass changes to the next plugin, or return a `ChatResult` only from gate hooks (`before_turn`, `before_session_start`, `before_engine_call`).

## Backward compatibility

The original `on_waking`, `on_turn_end`, `on_sleeping`, and the previous `on_shutdown` (via records dict) signatures are preserved. Existing plugins that do not implement new hooks are skipped automatically.
