# Plugin Contract

A valid `diploid-agent` plugin module:

- Exposes a top-level `Plugin` class.
- Inherits from `diploid_agent.plugins.base.StatePlugin` (recommended).
- Implements optional `start()` and `stop()` lifecycle hooks.
- Implements an optional `health()` method returning `{"healthy": bool, ...}` or `None`.
- Provides an MCP server via `PluginConfig.mcp_server` or by overriding `StatePlugin.mcp_server()`.

The harness wraps every plugin call in `BaseException` and records any failure to `plugin-incidents.jsonl`.

## Hot reload

`POST /plugin/reload` (or Telegram `/plugin reload <name>`) swaps plugin code in a running harness without a service restart. Semantics to design for:

- The configured `module` and every already-imported submodule beneath it are re-executed, deepest-first, then the entry module. **Keep module-level code side-effect free** — no threads, sockets, file writes, or background work at import time. Allocate resources in `start()`, release them in `stop()`.
- The reload runs *before* any instance is dropped. A module that fails to import raises out of the endpoint and the running instances keep working — a broken edit degrades to a failed reload, not a `FailedPlugin`.
- On success the old instance is `stop()`ed and dropped in **every** chat, then lazily recreated from the new class on next use. On-disk state survives because plugins re-read their `state_file` at construction.
- Reload covers only the plugin's own subtree (`<module>.*`). A change in a helper imported from *outside* that subtree is not picked up — restart, or reload every plugin that depends on it.
