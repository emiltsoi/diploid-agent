# Plugin Contract

A valid `diploid-agent` plugin module:

- Exposes a top-level `Plugin` class.
- Inherits from `diploid_agent.plugins.base.StatePlugin` (recommended).
- Implements optional `start()` and `stop()` lifecycle hooks.
- Implements an optional `health()` method returning `{"healthy": bool, ...}` or `None`.
- Provides an MCP server via `PluginConfig.mcp_server` or by overriding `StatePlugin.mcp_server()`.

The harness wraps every plugin call in `BaseException` and records any failure to `plugin-incidents.jsonl`.
