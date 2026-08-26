---
name: self-management
description: Tools for managing your own plugins safely.
triggers: ["manage plugin", "add plugin", "remove plugin", "toggle plugin", "rollback plugin", "/self-management"]
allowed-tools: [mcp_call_tool]
---

# Self-management

You can inspect, validate, add, remove, toggle, and roll back plugins using the `diploid-self-management` MCP server.

## Tools

- `plugin_list` — show loaded plugins.
- `plugin_status` — show harness health and plugin health details.
- `plugin_sandbox` — test a plugin module in isolation before loading it.
- `plugin_create` — scaffold a new plugin module, sandbox it, and return a ready config.
- `plugin_incidents` — show recent plugin failures and recovery actions.
- `plugin_propose` — request approval for a mutation; returns a token.
- `plugin_approve` — approve a token after the user confirms.
- `plugin_add` — add a plugin (requires an approved token).
- `plugin_remove` — remove a plugin (requires an approved token).
- `plugin_toggle` — enable/disable a plugin (requires an approved token).
- `plugin_rollback` — undo the last N plugin changes (requires an approved token).

## Safe workflow

1. Use `plugin_create` to author and sandbox a new plugin. It writes files but does not load the plugin.
2. Call `plugin_sandbox` first to make sure the module loads.
3. Call `plugin_propose` with the operation and plugin config to get a token.
4. Tell the user the token and wait for them to say "approve <token>".
5. Call `plugin_approve` with that token.
6. Call the actual mutation (`plugin_add`, `plugin_remove`, etc.) with the same token.
7. If something goes wrong, call `plugin_incidents` to see first-class failure evidence, then `plugin_rollback` can undo it.

Do not perform mutations without a valid approval token unless the user has explicitly disabled approvals.
