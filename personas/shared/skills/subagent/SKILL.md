# subagent

Use `harness_subagent` when the user asks for a task that:
- is expected to take longer than the current turn timeout,
- must keep running even if the parent turn is stopped or restarted,
- or is a long-running background job (research, batch processing, long builds, waiting on external events).

Do **not** use the built-in `run_subagent` tool for these cases; it runs inside the parent ACP turn and will be killed if the turn is killed.

Call `harness_subagent` with a clear, self-contained `prompt`. You may include a short `context` that will be shown to the user when the background task completes. The subagent runs in a separate ACP process and the harness will continue the chat with the result when it finishes.

Trigger words: `subagent`, `background`, `long running`, `long task`, `run in background`, `spawn agent`, `long timeout`, `research this`.

MCP tool: `diploid-harness::harness_subagent`
