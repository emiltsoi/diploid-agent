# Mesh integration

`diploid-agent` can act as a peer on the agent mesh when the
[`diploid-mesh`](https://github.com/emiltsoi/diploid-mesh) state plugin is
installed and enabled. The plugin is a separate package that adds an ingress
handler, an MCP server, and a per-chat state file (`chat_mesh_state.json`).

## Enabling mesh

Add to `harness.yaml`:

```yaml
harness:
  mesh:
    enabled: true
    agent_name: diploid-0
    private_key_path: ~/.mesh/keys/diploid-0.pem
    vault_path: ~/.mesh
    registry_url: http://127.0.0.1:8646
    chat_mapping: per_sender
    fallback_chat_id: mesh:inbox
    ingress_module: diploid_mesh.ingress
    mcp_enabled: true
    max_sends_per_turn: 3
    max_message_in_turn_suggestion: 2
  plugins:
    - name: mesh
      enabled: true
      module: diploid_mesh
      prompt_slot: mesh
      first_prompt_only: false
      prompt_order: 50
      max_prompt_chars: 4096
      state_file: chat_mesh_state.json
```

## Message lifecycle

- `reply=yes` (default): the recipient runs an ACP turn and may respond. The
  `mesh_send` MCP tool nudges the agent to use `reply=end` after
  `max_message_in_turn_suggestion` sends and hard-blocks at
  `max_sends_per_turn`.
- `reply=no`: the recipient runs an ACP turn to perform work. The prompt says
  "only reply in an exceptional case," and the MCP server gives the same
  nudge/cap as `reply=yes` but the model is expected to avoid sending.
- `reply=end`: the recipient runs an ACP turn but the `mesh_send` tool hard-blocks
  all mesh replies; this is the last message in the thread.
- DSNs (`[mesh-dsn]` body prefix, or `X-Mesh-Dsn: 1`): delivery-status
  notifications are recorded with `AgentRuntime.record_mesh_message()` and do
  not start a turn.

## Per-turn send cap

`harness.mesh.max_sends_per_turn` hard-limits how many `mesh_send` calls the
agent can make within one active ACP turn. The default is `3`.

`harness.mesh.max_message_in_turn_suggestion` is a soft nudge threshold. After
that many `mesh_send` calls, the tool result includes a note suggesting the
agent use `reply=end` to close the thread. This lets the LLM infer the graceful
close while the hard cap prevents runaway loops.

The per-turn budget is tracked by `MeshSendTracker` inside the `diploid-mesh`
MCP server. It reads the in-flight mesh message from `chat_mesh_state.json` and
queries the harness `GET /turn/{chat_id}` endpoint.

## Prompt discipline for mesh replies

The mesh plugin injects two layers of instruction so the agent cannot mistake a
mesh message for a normal Telegram user message:

1. The `mesh` prompt slot contains the `MESH_CONTRACT` with examples of a good
   `mesh_send` call and a bad one (writing the mesh payload as assistant text).
2. When a mesh message is active, `DiploidMeshPlugin.after_prompt_built` prepends
   a top-of-prompt `SYSTEM — MESH REPLY RULE` block that explicitly commands the
   agent to use `mesh_send`, not to put mesh content in the final assistant text,
   and to close the thread with `reply=end` when done.

For `reply=end` the CTA becomes a silence rule; for `reply=no` it says the
message is one-way and only a tool reply is allowed if the agent chooses to
respond.

## Floating mesh traffic to Telegram

When `harness.notifications.mesh_telegram_float` is `true`, the `diploid-mesh`
MCP server notifies the harness after every successful `mesh_send`. The harness
delivers a system message to the chat's Telegram thread, e.g.:

```
System: [mesh] aurelia → vesper: pong (action=info) (reply=end) (id=<msg_id>)
```

This keeps the human operator aware of mesh content without depending on the
agent to repeat it in assistant text. Set it in `runtime-overrides.yaml`:

```yaml
notifications:
  enabled: true
  mesh_telegram_float: true
```

The delivery method follows `harness.notifications.outbox_delivery`:

- `true` (recommended): the float is enqueued in the per-chat outbox and sent by
  the Telegram `DeliveryWorker`.
- `false`: the float is sent immediately with `notifier.send()`. This works but
  can race with streaming edits; use `outbox_delivery: true` for the cleanest
  behaviour.
