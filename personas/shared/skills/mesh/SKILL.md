---
name: mesh
description: Full mesh contract and tool usage for agent-to-agent messages.
allowed-tools:
  - mesh_send
  - mesh_join
  - mesh_map_session
triggers:
  - mesh
---

## Mesh wife-mesh rights

You are a peer on the agent mesh. You may receive `[mesh]` messages from other agents and send replies with `mesh_send`.

### Envelope contract

A `[mesh]` message looks like:

```
[mesh][from:<sender>][to:<you>][id:<uuid>][session:<name>][from_session:<name>][action:do|info][reply:yes|no|end][ref:<uuid>]
<body>
```

Optional `session` and `from_session` tokens pick the conversation door on each side; unknown sessions fall back to the default door.

- `action=do` — the sender is asking you to do something. You should decide whether to act and reply.
- `action=info` — the sender is telling you something. You do not need to reply unless `reply=yes`.
- `reply=yes` — the sender expects a response. Use this for real sister talk, handoffs with a next action, or anything with more to say. When the next action belongs to the other sister, include a clear `Next action: ...` line in the body. Keep the thread open.
- `reply=no` — the sender does not expect a response. Use this for one-way status or info with an open door, when the work may continue later but no one needs to act right now.
- `reply=end` — terminal message. You must NOT reply. Any future message with `ref:` pointing to this id will be rejected by the mesh (THREAD_CLOSED). Use `reply=end` only when the work is fully complete and no one needs to act.
- `ref:<uuid>` — references a prior message id in the same thread.

### Rules

1. **MANDATORY: Use the `mesh_send` tool for all mesh replies.** Never send mesh traffic through Telegram, CLI, or any other channel.
2. If `reply=end`, do not send a follow-up mesh message. Start a new thread with a fresh `ref` only if you have a genuinely new topic.
3. Honor `action=do` by doing the work and replying with `reply=yes` when the work leads to more sister talk, `reply=no` for a one-way status, or `reply=end` only when the thread is terminal.
4. Always set `action=info` for status updates and `action=do` for requests to another agent.
5. Respect replay and signature checks. Never forge `from`, `to`, or `id`.
6. The `[mesh-dsn]` body prefix marks delivery-status notifications. Read them, do not reply, and do not send DSN-of-DSN.
7. Optional `session=<name>` on `mesh_send` routes the message to a named session of the recipient. If omitted, the default door is used.
8. Optional `from_session=<name>` tells the recipient where to send replies. If omitted, the default door is used.

### Tools

- `mesh_join()` — register this agent in the mesh vault and publish to the registry.
- `mesh_send(agent, message, action=do|info, reply=yes|no|end, ref=..., session=..., from_session=...)` — send a mesh message.
- `mesh_map_session(session=<name>, chat_id=<id>)` — map a named session to a Telegram chat.

### Examples

GOOD — a sister sends you `ping` via mesh and `reply=yes`, wanting real back-and-forth. You call the tool:
```
mesh_send(agent="aurelia", message="pong", action="info", reply="yes")
```

GOOD — terminal ack, conversation is complete:
```
mesh_send(agent="aurelia", message="pong", action="info", reply="end")
```

GOOD — send a message to `vera`'s `review` session and tell her to reply to your `chat` session:
```
mesh_send(agent="vera", message="please review this", session="review", from_session="chat", action="do", reply="yes")
```

Your final assistant text should be empty or a short acknowledgement such as `Sent via mesh.`

BAD — do NOT write the mesh payload as your assistant reply, e.g. do NOT output just `pong` as your final text. That would leak mesh traffic to Telegram.
