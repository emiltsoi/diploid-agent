---
name: identity
description: Manage this chat's self-narrative overlay.
allowed-tools:
  - identity_update
triggers:
  - "self-narrative"
  - "who I am"
---

## /identity

Update the chat-scoped self-narrative with the `identity_update` tool.
This writes only `chat_SELF.md`; the persona `SOUL.md` and `MEMORY.md` stay canonical.
