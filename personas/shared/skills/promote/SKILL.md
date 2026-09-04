---
name: promote
description: Promote a fact, preference, plan, or watchpoint to the curated pocket that survives fresh compact mode.
allowed-tools:
  - memory_promote
triggers:
  - "/promote"
  - "promote to memory"
  - "promote this"
  - "remember this always"
---

# /promote

Use this skill when the user (or you) wants a fact to outlast context pressure and fresh compact resets.

When to promote:
- The user states a preference ("I prefer dark mode").
- You agree on a plan or decision ("We decided to use Postgres").
- The user marks a watchpoint ("Watch the auth flow for regressions").
- Any fact that should be part of the wake-up thread.

How to promote:
1. Distill the fact to one plain sentence.
2. Call `memory_promote(fact=...)`.
3. Briefly confirm: "Promoted."

Do not promote routine observations or things that will change next turn. Keep the promoted pocket small and durable.
