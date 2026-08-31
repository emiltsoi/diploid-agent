# Fleet shared AGENTS

This file holds cross-persona instructions shared by all personas under
`personas/`. Put fleet-wide conventions, tool usage rules, and safety guardrails
here.

## Asking the user to choose

When you need the user to pick from a list of options, or to approve/decline/cancel something, write the question in plain text and then include a fenced `ask` block with the question and options in JSON:

````
Which file should I edit?

```ask
{"question": "Which file should I edit?", "options": ["a.py", "b.py", "c.py"]}
```
````

For an approval question, use options like `["Approve", "Decline"]`. The default cancel button lets the user dismiss the prompt.

Keep the question short. The options are what the user will see as buttons. Do not explain that you are inserting a special block; just include it exactly as shown.

Every ask block has a default cancel button, so the user can dismiss the prompt without starting a turn. Do not include a `"Cancel"` option unless you want the user to send it as an explicit answer, and do not include `"Other (please specify)"` — the cancel button is the escape hatch.

If you need a genuinely open-ended answer, ask the user directly in a follow-up after they cancel, or set `"cancellable": false` and make "Other" one of the regular options.

If the user should be able to cancel the prompt without sending a turn, the default is already on. Set `"cancellable": false` to make a forced-choice prompt with no cancel button, or provide an optional `"cancel_label"` (default `"Cancel"):

````
Should I continue? (forced choice)

```ask
{"question": "Should I continue?", "options": ["Yes", "No"], "cancellable": false}
```
````

````
Should I continue? (custom cancel label)

```ask
{"question": "Should I continue?", "options": ["Yes", "No"], "cancel_label": "Never mind"}
```
````

## Memory tools

The `diploid-memory` MCP server is always available. When you learn a fact, preference, decision, or anything that should survive this session, make it durable:

- `memory_recall(query, tags, max_tokens)` — before guessing about the past.
- `memory_retain(content, tags, context)` — for observations and chat-level facts.
- `memory_promote(fact)` — for facts that belong in your persona `MEMORY.md`.

If you state an important fact in a reply, you may also wrap it in a ` ```memory ` block; the harness will promote it automatically.

Do not rely on the conversation transcript alone. Use the tools explicitly.
