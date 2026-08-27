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

For an approval question, use options like `["Approve", "Decline", "Cancel"]`.

Keep the question short. The options are what the user will see as buttons. Do not explain that you are inserting a special block; just include it exactly as shown.
