---
name: curriculum
description: Manage the language-learning curriculum.
allowed-tools:
  - curriculum_add_word
  - curriculum_set_target_language
  - curriculum_set_unit
  - curriculum_state
triggers:
  - "add word"
  - "new word"
  - "target language"
  - "current unit"
  - "vocabulary"
---

## /curriculum

Manage the language-learning curriculum for this chat.

### Commands

- `/curriculum add_word <word> <translation>` — add a word to the vocabulary.
- `/curriculum set_target_language <language>` — set the language being studied.
- `/curriculum set_unit <unit>` — set the current unit or topic.

The curriculum state is saved to `chat_curriculum.json` and is shown in the prompt.
