# Hindsight API contract

`diploid-agent` can use a [Hindsight](https://github.com/emiltsoi/hindsight) server as a long-term memory backend. This page describes the HTTP contract the harness expects.

## Endpoints

### `GET /health`

A liveness check. The harness treats `200 OK` as "reachable" and any other response or network error as unreachable, falling back to the local file backend if one is configured.

### `POST /v1/default/banks/{bank}/memories`

Store one or more memory items.

**Request body:**

```json
{
  "items": [
    {
      "content": "User prefers short replies.",
      "timestamp": "2026-09-10T12:00:00",
      "context": "chat-123",
      "metadata": {"role": "user"},
      "document_id": "mem-abc123",
      "tags": ["preference"],
      "update_mode": "replace"
    }
  ],
  "async": true
}
```

- `content` (string, required, non-empty, max 100,000 chars)
- `timestamp` (ISO-8601 string)
- `context` (string)
- `metadata` (object with string values)
- `document_id` (string, unique per document)
- `tags` (list of strings)
- `update_mode` (string, currently always `"replace"`)

**Responses:**

- `2xx` with `"success": true` — items stored.
- `4xx` (except `429`) — permanent payload rejection; the batch is dead-lettered.
- `429` / `5xx` — transient; the harness keeps the items in a local spool and retries.
- Network errors — retried from the local spool.

### `POST /v1/default/banks/{bank}/memories/recall`

Recall memories for a query.

**Request body:**

```json
{
  "query": "what were we building",
  "max_tokens": 1500,
  "prefer_observations": true,
  "types": ["world", "experience", "observation"],
  "tags": ["plan", "state"],
  "tags_match": "any",
  "min_scores": {"world": 0.7}
}
```

**Response body:**

```json
{
  "results": [
    {"text": "..."},
    {"text": "..."}
  ]
}
```

The harness joins the `text` of each result with a blank line and prefixes the result with `Memory from previous turns:` when any text is returned.

### `GET /v1/default/banks/{bank}/stats`

Return bank statistics. The harness includes the returned JSON under a `backend: "hindsight"` wrapper and falls back to `{"reachable": false}` on error.

## Local spool and fallback

When the Hindsight server is unreachable, the harness:

1. Writes validated items to `sessions/<chat_id>/hindsight-pending-retain.jsonl`.
2. Replays the spool on the next `retain()` or `recall()` call once the server is healthy.
3. Dead-letters permanently rejected items to `hindsight-dead-letter.jsonl` with a reason and timestamp.

If `fallback_to_file` is enabled, the file backend handles recall when Hindsight is down.

## Configuration

In `config/harness.yaml`:

```yaml
harness:
  memory:
    backend: hindsight
    hindsight:
      base_url: http://127.0.0.1:8080
      bank: default
      api_key: ${HINDSIGHT_API_KEY}
      async_writes: true
      fallback_to_file: true
```
