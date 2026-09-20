# RAG Lab API

A learning project that accepts text documents, breaks them into searchable chunks,
stores their embeddings in Postgres with pgvector, and answers questions using the
most relevant chunks as AI context.

## What the API does

```text
Create a document
  → index it into chunks and embeddings
  → search its chunks semantically
  → ask an AI a grounded question and see the source chunks used
```

`/search` performs retrieval only. `/ask` performs retrieval and then asks an AI
model to answer using only the retrieved chunks.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker Desktop, for Postgres + pgvector
- An OpenAI API key, for indexing, semantic search, and AI answers

## First-time setup

From this directory:

```powershell
uv sync
docker compose up -d
```

Create a `.env` file beside `pyproject.toml`:

```env
DATABASE_URL=postgresql+psycopg://rag_lab:rag_lab_dev_password@localhost:5434/rag_lab
OPENAI_API_KEY=your_openai_api_key

# Optional configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
ANSWER_MODEL=gpt-4.1-mini
ANSWER_MAX_OUTPUT_TOKENS=400
```

Run the database migrations:

```powershell
uv run alembic upgrade head
```

Start the API:

```powershell
uv run fastapi dev app/main.py
```

The API normally runs at `http://127.0.0.1:8000`.

- Interactive Swagger documentation: `http://127.0.0.1:8000/docs`
- OpenAPI JSON schema: `http://127.0.0.1:8000/openapi.json`

To stop the database later:

```powershell
docker compose down
```

## Normal usage workflow

1. Create a document with `POST /documents`.
2. Copy the returned document `id`.
3. Index it with `POST /documents/{document_id}/index`.
4. Inspect chunks with `GET /documents/{document_id}/chunks` if needed.
5. Use `POST /search` to inspect retrieval quality.
6. Use `POST /ask` to get an AI answer with its sources.

Indexing and asking use OpenAI API calls. If you change the chunking logic or a
document's content, index that document again to rebuild its chunks and embeddings.

## Endpoints

### `GET /health`

Basic process health check. It does not query the database.

**Response**

```json
{
  "status": "ok"
}
```

### `GET /ready`

Readiness check. It runs a small database query, so it confirms that the API can
talk to Postgres.

**Response**

```json
{
  "status": "ready"
}
```

### `POST /documents`

Stores original source text. Creating a document does not make it searchable; index
it afterwards.

**Request body**

```json
{
  "title": "Atlas Engineering Handbook",
  "content": "Atlas uses written decision notes for important engineering decisions.",
  "source_url": "https://example.com/atlas-handbook"
}
```

`source_url` is optional. `title` accepts 1–200 characters and `content` accepts
1–100,000 characters.

**Response: `201 Created`**

```json
{
  "id": "487606fc-9072-4233-a4a0-5de5df2fa2c2",
  "title": "Atlas Engineering Handbook",
  "content": "Atlas uses written decision notes for important engineering decisions.",
  "source_url": "https://example.com/atlas-handbook",
  "created_at": "2026-09-20T10:30:00Z"
}
```

### `GET /documents`

Lists stored documents with pagination.

**Query parameters**

| Name | Default | Allowed values | Meaning |
| --- | --- | --- | --- |
| `offset` | `0` | `0` or greater | Number of documents to skip. |
| `limit` | `20` | `1`–`100` | Maximum number of documents returned. |

Example: `GET /documents?offset=0&limit=20`

**Response**

```json
{
  "items": [],
  "total": 0,
  "offset": 0,
  "limit": 20
}
```

### `GET /documents/{document_id}`

Returns one original document by UUID.

Example: `GET /documents/487606fc-9072-4233-a4a0-5de5df2fa2c2`

The response has the same shape as `POST /documents`.

### `POST /documents/{document_id}/index`

Makes a document searchable.

The service:

1. Splits document text into overlapping, boundary-aware chunks.
2. Creates an embedding for every chunk.
3. Replaces any existing chunks and embeddings for this document.

**Request body:** none.

**Response**

```json
{
  "document_id": "487606fc-9072-4233-a4a0-5de5df2fa2c2",
  "chunk_count": 4
}
```

### `GET /documents/{document_id}/chunks`

Lists the stored chunks for a document. This is useful when debugging chunking and
confirming that a document was indexed.

**Query parameters**

| Name | Default | Allowed values | Meaning |
| --- | --- | --- | --- |
| `offset` | `0` | `0` or greater | Number of chunks to skip. |
| `limit` | `20` | `1`–`100` | Maximum number of chunks returned. |

**Response**

```json
{
  "items": [
    {
      "id": "be62abd1-eeb6-4059-bcbd-85e91b1bcbd6",
      "document_id": "487606fc-9072-4233-a4a0-5de5df2fa2c2",
      "chunk_index": 0,
      "content": "Atlas uses written decision notes...",
      "start_char": 0,
      "end_char": 455,
      "created_at": "2026-09-20T10:35:00Z"
    }
  ],
  "total": 4,
  "offset": 0,
  "limit": 20
}
```

### `POST /search`

Performs semantic retrieval without asking an answer model. It embeds the `query`,
then uses pgvector cosine similarity to return the closest indexed chunks.

Use this endpoint to check whether retrieval is finding appropriate context before
testing `/ask`.

**Request body**

```json
{
  "query": "How should important engineering decisions be recorded?",
  "limit": 3,
  "document_id": "487606fc-9072-4233-a4a0-5de5df2fa2c2"
}
```

| Field | Required | Default | Rules |
| --- | --- | --- | --- |
| `query` | Yes | — | Non-empty text, maximum 10,000 characters. |
| `limit` | No | `5` | Integer from `1` to `20`. |
| `document_id` | No | `null` | UUID of one document to search. Omit for global search. |

**Response**

```json
{
  "query": "How should important engineering decisions be recorded?",
  "items": [
    {
      "chunk_id": "be62abd1-eeb6-4059-bcbd-85e91b1bcbd6",
      "document_id": "487606fc-9072-4233-a4a0-5de5df2fa2c2",
      "chunk_index": 0,
      "content": "Important decisions must be recorded in a short decision note.",
      "start_char": 0,
      "end_char": 455,
      "score": 0.44
    }
  ]
}
```

Higher scores generally mean closer semantic matches. They are useful for comparing
results, not as a universal certainty percentage.

### `POST /ask`

Performs the complete RAG flow:

```text
question → embedding → retrieve closest chunks → answer model → answer + sources
```

The answer model is instructed to answer only from the retrieved context. The API
returns every source chunk sent to the model, so clients can verify its answer.

**Request body**

```json
{
  "question": "How should important engineering decisions be recorded?",
  "limit": 3,
  "document_id": "487606fc-9072-4233-a4a0-5de5df2fa2c2"
}
```

| Field | Required | Default | Rules |
| --- | --- | --- | --- |
| `question` | Yes | — | Non-empty text, maximum 10,000 characters. |
| `limit` | No | `3` | Integer from `1` to `10`. |
| `document_id` | No | `null` | UUID of one document to search. Omit for global search. |

**Response**

```json
{
  "answer": "Important engineering decisions should be recorded in a short decision note that explains the context, options considered, chosen path, and owner of the next action.",
  "sources": [
    {
      "chunk_id": "be62abd1-eeb6-4059-bcbd-85e91b1bcbd6",
      "document_id": "487606fc-9072-4233-a4a0-5de5df2fa2c2",
      "chunk_index": 0,
      "content": "Important decisions must be recorded in a short decision note...",
      "start_char": 0,
      "end_char": 455,
      "score": 0.44
    }
  ]
}
```

If no indexed chunks match, `/ask` returns a safe message with an empty `sources`
array and does not call the answer model.

## Error responses

Expected application failures use this shape:

```json
{
  "error": {
    "code": "embedding_not_configured",
    "message": "Set OPENAI_API_KEY before indexing documents."
  }
}
```

Common cases:

| Status | Example code | Meaning |
| --- | --- | --- |
| `422` | Validation error | Request JSON is missing a required field or has invalid data. |
| `404` | `document_not_found` | The supplied document UUID does not exist. |
| `502` | `embedding_provider_error` | The OpenAI provider rejected a request. |
| `503` | `embedding_not_configured` | `OPENAI_API_KEY` is missing. |
| `503` | `answer_not_configured` | `OPENAI_API_KEY` is missing when calling `/ask`. |

## Development commands

```powershell
# Run unit tests
uv run pytest

# Compile-check application and test files
uv run python -m compileall app tests

# Check which Alembic migration the database is on
uv run alembic current

# Apply pending migrations
uv run alembic upgrade head
```
