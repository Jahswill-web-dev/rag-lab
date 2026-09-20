from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """A semantic search request over all chunks or one document's chunks."""

    query: str = Field(min_length=1, max_length=10_000)
    limit: int = Field(default=5, ge=1, le=20)
    document_id: UUID | None = None

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("Query must contain non-whitespace text.")
        return query


class SearchResult(BaseModel):
    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    score: float


class SearchResponse(BaseModel):
    query: str
    items: list[SearchResult]
