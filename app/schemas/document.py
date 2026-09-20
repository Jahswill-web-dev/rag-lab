from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DocumentCreate(BaseModel):
    """Validated input for adding source material to the RAG knowledge base."""

    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=100_000)
    source_url: HttpUrl | None = None


class DocumentResponse(BaseModel):
    """Public document representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    content: str
    source_url: HttpUrl | None
    created_at: datetime


class DocumentListResponse(BaseModel):
    """A single, paginated page of source documents."""

    items: list[DocumentResponse]
    total: int
    offset: int
    limit: int


class DocumentIndexResponse(BaseModel):
    """Summary returned after a document's current chunks are rebuilt."""

    document_id: UUID
    chunk_count: int


class DocumentChunkResponse(BaseModel):
    """One stored chunk, including its place in the original document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    created_at: datetime


class DocumentChunkListResponse(BaseModel):
    """A paginated page of a document's chunks."""

    items: list[DocumentChunkResponse]
    total: int
    offset: int
    limit: int
