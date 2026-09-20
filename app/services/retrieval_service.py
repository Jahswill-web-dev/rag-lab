from collections.abc import Sequence
from uuid import UUID

from app.db.models.document_chunk import DocumentChunk
from app.db.repositories.document_chunk import DocumentChunkRepository
from app.services.embeddings import EmbeddingService


class RetrievalService:
    """Embeds a question and finds the most semantically similar chunks."""

    def __init__(
        self,
        chunks: DocumentChunkRepository,
        embeddings: EmbeddingService,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings

    async def search(
        self,
        query: str,
        *,
        limit: int,
        document_id: UUID | None = None,
    ) -> Sequence[tuple[DocumentChunk, float]]:
        query_embedding = (await self.embeddings.embed_many([query]))[0]
        return await self.chunks.search_similar(
            query_embedding,
            limit=limit,
            document_id=document_id,
        )
