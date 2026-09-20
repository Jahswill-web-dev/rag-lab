from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document_chunk import DocumentChunk
from app.services.chunking import ChunkDraft


class DocumentChunkRepository:
    """Persistence operations for the retrievable chunks of a source document."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def replace_for_document(
        self,
        document_id: UUID,
        chunks: Sequence[ChunkDraft],
        embeddings: Sequence[Sequence[float]],
    ) -> list[DocumentChunk]:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")

        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

        records = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                embedding=list(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self.session.add_all(records)
        await self.session.flush()
        return records

    async def list_for_document(
        self,
        document_id: UUID,
        *,
        offset: int,
        limit: int,
    ) -> Sequence[DocumentChunk]:
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.scalars(statement)
        return result.all()

    async def count_for_document(self, document_id: UUID) -> int:
        statement = (
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        result = await self.session.scalar(statement)
        return result or 0

    async def search_similar(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int,
        document_id: UUID | None = None,
    ) -> Sequence[tuple[DocumentChunk, float]]:
        distance = DocumentChunk.embedding.cosine_distance(list(query_embedding)).label(
            "distance"
        )
        statement = (
            select(DocumentChunk, distance)
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance.asc())
            .limit(limit)
        )
        if document_id is not None:
            statement = statement.where(DocumentChunk.document_id == document_id)
        rows = (await self.session.execute(statement)).all()
        return [(chunk, 1 - float(value)) for chunk, value in rows]
