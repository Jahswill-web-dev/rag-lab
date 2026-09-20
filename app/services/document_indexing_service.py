from collections.abc import Sequence
from uuid import UUID

from app.db.models.document_chunk import DocumentChunk
from app.db.repositories.document_chunk import DocumentChunkRepository
from app.services.chunking import chunk_text
from app.services.document_service import DocumentService
from app.services.embeddings import EmbeddingService


class DocumentIndexingService:
    """Converts one stored source document into retrievable database chunks."""

    def __init__(
        self,
        documents: DocumentService,
        chunks: DocumentChunkRepository,
        embeddings: EmbeddingService,
    ) -> None:
        self.documents = documents
        self.chunks = chunks
        self.embeddings = embeddings

    async def index(self, document_id: UUID) -> int:
        document = await self.documents.get(document_id)
        drafts = chunk_text(document.content)
        vectors = await self.embeddings.embed_many([draft.content for draft in drafts])
        records = await self.chunks.replace_for_document(document.id, drafts, vectors)
        return len(records)

    async def list_chunks(
        self,
        document_id: UUID,
        *,
        offset: int,
        limit: int,
    ) -> tuple[Sequence[DocumentChunk], int]:
        await self.documents.get(document_id)
        chunks = await self.chunks.list_for_document(
            document_id,
            offset=offset,
            limit=limit,
        )
        total = await self.chunks.count_for_document(document_id)
        return chunks, total
