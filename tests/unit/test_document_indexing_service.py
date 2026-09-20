import asyncio
from types import SimpleNamespace
from uuid import UUID

from app.services.document_indexing_service import DocumentIndexingService


class FakeDocumentService:
    async def get(self, _document_id: UUID) -> SimpleNamespace:
        return SimpleNamespace(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            content="abcdefghij",
        )


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.texts: list[str] | None = None

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[float(index)] * 3 for index, _ in enumerate(texts)]


class FakeChunkRepository:
    def __init__(self) -> None:
        self.document_id: UUID | None = None
        self.chunks: object | None = None
        self.embeddings: object | None = None

    async def replace_for_document(
        self,
        document_id: UUID,
        chunks: object,
        embeddings: object,
    ) -> list[object]:
        self.document_id = document_id
        self.chunks = chunks
        self.embeddings = embeddings
        return [object()]


def test_indexing_embeds_chunk_text_before_persisting_chunks() -> None:
    embeddings = FakeEmbeddingService()
    chunks = FakeChunkRepository()
    service = DocumentIndexingService(
        FakeDocumentService(),  # type: ignore[arg-type]
        chunks,  # type: ignore[arg-type]
        embeddings,
    )

    result = asyncio.run(service.index(UUID("00000000-0000-0000-0000-000000000001")))

    assert result == 1
    assert embeddings.texts == ["abcdefghij"]
    assert chunks.embeddings == [[0.0, 0.0, 0.0]]
