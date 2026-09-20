import asyncio
from types import SimpleNamespace
from uuid import UUID

from app.services.retrieval_service import RetrievalService


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.texts: list[str] | None = None

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[0.1, 0.2, 0.3]]


class FakeChunkRepository:
    def __init__(self) -> None:
        self.embedding: list[float] | None = None
        self.limit: int | None = None

    async def search_similar(
        self,
        query_embedding: list[float],
        *,
        limit: int,
        document_id: UUID | None = None,
    ) -> list[tuple[SimpleNamespace, float]]:
        self.embedding = query_embedding
        self.limit = limit
        self.document_id = document_id
        return [(SimpleNamespace(content="relevant chunk"), 0.8)]


def test_retrieval_embeds_the_query_then_searches_similar_chunks() -> None:
    embeddings = FakeEmbeddingService()
    chunks = FakeChunkRepository()
    service = RetrievalService(
        chunks,  # type: ignore[arg-type]
        embeddings,
    )

    matches = asyncio.run(service.search("How does retrieval work?", limit=3))

    assert embeddings.texts == ["How does retrieval work?"]
    assert chunks.embedding == [0.1, 0.2, 0.3]
    assert chunks.limit == 3
    assert chunks.document_id is None
    assert matches[0][0].content == "relevant chunk"
    assert matches[0][1] == 0.8


def test_retrieval_forwards_an_optional_document_scope() -> None:
    document_id = UUID("11111111-1111-1111-1111-111111111111")
    embeddings = FakeEmbeddingService()
    chunks = FakeChunkRepository()
    service = RetrievalService(chunks, embeddings)  # type: ignore[arg-type]

    asyncio.run(
        service.search(
            "How does retrieval work?",
            limit=3,
            document_id=document_id,
        )
    )

    assert chunks.document_id == document_id
