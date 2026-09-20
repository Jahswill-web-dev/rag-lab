import asyncio
from types import SimpleNamespace
from uuid import UUID

from app.services.rag_service import RAGService


class FakeRetrievalService:
    def __init__(self, matches: list[tuple[SimpleNamespace, float]]) -> None:
        self.matches = matches
        self.request: tuple[str, int, UUID | None] | None = None

    async def search(
        self,
        question: str,
        *,
        limit: int,
        document_id: UUID | None = None,
    ) -> list[tuple[SimpleNamespace, float]]:
        self.request = (question, limit, document_id)
        return self.matches


class FakeAnswerService:
    def __init__(self) -> None:
        self.request: tuple[str, list[tuple[SimpleNamespace, float]]] | None = None

    async def answer(
        self,
        question: str,
        sources: list[tuple[SimpleNamespace, float]],
    ) -> str:
        self.request = (question, sources)
        return "The grounded answer."


def test_rag_service_retrieves_then_answers_with_the_same_sources() -> None:
    sources = [(SimpleNamespace(content="Relevant context"), 0.8)]
    retrieval = FakeRetrievalService(sources)
    answers = FakeAnswerService()
    service = RAGService(retrieval, answers)  # type: ignore[arg-type]

    result = asyncio.run(service.ask("What is relevant?", limit=3))

    assert retrieval.request == ("What is relevant?", 3, None)
    assert answers.request == ("What is relevant?", sources)
    assert result.answer == "The grounded answer."
    assert result.sources == sources


def test_rag_service_does_not_call_the_model_when_nothing_is_retrieved() -> None:
    retrieval = FakeRetrievalService([])
    answers = FakeAnswerService()
    service = RAGService(retrieval, answers)  # type: ignore[arg-type]

    result = asyncio.run(service.ask("What is relevant?", limit=3))

    assert result.answer == "I don't have enough indexed context to answer that question."
    assert result.sources == []
    assert answers.request is None


def test_rag_service_forwards_an_optional_document_scope() -> None:
    document_id = UUID("33333333-3333-3333-3333-333333333333")
    retrieval = FakeRetrievalService([])
    answers = FakeAnswerService()
    service = RAGService(retrieval, answers)  # type: ignore[arg-type]

    asyncio.run(service.ask("What is relevant?", limit=3, document_id=document_id))

    assert retrieval.request == ("What is relevant?", 3, document_id)
