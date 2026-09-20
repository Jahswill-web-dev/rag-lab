from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from app.db.models.document_chunk import DocumentChunk
from app.services.answers import AnswerService
from app.services.retrieval_service import RetrievalService


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    answer: str
    sources: Sequence[tuple[DocumentChunk, float]]


class RAGService:
    """Coordinates retrieval and generation for the /ask endpoint."""

    _NO_CONTEXT_ANSWER = "I don't have enough indexed context to answer that question."

    def __init__(self, retrieval: RetrievalService, answers: AnswerService) -> None:
        self.retrieval = retrieval
        self.answers = answers

    async def ask(
        self,
        question: str,
        *,
        limit: int,
        document_id: UUID | None = None,
    ) -> RAGAnswer:
        sources = await self.retrieval.search(
            question,
            limit=limit,
            document_id=document_id,
        )
        if not sources:
            return RAGAnswer(answer=self._NO_CONTEXT_ANSWER, sources=[])

        answer = await self.answers.answer(question, sources)
        return RAGAnswer(answer=answer, sources=sources)
