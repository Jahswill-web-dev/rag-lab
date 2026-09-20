from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_answer_service, get_embedding_service
from app.db.repositories.document_chunk import DocumentChunkRepository
from app.db.session import get_db_session
from app.schemas.ask import AskRequest, AskResponse, AskSource
from app.services.answers import AnswerService
from app.services.embeddings import EmbeddingService
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalService

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_service)],
    answers: Annotated[AnswerService, Depends(get_answer_service)],
) -> AskResponse:
    retrieval = RetrievalService(DocumentChunkRepository(session), embeddings)
    result = await RAGService(retrieval, answers).ask(
        payload.question,
        limit=payload.limit,
        document_id=payload.document_id,
    )
    return AskResponse(
        answer=result.answer,
        sources=[
            AskSource(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                score=score,
            )
            for chunk, score in result.sources
        ],
    )
