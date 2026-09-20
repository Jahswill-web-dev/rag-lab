from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_embedding_service
from app.db.repositories.document_chunk import DocumentChunkRepository
from app.db.session import get_db_session
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.embeddings import EmbeddingService
from app.services.retrieval_service import RetrievalService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> SearchResponse:
    service = RetrievalService(DocumentChunkRepository(session), embeddings)
    matches = await service.search(
        payload.query,
        limit=payload.limit,
        document_id=payload.document_id,
    )
    return SearchResponse(
        query=payload.query,
        items=[
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                score=score,
            )
            for chunk, score in matches
        ],
    )
