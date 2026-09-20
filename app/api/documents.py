from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_embedding_service
from app.db.repositories.document import DocumentRepository
from app.db.repositories.document_chunk import DocumentChunkRepository
from app.db.session import get_db_session
from app.schemas.document import (
    DocumentCreate,
    DocumentChunkListResponse,
    DocumentChunkResponse,
    DocumentIndexResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.document_indexing_service import DocumentIndexingService
from app.services.document_service import DocumentService
from app.services.embeddings import EmbeddingService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: DocumentCreate,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    service = DocumentService(DocumentRepository(session))
    document = await service.create(payload)
    return DocumentResponse.model_validate(document)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DocumentListResponse:
    service = DocumentService(DocumentRepository(session))
    documents, total = await service.list(offset=offset, limit=limit)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(document) for document in documents],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    service = DocumentService(DocumentRepository(session))
    document = await service.get(document_id)
    return DocumentResponse.model_validate(document)


@router.post("/{document_id}/index", response_model=DocumentIndexResponse)
async def index_document(
    document_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> DocumentIndexResponse:
    documents = DocumentService(DocumentRepository(session))
    service = DocumentIndexingService(
        documents,
        DocumentChunkRepository(session),
        embeddings,
    )
    chunk_count = await service.index(document_id)
    return DocumentIndexResponse(document_id=document_id, chunk_count=chunk_count)


@router.get("/{document_id}/chunks", response_model=DocumentChunkListResponse)
async def list_document_chunks(
    document_id: UUID,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_service)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> DocumentChunkListResponse:
    documents = DocumentService(DocumentRepository(session))
    service = DocumentIndexingService(
        documents,
        DocumentChunkRepository(session),
        embeddings,
    )
    chunks, total = await service.list_chunks(
        document_id,
        offset=offset,
        limit=limit,
    )
    return DocumentChunkListResponse(
        items=[DocumentChunkResponse.model_validate(chunk) for chunk in chunks],
        total=total,
        offset=offset,
        limit=limit,
    )
