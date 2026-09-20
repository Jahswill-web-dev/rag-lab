from collections.abc import Sequence
from uuid import UUID

from app.core.exceptions import AppError
from app.db.models.document import Document
from app.db.repositories.document import DocumentRepository
from app.schemas.document import DocumentCreate


class DocumentService:
    """Creates source documents that will later be chunked and embedded."""

    def __init__(self, documents: DocumentRepository) -> None:
        self.documents = documents

    async def create(self, payload: DocumentCreate) -> Document:
        document = Document(
            title=payload.title,
            content=payload.content,
            source_url=str(payload.source_url) if payload.source_url is not None else None,
        )
        return await self.documents.add(document)

    async def list(self, *, offset: int, limit: int) -> tuple[Sequence[Document], int]:
        documents = await self.documents.list(offset=offset, limit=limit)
        total = await self.documents.count()
        return documents, total

    async def get(self, document_id: UUID) -> Document:
        document = await self.documents.get(document_id)
        if document is None:
            raise AppError(
                status_code=404,
                code="document_not_found",
                message="The document was not found",
            )
        return document
from collections.abc import Sequence
