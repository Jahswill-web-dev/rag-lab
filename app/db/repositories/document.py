from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document


class DocumentRepository:
    """Persistence operations for RAG source documents."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, document: Document) -> Document:
        self.session.add(document)
        await self.session.flush()
        await self.session.refresh(document)
        return document

    async def list(self, *, offset: int, limit: int) -> Sequence[Document]:
        statement = (
            select(Document)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.scalars(statement)
        return result.all()

    async def count(self) -> int:
        result = await self.session.scalar(select(func.count()).select_from(Document))
        return result or 0

    async def get(self, document_id: UUID) -> Document | None:
        return await self.session.get(Document, document_id)
