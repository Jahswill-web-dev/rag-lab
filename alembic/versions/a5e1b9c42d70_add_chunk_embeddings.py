"""add chunk embeddings

Revision ID: a5e1b9c42d70
Revises: cbc30050b07e
Create Date: 2026-09-20
"""

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a5e1b9c42d70"
down_revision: Union[str, Sequence[str], None] = "cbc30050b07e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable pgvector and add room for one 1,536-value embedding per chunk."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        "document_chunks",
        sa.Column("embedding", Vector(1536), nullable=True),
    )


def downgrade() -> None:
    """Remove the embedding column and the extension used only by this project."""
    op.drop_column("document_chunks", "embedding")
    op.execute("DROP EXTENSION IF EXISTS vector")
