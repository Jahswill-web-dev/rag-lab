from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """A question answered from all chunks or one document's nearest chunks."""

    question: str = Field(min_length=1, max_length=10_000)
    limit: int = Field(default=3, ge=1, le=10)
    document_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question must contain non-whitespace text.")
        return question


class AskSource(BaseModel):
    """A chunk actually supplied to the answer model as context."""

    chunk_id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]
