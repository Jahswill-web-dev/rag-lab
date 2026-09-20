from fastapi import Request

from app.services.answers import OpenAIAnswerService
from app.services.embeddings import OpenAIEmbeddingService


def get_embedding_service(request: Request) -> OpenAIEmbeddingService:
    """Get the app-wide client instead of recreating it for every request."""
    return request.app.state.embedding_service


def get_answer_service(request: Request) -> OpenAIAnswerService:
    """Get the app-wide answer client instead of recreating it for every request."""
    return request.app.state.answer_service
