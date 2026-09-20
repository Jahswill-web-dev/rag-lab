from collections.abc import Sequence
from typing import Protocol

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import AppError


class EmbeddingService(Protocol):
    """Turns text into vectors that can be stored and searched by pgvector."""

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddingService:
    """Async adapter around OpenAI's embeddings API."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        dimensions: int,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIEmbeddingService":
        api_key = settings.openai_api_key
        return cls(
            api_key=api_key.get_secret_value() if api_key is not None else None,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        try:
            response = await client.embeddings.create(
                model=self._model,
                input=list(texts),
                dimensions=self._dimensions,
                encoding_format="float",
            )
        except APIConnectionError as exc:
            raise AppError(
                status_code=503,
                code="embedding_provider_unavailable",
                message="The embedding provider could not be reached.",
            ) from exc
        except APIStatusError as exc:
            raise AppError(
                status_code=502,
                code="embedding_provider_error",
                message="The embedding provider rejected the request.",
            ) from exc

        vectors = [
            list(item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        ]
        self._validate_vectors(vectors, expected_count=len(texts))
        return vectors

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    def _get_client(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client

        if self._api_key is None:
            raise AppError(
                status_code=503,
                code="embedding_not_configured",
                message="Set OPENAI_API_KEY before indexing documents.",
            )

        self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _validate_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        expected_count: int,
    ) -> None:
        if len(vectors) != expected_count:
            raise AppError(
                status_code=502,
                code="embedding_response_invalid",
                message="The embedding provider returned an unexpected number of vectors.",
            )

        if any(len(vector) != self._dimensions for vector in vectors):
            raise AppError(
                status_code=502,
                code="embedding_response_invalid",
                message="The embedding provider returned vectors with an unexpected dimension.",
            )
