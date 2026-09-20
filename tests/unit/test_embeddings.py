import asyncio
from types import SimpleNamespace

import pytest

from app.core.exceptions import AppError
from app.services.embeddings import OpenAIEmbeddingService


class FakeEmbeddingsAPI:
    def __init__(self, data: list[SimpleNamespace]) -> None:
        self.data = data
        self.request: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(data=self.data)


class FakeOpenAIClient:
    def __init__(self, embeddings: FakeEmbeddingsAPI) -> None:
        self.embeddings = embeddings
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_embedding_service_sorts_response_items_and_uses_configured_dimensions() -> None:
    api = FakeEmbeddingsAPI(
        [
            SimpleNamespace(index=1, embedding=[4.0, 5.0, 6.0]),
            SimpleNamespace(index=0, embedding=[1.0, 2.0, 3.0]),
        ]
    )
    service = OpenAIEmbeddingService(
        api_key="test-key",
        model="test-model",
        dimensions=3,
        client=FakeOpenAIClient(api),  # type: ignore[arg-type]
    )

    vectors = asyncio.run(service.embed_many(["first", "second"]))

    assert vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert api.request == {
        "model": "test-model",
        "input": ["first", "second"],
        "dimensions": 3,
        "encoding_format": "float",
    }


def test_embedding_service_requires_an_api_key_only_when_embedding() -> None:
    service = OpenAIEmbeddingService(
        api_key=None,
        model="test-model",
        dimensions=3,
    )

    with pytest.raises(AppError, match="OPENAI_API_KEY") as error:
        asyncio.run(service.embed_many(["a chunk"] ))

    assert error.value.code == "embedding_not_configured"


def test_embedding_service_rejects_an_unexpected_vector_dimension() -> None:
    api = FakeEmbeddingsAPI([SimpleNamespace(index=0, embedding=[1.0, 2.0])])
    service = OpenAIEmbeddingService(
        api_key="test-key",
        model="test-model",
        dimensions=3,
        client=FakeOpenAIClient(api),  # type: ignore[arg-type]
    )

    with pytest.raises(AppError, match="unexpected dimension") as error:
        asyncio.run(service.embed_many(["a chunk"]))

    assert error.value.code == "embedding_response_invalid"
