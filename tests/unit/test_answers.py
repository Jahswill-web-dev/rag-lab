import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.core.exceptions import AppError
from app.services.answers import OpenAIAnswerService


class FakeResponsesAPI:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.request: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.request = kwargs
        return SimpleNamespace(output_text=self.output_text)


class FakeOpenAIClient:
    def __init__(self, responses: FakeResponsesAPI) -> None:
        self.responses = responses
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def make_source(content: str) -> tuple[SimpleNamespace, float]:
    return (
        SimpleNamespace(
            id=UUID("11111111-1111-1111-1111-111111111111"),
            document_id=UUID("22222222-2222-2222-2222-222222222222"),
            content=content,
        ),
        0.9,
    )


def test_answer_service_sends_only_supplied_context_to_the_model() -> None:
    responses = FakeResponsesAPI("  Write a short decision note.  ")
    service = OpenAIAnswerService(
        api_key="test-key",
        model="test-model",
        max_output_tokens=123,
        client=FakeOpenAIClient(responses),  # type: ignore[arg-type]
    )

    answer = asyncio.run(
        service.answer(
            "How are decisions recorded?",
            [make_source("Decisions belong in a short decision note.")],  # type: ignore[list-item]
        )
    )

    assert answer == "Write a short decision note."
    assert responses.request is not None
    assert responses.request["model"] == "test-model"
    assert responses.request["max_output_tokens"] == 123
    assert responses.request["store"] is False
    assert "How are decisions recorded?" in str(responses.request["input"])
    assert "Decisions belong in a short decision note." in str(responses.request["input"])
    assert "only the supplied reference context" in str(responses.request["instructions"])


def test_answer_service_requires_an_api_key_only_when_answering() -> None:
    service = OpenAIAnswerService(
        api_key=None,
        model="test-model",
        max_output_tokens=123,
    )

    with pytest.raises(AppError, match="OPENAI_API_KEY") as error:
        asyncio.run(service.answer("Question", [make_source("Context")]))  # type: ignore[list-item]

    assert error.value.code == "answer_not_configured"
