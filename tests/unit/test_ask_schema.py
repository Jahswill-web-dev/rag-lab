import pytest
from pydantic import ValidationError
from uuid import UUID

from app.schemas.ask import AskRequest


def test_ask_request_trims_the_question() -> None:
    request = AskRequest(question="  What is FastAPI?  ")

    assert request.question == "What is FastAPI?"
    assert request.limit == 3


def test_ask_request_rejects_a_whitespace_only_question() -> None:
    with pytest.raises(ValidationError, match="non-whitespace"):
        AskRequest(question=" \n\t ")


def test_ask_request_accepts_an_optional_document_scope() -> None:
    document_id = UUID("55555555-5555-5555-5555-555555555555")
    request = AskRequest(question="What is FastAPI?", document_id=document_id)

    assert request.document_id == document_id
