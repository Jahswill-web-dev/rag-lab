import pytest
from uuid import UUID
from pydantic import ValidationError

from app.schemas.search import SearchRequest


def test_search_request_strips_surrounding_whitespace() -> None:
    request = SearchRequest(query="  What is FastAPI?  ")

    assert request.query == "What is FastAPI?"
    assert request.limit == 5


def test_search_request_rejects_a_whitespace_only_query() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query=" \n\t ")


def test_search_request_accepts_an_optional_document_scope() -> None:
    document_id = UUID("44444444-4444-4444-4444-444444444444")
    request = SearchRequest(query="What is FastAPI?", document_id=document_id)

    assert request.document_id == document_id
