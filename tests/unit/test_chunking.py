import pytest

from app.services.chunking import ChunkDraft, chunk_text


def test_chunk_text_returns_no_chunks_for_empty_or_whitespace_text() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\t") == []


def test_chunk_text_returns_one_chunk_when_text_fits_within_the_window() -> None:
    assert chunk_text("FastAPI") == [
        ChunkDraft(
            chunk_index=0,
            content="FastAPI",
            start_char=0,
            end_char=7,
        )
    ]


def test_chunk_text_creates_overlapping_character_windows() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=2)

    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 10),
        (8, 18),
        (16, 26),
    ]
    assert [chunk.content for chunk in chunks] == ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_chunk_text_covers_the_entire_source_document() -> None:
    text = "x" * 2_001

    chunks = chunk_text(text, chunk_size=1_000, overlap=150)

    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)
    assert all(text[chunk.start_char : chunk.end_char] == chunk.content for chunk in chunks)


def test_chunk_text_prefers_a_paragraph_break_before_the_hard_limit() -> None:
    text = "Decision notes record context and owners.\n\nIncident reviews improve systems."

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert chunks[0].content == "Decision notes record context and owners."
    assert chunks[0].end_char == text.index("\n\n")


def test_chunk_text_prefers_a_sentence_end_before_the_hard_limit() -> None:
    text = "Write a decision note. Record an owner. Review the outcome later."

    chunks = chunk_text(text, chunk_size=35, overlap=8)

    assert chunks[0].content.endswith(".")
    assert chunks[0].content == "Write a decision note."


def test_chunk_text_uses_word_boundaries_when_no_sentence_boundary_is_available() -> None:
    text = "alpha bravo charlie delta echo foxtrot golf hotel"

    chunks = chunk_text(text, chunk_size=25, overlap=5)

    assert chunks[0].content == "alpha bravo charlie"
    assert all(
        not (
            chunk.end_char < len(text)
            and text[chunk.end_char - 1].isalnum()
            and text[chunk.end_char].isalnum()
        )
        for chunk in chunks
    )


def test_chunk_text_falls_back_to_hard_cuts_for_an_unbroken_word() -> None:
    chunks = chunk_text("x" * 21, chunk_size=10, overlap=2)

    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [
        (0, 10),
        (8, 18),
        (16, 21),
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_chunk_text_rejects_invalid_configuration(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=chunk_size, overlap=overlap)
