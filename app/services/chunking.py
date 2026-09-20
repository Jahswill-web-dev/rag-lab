from dataclasses import dataclass
import re


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
_SENTENCE_END = re.compile(r"(?<=[.!?])(?=\s|$)")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """An in-memory chunk ready to be stored for a source document."""

    chunk_index: int
    content: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1_000,
    overlap: int = 150,
) -> list[ChunkDraft]:
    """Split text into overlapping windows that prefer natural boundaries.

    Character spans refer directly to the original text, making every stored
    chunk traceable to its source document. Paragraph, sentence, and word
    boundaries are preferred over a hard character cut whenever possible.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be greater than or equal to zero and less than chunk_size")
    if not text.strip():
        return []

    chunks: list[ChunkDraft] = []
    start_char = 0

    while start_char < len(text):
        hard_end = min(start_char + chunk_size, len(text))
        end_char = _find_preferred_end(text, start_char, hard_end)
        chunks.append(
            ChunkDraft(
                chunk_index=len(chunks),
                content=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
        )
        if end_char == len(text):
            break
        next_start = _find_clean_next_start(text, end_char, overlap)
        start_char = next_start if next_start > start_char else end_char

    return chunks


def _find_preferred_end(text: str, start: int, hard_end: int) -> int:
    """Find the last useful boundary in the latter half of a chunk window."""
    if hard_end == len(text):
        return hard_end

    minimum_break = start + max(1, (hard_end - start) // 2)
    for pattern in (_PARAGRAPH_BREAK, _SENTENCE_END, _WHITESPACE):
        boundaries = [
            match.start()
            for match in pattern.finditer(text, start, hard_end)
            if minimum_break <= match.start() < hard_end
        ]
        if boundaries:
            return boundaries[-1]

    return hard_end


def _find_clean_next_start(text: str, end: int, overlap: int) -> int:
    """Keep approximate overlap while starting the next chunk on clean text."""
    proposed_start = max(0, end - overlap)
    if proposed_start == 0:
        return 0

    for pattern in (_PARAGRAPH_BREAK, _SENTENCE_END, _WHITESPACE):
        for match in pattern.finditer(text, proposed_start, end):
            boundary = match.end()
            if proposed_start < boundary < end:
                return boundary

    return proposed_start
