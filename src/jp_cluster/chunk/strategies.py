"""Chunking strategies. Output: list[Chunk]."""

from __future__ import annotations

from functools import lru_cache

from jp_cluster.config import settings
from jp_cluster.models.data import Chunk, Letter


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(settings.embed.model)


def chunk_letter(letter: Letter, text: str, norm_stage: str) -> list[Chunk]:
    """One chunk per letter — Default für Brief-als-Einheit."""
    if len(text) < settings.chunk.min_chunk_chars:
        return []
    return [Chunk(
        chunk_id=f"{letter.id}::0",
        letter_id=letter.id,
        idx=0,
        text=text,
        norm_stage=norm_stage,  # type: ignore[arg-type]
        chunk_mode="letter",
    )]


def chunk_sliding(letter: Letter, text: str, norm_stage: str) -> list[Chunk]:
    """Sliding window über Token-IDs des Embedding-Tokenizers — garantiert konsistent."""
    if len(text) < settings.chunk.min_chunk_chars:
        return []

    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    win = settings.chunk.sliding_tokens
    overlap = settings.chunk.sliding_overlap
    step = win - overlap

    chunks: list[Chunk] = []
    if len(ids) <= win:
        chunks.append(Chunk(
            chunk_id=f"{letter.id}::0",
            letter_id=letter.id,
            idx=0,
            text=text,
            norm_stage=norm_stage,  # type: ignore[arg-type]
            chunk_mode="sliding_512",
        ))
        return chunks

    for i, start in enumerate(range(0, len(ids), step)):
        window_ids = ids[start:start + win]
        if len(window_ids) < 32:  # winzige Tail-Fragmente verwerfen
            break
        sub_text = tok.decode(window_ids, skip_special_tokens=True)
        chunks.append(Chunk(
            chunk_id=f"{letter.id}::{i}",
            letter_id=letter.id,
            idx=i,
            text=sub_text,
            norm_stage=norm_stage,  # type: ignore[arg-type]
            chunk_mode="sliding_512",
        ))
    return chunks


CHUNKERS = {"letter": chunk_letter, "sliding_512": chunk_sliding}


def apply(letter: Letter, text: str, mode: str, norm_stage: str) -> list[Chunk]:
    return CHUNKERS[mode](letter, text, norm_stage)
