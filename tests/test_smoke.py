"""Smoke tests — laufen ohne echtes Korpus und prüfen das Pipeline-Wiring."""

from datetime import date

from jp_cluster.config import VARIANTS
from jp_cluster.models import Letter


def test_variants_unique_collection_names() -> None:
    names = [v.collection_name for v in VARIANTS]
    assert len(set(names)) == len(names)


def test_letter_roundtrip() -> None:
    ltr = Letter(
        id="L0001", sender="Jean Paul", addressee="Karoline",
        date_iso=date(1801, 5, 14), date_raw="14. May 1801",
        text_raw="Liebste Karoline, …", register_terms=["#GND/118557823"],
    )
    js = ltr.model_dump_json()
    assert Letter.model_validate_json(js) == ltr


def test_chunk_letter_mode_no_tokenizer_needed() -> None:
    """letter-mode darf keinen HF-Tokenizer-Download triggern."""
    from jp_cluster.chunk import strategies as ch
    ltr = Letter(id="L1", text_raw="x" * 200)
    chunks = ch.chunk_letter(ltr, "x " * 100, "raw")
    assert len(chunks) == 1
    assert chunks[0].chunk_mode == "letter"
