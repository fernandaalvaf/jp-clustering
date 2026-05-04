"""Canonical data records. Everything between pipeline stages flows through these."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Letter(BaseModel):
    """One letter from the Jean-Paul-Edition. Output of `ingest`, input to everything else."""

    id: str                              # stable ID, e.g. TEI xml:id
    sender: str | None = None
    addressee: str | None = None
    date_iso: date | None = None         # normalised date for binning
    date_raw: str | None = None          # original date string (oft unscharf bei JP)
    place: str | None = None
    text_raw: str                        # transcribed body, no markup
    register_terms: list[str] = Field(default_factory=list)  # GND-IDs / register entries
    tei_path: str | None = None          # for traceback


class Chunk(BaseModel):
    """One embeddable unit. Either a whole letter or a sliding window."""

    chunk_id: str                        # f"{letter_id}::{idx}"
    letter_id: str
    idx: int                             # 0 for whole-letter chunks
    text: str                            # post-normalisation, ready to embed
    norm_stage: Literal["raw", "transnormer", "transnormer_lemma"]
    chunk_mode: Literal["letter", "sliding_512"]


class ClusterAssignment(BaseModel):
    letter_id: str
    cluster: int                         # -1 = HDBSCAN noise
    probability: float | None = None
