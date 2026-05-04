"""
Text splitting utilities.

Provides sentence-level chunking that keeps each chunk within a configurable
byte limit. Used by the Transnormer normalizer and available for reuse in
other pipeline stages (e.g. chunking strategies).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

DEFAULT_SPACY_MODEL = "de_core_news_sm"

# Soft-break characters used to subdivide overlong sentences before they
# blow past a model's input window.
SOFT_BREAK_RE = re.compile(r"(?<=[;:–—)])\s+|(?<=,)\s+")

# Hard chunk limit (bytes). The byte-level transnormer model has a
# 1024-token input window; staying well under it avoids silent
# truncation on long passages with no punctuation.
DEFAULT_MAX_CHUNK_BYTES = 400

# Lazy spaCy pipeline (loaded once per process).
_SPACY_NLP = None


def _get_spacy(model_name: str = DEFAULT_SPACY_MODEL):
    """Load and cache a spaCy pipeline for sentence splitting."""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    import spacy  # local import: heavy and only needed here

    try:
        nlp = spacy.load(model_name, disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"])
        if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    except OSError:
        logger.warning(
            "spaCy model %r not installed; falling back to blank 'de' + sentencizer. "
            "Install with: python -m spacy download %s",
            model_name, model_name,
        )
        nlp = spacy.blank("de")
        nlp.add_pipe("sentencizer")
    # Allow long historical letters without raising.
    nlp.max_length = 2_000_000
    _SPACY_NLP = nlp
    return nlp


def hard_chunk(s: str, max_bytes: int) -> list[str]:
    """Greedy whitespace chunker as a last resort for run-on text."""
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for word in s.split():
        wlen = len(word.encode("utf-8")) + (1 if cur else 0)
        if cur and cur_len + wlen > max_bytes:
            out.append(" ".join(cur))
            cur, cur_len = [word], len(word.encode("utf-8"))
        else:
            cur.append(word)
            cur_len += wlen
    if cur:
        out.append(" ".join(cur))
    return out


def split_sentences(text: str, max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES) -> list[str]:
    """
    Split text into sentences using the spaCy German sentencizer.
    Long sentences are progressively subdivided (soft punctuation,
    then whitespace) so no chunk exceeds *max_chunk_bytes* —
    protecting the model's input window.
    """
    nlp = _get_spacy()
    raw = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    out: list[str] = []
    for s in raw:
        if len(s.encode("utf-8")) <= max_chunk_bytes:
            out.append(s)
            continue
        # Try soft breaks (commas, semicolons, dashes …)
        parts = [p.strip() for p in SOFT_BREAK_RE.split(s) if p.strip()]
        for p in parts:
            if len(p.encode("utf-8")) <= max_chunk_bytes:
                out.append(p)
            else:
                out.extend(hard_chunk(p, max_chunk_bytes))
    return out
