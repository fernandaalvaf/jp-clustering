"""Normalisation pipeline. Three stages — raw, transnormer, transnormer+lemma.

transnormer (https://github.com/ybracke/transnormer) ist ein seq2seq-Modell für
historisches Deutsch (1700–1900). Falls das Setup zu schwer wird, ist als Fallback
ein einfacher Regel-Normalizer skizziert. Lemmatisierung via spaCy `de_core_news_lg`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from jp_cluster.models import Letter


class Normalizer(Protocol):
    def __call__(self, text: str) -> str: ...


# ---------- Stage 1: raw (pass-through, leichtes Whitespace-Cleanup) ----------

_WS = re.compile(r"\s+")


def normalize_raw(text: str) -> str:
    return _WS.sub(" ", text).strip()


# ---------- Stage 2: transnormer ----------

@lru_cache(maxsize=1)
def _transnormer():
    """Lazy load. Falls Modell nicht greifbar, fällt der Caller auf raw zurück."""
    try:
        from transformers import pipeline
        # Modell-ID ggf. an aktuell verfügbares Release anpassen
        return pipeline(
            "text2text-generation",
            model="ybracke/transnormer-19c-beta-v02",
            max_length=512,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[normalize] transnormer nicht verfügbar ({e}); raw fallback aktiv")
        return None


def normalize_transnormer(text: str) -> str:
    pipe = _transnormer()
    if pipe is None:
        return normalize_raw(text)
    # In Sätzen verarbeiten — Modell ist auf kurze Spannen trainiert
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        try:
            res = pipe(sent)[0]["generated_text"]
            out.append(res.strip())
        except Exception:  # noqa: BLE001
            out.append(sent)
    return normalize_raw(" ".join(out))


# ---------- Stage 3: transnormer + lemma ----------

@lru_cache(maxsize=1)
def _spacy():
    try:
        import spacy
        return spacy.load("de_core_news_lg", disable=["ner", "parser"])
    except Exception as e:  # noqa: BLE001
        print(f"[normalize] spaCy de_core_news_lg fehlt ({e}); python -m spacy download de_core_news_lg")
        return None


def normalize_transnormer_lemma(text: str) -> str:
    normalised = normalize_transnormer(text)
    nlp = _spacy()
    if nlp is None:
        return normalised
    doc = nlp(normalised)
    return " ".join(tok.lemma_.lower() for tok in doc if not tok.is_space and not tok.is_punct)


# ---------- Dispatch ----------

NORMALIZERS: dict[str, Normalizer] = {
    "raw": normalize_raw,
    "transnormer": normalize_transnormer,
    "transnormer_lemma": normalize_transnormer_lemma,
}


def apply(letter: Letter, stage: str) -> str:
    return NORMALIZERS[stage](letter.text_raw)


def read_precomputed(letter: Letter, stage: str, norm_base: Path) -> str | None:
    """Read pre-computed normalized text from disk; return None if not available.

    ``transnormer_lemma`` piggybacks on the transnormer files and applies
    spaCy lemmatisation on top so that we never have to rerun the model.
    ``raw`` has no pre-computed files — the caller should use ``letter.text_raw``.
    """
    src_stage = "transnormer" if stage == "transnormer_lemma" else stage
    if src_stage == "raw":
        return None

    norm_file = norm_base / src_stage / f"{letter.id}.normalized.txt"
    if not norm_file.exists():
        return None

    text = norm_file.read_text(encoding="utf-8").strip()

    if stage == "transnormer_lemma":
        nlp = _spacy()
        if nlp is not None:
            doc = nlp(text)
            return " ".join(
                tok.lemma_.lower() for tok in doc if not tok.is_space and not tok.is_punct
            )

    return text
