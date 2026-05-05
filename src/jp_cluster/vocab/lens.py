"""Kontrolliertes Vokabular als Lens (nicht als Embedding-Input).

Zwei Anwendungen:
  1. Cluster × GND-Subject-Heatmap (welche Themen dominieren welchen Cluster?)
  2. TF-IDF-basierte Cluster-Labels aus dem Vokabular (statt aller Wörter)

Erwartete Vokabular-Quelle: `data/raw/vocab.csv` mit Spalten {term, gnd_id, label}.
Wenn nicht vorhanden, wird aus den Register-Tags der Briefe selbst extrahiert.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from jp_cluster.models.data import Letter


def load_vocab(path: Path | None) -> pd.DataFrame:
    if path and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["term", "gnd_id", "label"])


def cluster_subject_heatmap(
    letter_ids: list[str],
    labels: np.ndarray,
    letters: dict[str, Letter],
) -> pd.DataFrame:
    """Counts of register_terms per cluster. Zeilen: cluster, Spalten: term."""
    counts: dict[int, Counter] = defaultdict(Counter)
    for lid, lbl in zip(letter_ids, labels, strict=True):
        for term in letters[lid].register_terms:
            counts[int(lbl)][term] += 1
    df = pd.DataFrame(counts).fillna(0).T.sort_index()
    df.index.name = "cluster"
    return df


def tfidf_labels(
    letter_ids: list[str],
    labels: np.ndarray,
    letters: dict[str, Letter],
    top_k: int = 5,
) -> dict[int, list[str]]:
    """TF-IDF auf Brieftexten, Top-Terme pro Cluster als Label."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    from jp_cluster.clean.cleaner import STOP_WORDS
    stop_words = list(STOP_WORDS)

    docs = [letters[lid].text_raw for lid in letter_ids]
    vec = TfidfVectorizer(max_df=0.7, min_df=3, ngram_range=(1, 2), stop_words=stop_words)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    out: dict[int, list[str]] = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        mask = labels == c
        mean_tfidf = np.asarray(X[mask].mean(axis=0)).ravel()
        top = terms[np.argsort(-mean_tfidf)[:top_k]]
        out[int(c)] = top.tolist()
    return out
