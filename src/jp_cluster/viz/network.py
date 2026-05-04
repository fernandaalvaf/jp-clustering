"""kNN-Ähnlichkeitsnetzwerk. Cosine-Schwelle als Slider im Streamlit-Frontend."""

from __future__ import annotations

import numpy as np


def knn_edges(
    X: np.ndarray,
    letter_ids: list[str],
    k: int = 8,
    threshold: float = 0.6,
) -> list[tuple[str, str, float]]:
    """Top-k Cosine-Nachbarn pro Brief, gefiltert nach Schwelle.
    X muss L2-normalisiert sein (siehe embedder._l2)."""
    sims = X @ X.T
    np.fill_diagonal(sims, -1.0)
    edges: list[tuple[str, str, float]] = []
    for i, src in enumerate(letter_ids):
        nn_idx = np.argpartition(-sims[i], k)[:k]
        for j in nn_idx:
            s = float(sims[i, j])
            if s >= threshold and i < j:  # ungerichtet, je Paar nur einmal
                edges.append((src, letter_ids[j], s))
    return edges
