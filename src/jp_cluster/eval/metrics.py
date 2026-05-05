"""Evaluation metrics. Das ist die zentrale methodische Erzählung des Notebooks:

  - Silhouette: interne Cluster-Kompaktheit
  - AMI vs. Korrespondent: trivialer Trigger? Hoch = Cluster ≈ Adressat (langweilig)
  - AMI vs. Dekade: zeitliche Strukturierung
  - Differenz Silhouette − AMI(Korrespondent): zeigt, ob Embeddings ZUSÄTZLICHE
    inhaltliche Struktur jenseits der Adressaten finden
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from jp_cluster.cluster.algorithms import ClusterResult
from jp_cluster.models.data import Letter


def _decade(year: int | None) -> int | None:
    return (year // 10) * 10 if year else None


def evaluate(result: ClusterResult, X: np.ndarray, letters: dict[str, Letter]) -> dict:
    addressees = [letters[lid].addressee or "?" for lid in result.letter_ids]
    decades = [_decade(letters[lid].date_iso.year if letters[lid].date_iso else None)
               for lid in result.letter_ids]

    # Silhouette nur über non-noise-Punkte sinnvoll
    mask = result.labels_hdbscan != -1
    sil_hdb = (
        silhouette_score(X[mask], result.labels_hdbscan[mask], metric="cosine")
        if mask.sum() > 1 and len(set(result.labels_hdbscan[mask])) > 1
        else float("nan")
    )
    sil_agg = silhouette_score(X, result.labels_agglomerative, metric="cosine")

    return {
        "n_letters": len(result.letter_ids),
        "n_clusters_hdb": int(len(set(result.labels_hdbscan)) - (1 if -1 in result.labels_hdbscan else 0)),
        "noise_ratio": float((result.labels_hdbscan == -1).mean()),
        "silhouette_hdbscan": float(sil_hdb),
        "silhouette_agglomerative": float(sil_agg),
        "ami_hdb_vs_addressee": float(adjusted_mutual_info_score(addressees, result.labels_hdbscan)),
        "ami_hdb_vs_decade": float(adjusted_mutual_info_score(
            [str(d) for d in decades], result.labels_hdbscan)),
        "ami_agg_vs_addressee": float(adjusted_mutual_info_score(addressees, result.labels_agglomerative)),
        "ami_agg_vs_decade": float(adjusted_mutual_info_score(
            [str(d) for d in decades], result.labels_agglomerative)),
    }


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("variant")
