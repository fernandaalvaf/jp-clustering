"""Clustering algorithms + UMAP projection for visualisation.

Input: a letter-level embedding matrix X of shape [n_letters, embedding_dim],
produced by embedder.load_letter_matrix() (chunk embeddings mean-pooled per letter).

Two clustering algorithms are run in parallel so results can be compared:
  - HDBSCAN: density-based, discovers the number of clusters automatically,
    assigns noise points label -1.
  - Agglomerative: hierarchical, fixed cluster count, no noise — used as a
    reference baseline and for dendrogram-style analysis.

UMAP is run separately for 2D visualisation only. It is intentionally NOT used
as a preprocessing step for clustering: reducing dimensions before HDBSCAN would
distort the density structure that HDBSCAN relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jp_cluster.config import settings


@dataclass
class ClusterResult:
    """All outputs from one clustering run, keyed by letter_id order.

    letter_ids: ordered list of letter IDs; the index into this list is the
        row index for all arrays below.
    labels_hdbscan: cluster label per letter (-1 = noise / unassigned).
    labels_agglomerative: cluster label per letter (no noise, always assigned).
    umap_xy: 2D coordinates for visualisation, shape [n_letters, 2].
    probabilities: HDBSCAN soft-membership score in [0, 1]; how confidently
        each letter belongs to its assigned cluster (0 for noise points).
    """

    letter_ids: list[str]
    labels_hdbscan: np.ndarray
    labels_agglomerative: np.ndarray
    umap_xy: np.ndarray
    probabilities: np.ndarray


def run(letter_ids: list[str], X: np.ndarray) -> ClusterResult:
    """Cluster letter embeddings and project them to 2D.

    Args:
        letter_ids: ordered list of letter IDs matching the rows of X.
        X: L2-normalised embedding matrix, shape [n_letters, embedding_dim].
           Euclidean distance on L2-normalised vectors equals cosine distance,
           so euclidean metric is used for HDBSCAN.

    Returns:
        ClusterResult with HDBSCAN labels, agglomerative labels, UMAP
        coordinates, and HDBSCAN soft-membership probabilities.
    """
    import hdbscan
    import umap
    from sklearn.cluster import AgglomerativeClustering

    cfg = settings.cluster

    # Primary method: density-based, discovers cluster count automatically.
    # prediction_data=True enables soft-membership probabilities.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric=cfg.hdbscan_metric,
        prediction_data=True,
    )
    labels_hdb = clusterer.fit_predict(X)

    # Reference baseline: fixed cluster count, no noise points.
    # Average linkage with cosine metric handles the high-dimensional space well.
    agg = AgglomerativeClustering(
        n_clusters=cfg.agglomerative_n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels_agg = agg.fit_predict(X)

    # 2D projection for the Streamlit scatter plot. Fitted on the full
    # high-dimensional X — not on cluster labels — to preserve global structure.
    reducer = umap.UMAP(
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.random_state,
    )
    xy = reducer.fit_transform(X)

    return ClusterResult(
        letter_ids=letter_ids,
        labels_hdbscan=labels_hdb,
        labels_agglomerative=labels_agg,
        umap_xy=np.asarray(xy),
        probabilities=clusterer.probabilities_,
    )
