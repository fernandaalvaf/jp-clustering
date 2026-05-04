"""Clustering algorithms + UMAP projection for visualisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jp_cluster.config import settings


@dataclass
class ClusterResult:
    letter_ids: list[str]
    labels_hdbscan: np.ndarray
    labels_agglomerative: np.ndarray
    umap_xy: np.ndarray
    probabilities: np.ndarray  # HDBSCAN soft-membership


def run(letter_ids: list[str], X: np.ndarray) -> ClusterResult:
    import hdbscan
    import umap
    from sklearn.cluster import AgglomerativeClustering

    cfg = settings.cluster

    # HDBSCAN — Hauptverfahren, erlaubt Noise (-1)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.hdbscan_min_cluster_size,
        min_samples=cfg.hdbscan_min_samples,
        metric=cfg.hdbscan_metric,
        prediction_data=True,
    )
    labels_hdb = clusterer.fit_predict(X)

    # Agglomerative — feste Clusterzahl, gut für Dendrogramm-Vergleich
    agg = AgglomerativeClustering(
        n_clusters=cfg.agglomerative_n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels_agg = agg.fit_predict(X)

    # UMAP nur für Viz, NICHT für Clustering (würde Dichte verfälschen)
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
