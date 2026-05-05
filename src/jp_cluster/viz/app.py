"""Streamlit-Prototyp.

Drei Tabs: Cluster-Scatter (UMAP), Netzwerk (k-NN), Brief-Detail.
Variant-Switcher in der Sidebar zeigt Eval-Metriken — der zentrale methodische Move:
Tool und Befund in einem Interface.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from jp_cluster.config import VARIANTS, settings
from jp_cluster.models.data import Letter
from jp_cluster.viz.network import knn_edges
from jp_cluster.vocab.lens import cluster_subject_heatmap, tfidf_labels

st.set_page_config(page_title="Jean Paul · Korrespondenz-Cluster", layout="wide")


@st.cache_data
def load_letters() -> dict[str, Letter]:
    path = settings.paths.interim / "letters.jsonl"
    out: dict[str, Letter] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            ltr = Letter.model_validate_json(line)
            out[ltr.id] = ltr
    return out


@st.cache_data
def load_variant(variant_id: str):
    pkl = settings.paths.processed / f"result_{variant_id}.pkl"
    with pkl.open("rb") as f:
        data = pickle.load(f)
    metrics_path = settings.paths.eval_out / f"metrics_{variant_id}.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return data["result"], data["X"], metrics


def all_metrics_df() -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        p = settings.paths.eval_out / f"metrics_{v.id}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()) | {
                "norm": v.norm, "chunk": v.chunk,
            })
    return pd.DataFrame(rows).set_index("variant") if rows else pd.DataFrame()


# ---------- Sidebar ----------

st.sidebar.title("Pipeline-Variante")
variant_id = st.sidebar.selectbox(
    "Variante",
    [v.id for v in VARIANTS],
    format_func=lambda vid: f"{vid} · {next(v for v in VARIANTS if v.id == vid).norm} / "
                            f"{next(v for v in VARIANTS if v.id == vid).chunk}",
)
algo = st.sidebar.radio("Cluster-Algorithmus", ["HDBSCAN", "Agglomerative"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Eval-Metriken (alle Varianten)**")
mdf = all_metrics_df()
if not mdf.empty:
    st.sidebar.dataframe(
        mdf[["silhouette_hdbscan", "ami_hdb_vs_addressee", "ami_hdb_vs_decade", "noise_ratio"]]
        .round(3),
        use_container_width=True,
    )

# ---------- Main ----------

letters_path = settings.paths.interim / "letters.jsonl"
result_path = settings.paths.processed / f"result_{variant_id}.pkl"

if not letters_path.exists():
    st.warning("No letters found. Run: `python -m jp_cluster.cli ingest data/raw/extracted.json`")
    st.stop()

if not result_path.exists():
    st.warning(f"No clustering results for {variant_id}. Run the pipeline first: `ingest → normalize-chunk → embed → cluster`")
    st.stop()

letters = load_letters()
result, X, metrics = load_variant(variant_id)
labels = result.labels_hdbscan if algo == "HDBSCAN" else result.labels_agglomerative

st.title("Jean Paul · semantische Cluster der Korrespondenz")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Briefe", metrics.get("n_letters", "–"))
c2.metric("Cluster (HDB)", metrics.get("n_clusters_hdb", "–"))
c3.metric("Silhouette", f"{metrics.get('silhouette_hdbscan', 0):.3f}")
c4.metric("AMI vs. Adressat", f"{metrics.get('ami_hdb_vs_addressee', 0):.3f}",
          help="Hoch = Cluster spiegeln vor allem Adressaten wider (trivial). "
               "Niedrig + hohe Silhouette = inhaltliche Struktur jenseits der Empfänger.")

tab_scatter, tab_net, tab_letter, tab_vocab = st.tabs(
    ["UMAP-Scatter", "Netzwerk", "Brief-Detail", "Vokabular-Lens"]
)

with tab_scatter:
    st.markdown(
        "Each point is one letter projected to 2D by UMAP. "
        "UMAP compresses the high-dimensional embedding (1024 dimensions) into a 2D map "
        "while preserving neighbourhood structure — letters written about similar topics "
        "appear close together. "
        "Color encodes cluster assignment. "
        "Points labelled **−1** (grey) are **noise**: letters HDBSCAN could not confidently "
        "place in any cluster. A high noise ratio usually means the corpus is too small or "
        "`min_cluster_size` needs lowering."
    )

    # Build explicit color map: noise → grey, clusters → large qualitative palette.
    # Alphabet has 26 entries, enough for agglomerative's 25 fixed clusters.
    unique_labels = sorted(set(labels.tolist()))
    palette = px.colors.qualitative.Alphabet
    color_map = {}
    color_idx = 0
    for lbl in unique_labels:
        if lbl == -1:
            color_map["-1"] = "#cccccc"
        else:
            color_map[str(lbl)] = palette[color_idx % len(palette)]
            color_idx += 1

    df = pd.DataFrame({
        "x": result.umap_xy[:, 0],
        "y": result.umap_xy[:, 1],
        "cluster number": labels.astype(str),
        "letter_id": result.letter_ids,
        "addressee": [letters[lid].addressee or "?" for lid in result.letter_ids],
        "year": [letters[lid].date_iso.year if letters[lid].date_iso else None
                 for lid in result.letter_ids],
    })
    fig = px.scatter(df, x="x", y="y", color="cluster number",
                     color_discrete_map=color_map,
                     hover_data=["letter_id", "addressee", "year"],
                     height=700)
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

with tab_net:
    st.markdown(
        "Each node is a letter; edges connect pairs whose embeddings exceed the cosine "
        "similarity threshold. Node color matches the cluster assignment from the scatter plot. "
        "**Threshold** controls how similar two letters must be to be connected — raise it to "
        "see only the strongest links, lower it to reveal broader thematic neighbourhoods. "
        "**k** sets the maximum number of neighbours considered per letter before the threshold "
        "filter is applied. Isolated letters (no edge above the threshold) are hidden."
    )
    threshold = st.slider("Cosine threshold", 0.3, 0.95, 0.65, 0.01)
    k = st.slider("k (top neighbours per letter)", 3, 20, 8)
    edges = knn_edges(X, result.letter_ids, k=k, threshold=threshold)
    if not edges:
        st.warning("No edges at this threshold — try lowering it.")
    else:
        st.write(f"**{len(edges)} edges** across {len(set(e for edge in edges for e in edge[:2]))} letters")
    show_isolated = st.checkbox("Show isolated nodes (no edges at current threshold)", value=False)

    # pyvis HTML einbetten
    try:
        from pyvis.network import Network

        net = Network(height="700px", width="100%", bgcolor="#ffffff", notebook=False)

        # Build a set of nodes that have at least one edge for fast lookup
        connected = {n for edge in edges for n in edge[:2]}
        label_to_lid = {lid: lbl for lid, lbl in zip(result.letter_ids, labels, strict=True)}

        for lid in result.letter_ids:
            if not show_isolated and lid not in connected:
                continue
            lbl = int(label_to_lid[lid])
            node_color = color_map.get(str(lbl), "#cccccc")
            net.add_node(
                lid,
                label=lid[:12],
                color=node_color,
                title=f"cluster {lbl}",
            )

        for s, t, w in edges:
            # width scales with similarity; title shows the exact score on hover
            net.add_edge(s, t, value=w, title=f"similarity: {w:.3f}")

        # Legend: explain edge width encoding
        st.caption("Edge thickness encodes cosine similarity — thicker edges mean more similar letters.")

        html_path = Path(settings.paths.processed) / "jp_net.html"
        net.save_graph(str(html_path))
        st.components.v1.html(html_path.read_text(), height=720, scrolling=True)
    except ImportError:
        st.warning("pyvis not installed — run `pip install pyvis`.")

with tab_letter:
    sel = st.selectbox("Brief", result.letter_ids)
    ltr = letters[sel]
    cl = int(labels[result.letter_ids.index(sel)])
    st.markdown(f"**Cluster:** {cl}  ·  **Adressat:** {ltr.addressee or '?'}  ·  "
                f"**Datum:** {ltr.date_raw or '?'}  ·  **Ort:** {ltr.place or '?'}")
    st.markdown("**Register-Tags:** " + (", ".join(ltr.register_terms) or "–"))
    st.text_area("Text", ltr.text_raw, height=400)

with tab_vocab:
    st.markdown(
        "This tab provides two lenses for interpreting what each cluster is *about*, "
        "complementing the geometric view in the scatter plot. "
        "**TF-IDF labels** extract the words and bigrams most characteristic of each cluster "
        "relative to the rest of the corpus — high-scoring terms appear frequently within the "
        "cluster but rarely elsewhere, making them good proxies for the cluster's theme. "
        "**GND heatmap** counts subject/entity tags (from the TEI register) per cluster, "
        "showing which named entities or topics are concentrated in which groups. "
        "Both views are most useful once you have a hypothesis from the scatter plot and want "
        "to put a label on a cluster."
    )
    top_k = st.slider("Terms per cluster", min_value=3, max_value=15, value=8)
    labels_dict = tfidf_labels(result.letter_ids, labels, letters, top_k=top_k)

    cluster_sizes = {int(c): int((labels == c).sum()) for c in sorted(set(labels)) if c != -1}

    st.markdown("### TF-IDF labels per cluster")
    st.dataframe(pd.DataFrame(
        [(c, cluster_sizes.get(c, 0), ", ".join(terms)) for c, terms in sorted(labels_dict.items())],
        columns=["Cluster", "Letters", "Top terms"],
    ), use_container_width=True)

    st.markdown("### Cluster × Register-Tag (GND)")
    heat = cluster_subject_heatmap(result.letter_ids, labels, letters)
    if not heat.empty:
        # Top 30 Spalten nach Häufigkeit
        top_cols = heat.sum().sort_values(ascending=False).head(30).index
        st.dataframe(heat[top_cols].astype(int), use_container_width=True)
    else:
        st.info("Keine Register-Tags im Korpus gefunden.")
