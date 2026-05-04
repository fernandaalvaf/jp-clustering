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
from jp_cluster.models import Letter
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
    df = pd.DataFrame({
        "x": result.umap_xy[:, 0],
        "y": result.umap_xy[:, 1],
        "cluster": labels.astype(str),
        "letter_id": result.letter_ids,
        "addressee": [letters[lid].addressee or "?" for lid in result.letter_ids],
        "year": [letters[lid].date_iso.year if letters[lid].date_iso else None
                 for lid in result.letter_ids],
    })
    fig = px.scatter(df, x="x", y="y", color="cluster",
                     hover_data=["letter_id", "addressee", "year"],
                     height=700)
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

with tab_net:
    threshold = st.slider("Cosine-Schwelle", 0.3, 0.95, 0.65, 0.01)
    k = st.slider("k (Top-Nachbarn)", 3, 20, 8)
    edges = knn_edges(X, result.letter_ids, k=k, threshold=threshold)
    st.write(f"**{len(edges)} Kanten** über {len(set(e for edge in edges for e in edge[:2]))} Briefen")
    # pyvis HTML einbetten
    try:
        from pyvis.network import Network
        net = Network(height="700px", width="100%", bgcolor="#ffffff", notebook=False)
        nodes = {lid: int(lbl) for lid, lbl in zip(result.letter_ids, labels, strict=True)}
        for lid, cl in nodes.items():
            if any(lid in e[:2] for e in edges):
                net.add_node(lid, label=lid[:12], group=cl,
                             title=f"{letters[lid].addressee or '?'} · {letters[lid].date_raw or ''}")
        for s, t, w in edges:
            net.add_edge(s, t, value=w)
        html_path = Path("/tmp/jp_net.html")
        net.save_graph(str(html_path))
        st.components.v1.html(html_path.read_text(), height=720, scrolling=True)
    except ImportError:
        st.warning("pyvis nicht installiert.")

with tab_letter:
    sel = st.selectbox("Brief", result.letter_ids)
    ltr = letters[sel]
    cl = int(labels[result.letter_ids.index(sel)])
    st.markdown(f"**Cluster:** {cl}  ·  **Adressat:** {ltr.addressee or '?'}  ·  "
                f"**Datum:** {ltr.date_raw or '?'}  ·  **Ort:** {ltr.place or '?'}")
    st.markdown("**Register-Tags:** " + (", ".join(ltr.register_terms) or "–"))
    st.text_area("Text", ltr.text_raw, height=400)

with tab_vocab:
    labels_dict = tfidf_labels(result.letter_ids, labels, letters, top_k=8)
    st.markdown("### TF-IDF-Labels pro Cluster")
    st.dataframe(pd.DataFrame(
        [(c, ", ".join(terms)) for c, terms in sorted(labels_dict.items())],
        columns=["Cluster", "Top-Terme"],
    ), use_container_width=True)

    st.markdown("### Cluster × Register-Tag (GND)")
    heat = cluster_subject_heatmap(result.letter_ids, labels, letters)
    if not heat.empty:
        # Top 30 Spalten nach Häufigkeit
        top_cols = heat.sum().sort_values(ascending=False).head(30).index
        st.dataframe(heat[top_cols].astype(int), use_container_width=True)
    else:
        st.info("Keine Register-Tags im Korpus gefunden.")
