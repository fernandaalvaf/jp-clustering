# Jean-Paul-Cluster

Semantische Cluster-Analyse der Korrespondenz Jean Pauls. TELOTA-Challenge-Prototyp (22 h).

## Setup

```bash
uv venv && source .venv/bin/activate          # oder python -m venv
uv pip install -e ".[dev]"
python -m spacy download de_core_news_lg
# optional, für Stage 2/3 der Normalisierung:
pip install transnormer  # bzw. transformers-Modell direkt ziehen
```

## Pipeline

```bash
# 1. TEI ingest
jpc ingest data/raw/jp-tei/

# 2-5. Eine Variante end-to-end
jpc normalize-chunk jp_raw_letter_me5l
jpc embed jp_raw_letter_me5l
jpc cluster jp_raw_letter_me5l

# Alle 6 Varianten
jpc run-all

# Streamlit-Frontend
jpc app-streamlit
```

## Variantenmatrix

| ID                              | Normalisierung      | Chunking      |
|---------------------------------|---------------------|---------------|
| jp_raw_letter_me5l              | raw                 | letter        |
| jp_raw_sliding_512_me5l         | raw                 | sliding 512   |
| jp_transnormer_letter_me5l      | transnormer         | letter        |
| jp_transnormer_sliding_512_me5l | transnormer         | sliding 512   |
| jp_transnormer_lemma_letter_me5l      | transnormer + lemma | letter        |
| jp_transnormer_lemma_sliding_512_me5l | transnormer + lemma | sliding 512   |

Jede Variante schreibt in eine eigene Chroma-Collection und produziert ein Eval-JSON.

## 22-h-Schedule

| Stunden | Aufgabe | Kommando / Datei |
|---------|---------|------------------|
| 0–2     | TEI-Pull, Schema klären, XPaths in `ingest/tei.py` finalisieren | `jpc ingest …` |
| 2–5     | Normalizer testen (transnormer auf Sample, Lemmatizer-Sanity) | `notebooks/00_norm_sanity.ipynb` |
| 5–9     | 6× embed-Läufe, idealerweise auf telota-ai (CUDA) | `jpc run-all` |
| 9–12    | Eval-Tabelle, Notebook-Plots | `notebooks/01_eval_variants.ipynb` |
| 12–16   | Streamlit-App, pyvis-Netzwerk feinschleifen | `jpc app-streamlit` |
| 16–18   | Vokabular-Lens, GND-Heatmap | `vocab/lens.py`, App-Tab |
| 18–20   | Eval-Notebook ausarbeiten, Diskussion | `notebooks/01_eval_variants.ipynb` |
| 20–22   | README, Demo-GIF, ggf. Deployment | – |

## Offene Punkte für Stunde 0

- [ ] TEI-Schema der Jean-Paul-Edition: konkrete XPaths in `ingest/tei.py:parse_letter` setzen
- [ ] Datierungs-Konvention klären (`@when`, `@notBefore/@notAfter`, freier Text)
- [ ] Register-Quelle: GND-URIs in `@ref`, oder externes Register?
- [ ] GPU vs. CPU: bei CPU `JPC_EMBED__DEVICE=cpu` und Batchsize runter
