"""CLI. Each pipeline stage is a separate command so you can rerun cheaply during the 22h."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import typer
from rich import print
from tqdm import tqdm

from jp_cluster.config import VARIANTS, settings
from jp_cluster.models import Letter

app = typer.Typer(add_completion=False, help="Jean-Paul-Cluster pipeline.")


@app.command()
def ingest(json_file: Path = typer.Argument(..., help="Path to extracted.json")) -> None:
    """Stage 1: extracted.json → letters.jsonl"""
    from jp_cluster.ingest.tei import iter_letters

    settings.ensure_dirs()
    out = settings.paths.interim / "letters.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for letter in iter_letters(json_file):
            f.write(letter.model_dump_json() + "\n")
            n += 1
    print(f"[green]wrote {n} letters → {out}[/green]")


def _load_letters() -> dict[str, Letter]:
    path = settings.paths.interim / "letters.jsonl"
    out: dict[str, Letter] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            ltr = Letter.model_validate_json(line)
            out[ltr.id] = ltr
    return out


@app.command()
def normalize_chunk(variant_id: str = typer.Option(..., help="z.B. v1, v2 …")) -> None:
    """Stage 2 + 3: normalisieren + chunken für eine Variante. Persistiert chunks_<id>.pkl."""
    from jp_cluster.chunk import strategies as ch
    from jp_cluster.normalize import pipeline as norm

    variant = next(v for v in VARIANTS if v.id == variant_id)
    letters = _load_letters()

    chunks = []
    for letter in tqdm(letters.values(), desc=f"norm+chunk {variant.id}"):
        text = norm.apply(letter, variant.norm)
        chunks.extend(ch.apply(letter, text, variant.chunk, variant.norm))

    out = settings.paths.processed / f"chunks_{variant.id}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(chunks, f)
    print(f"[green]{variant.id}: {len(chunks)} chunks → {out}[/green]")


@app.command()
def embed(variant_id: str) -> None:
    """Stage 4: chunks embedden + Chroma-Collection schreiben."""
    from jp_cluster.embed.embedder import Embedder, write_variant

    variant = next(v for v in VARIANTS if v.id == variant_id)
    chunks_path = settings.paths.processed / f"chunks_{variant.id}.pkl"
    with chunks_path.open("rb") as f:
        chunks = pickle.load(f)
    write_variant(variant, chunks, Embedder())
    print(f"[green]embedded → collection {variant.collection_name}[/green]")


@app.command()
def cluster(variant_id: str) -> None:
    """Stage 5: clustern + UMAP + Eval. Schreibt result_<id>.pkl + metrics_<id>.json."""
    from jp_cluster.cluster.algorithms import run
    from jp_cluster.embed.embedder import load_letter_matrix
    from jp_cluster.eval.metrics import evaluate

    variant = next(v for v in VARIANTS if v.id == variant_id)
    ids, X = load_letter_matrix(variant)
    letters = _load_letters()

    result = run(ids, X)
    metrics = evaluate(result, X, letters) | {"variant": variant.id}

    out_pkl = settings.paths.processed / f"result_{variant.id}.pkl"
    out_json = settings.paths.eval_out / f"metrics_{variant.id}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump({"result": result, "X": X}, f)
    out_json.write_text(json.dumps(metrics, indent=2))
    print(metrics)


@app.command(name="run-all")
def run_all() -> None:
    """Convenience: alle 6 Varianten end-to-end (nach `ingest`)."""
    for v in VARIANTS:
        normalize_chunk(v.id)
        embed(v.id)
        cluster(v.id)


@app.command()
def app_streamlit() -> None:
    """Startet das Streamlit-Frontend."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    str(Path(__file__).parent / "viz" / "app.py")])


if __name__ == "__main__":
    app()
