"""Embedding & Chroma persistence.

Wichtige Designentscheidung: Wir embedden auf Chunk-Ebene, persistieren Chunk-Vektoren
in Chroma (für spätere Suche), poolen aber für das Clustering auf Brief-Ebene zurück
(Mean der L2-normalisierten Chunk-Vektoren). So bleibt die Cluster-Einheit der Brief —
wie es die Challenge fordert ("auf Dokumentenebene") — und Chunking wirkt nur als
Encoding-Strategie, nicht als Granularitätswechsel.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from jp_cluster.config import Variant, settings
from jp_cluster.models import Chunk


class Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embed.model, device=settings.embed.device)
        self.model.max_seq_length = settings.embed.max_seq_len

    def encode(self, texts: list[str]) -> np.ndarray:
        prefixed = [settings.embed.passage_prefix + t for t in texts]
        emb = self.model.encode(
            prefixed,
            batch_size=settings.embed.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-norm; Cosine == Dot-Product
        )
        return emb


def _client(persist_dir: Path) -> chromadb.api.ClientAPI:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def write_variant(variant: Variant, chunks: list[Chunk], embedder: Embedder) -> None:
    """Embed all chunks of one variant and persist to its Chroma collection."""
    client = _client(settings.paths.chroma)
    coll = client.get_or_create_collection(
        name=variant.collection_name,
        metadata={"norm": variant.norm, "chunk": variant.chunk, "model": settings.embed.model},
    )

    bs = 64
    for i in tqdm(range(0, len(chunks), bs), desc=f"embed {variant.id}"):
        batch = chunks[i:i + bs]
        embs = embedder.encode([c.text for c in batch])
        coll.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=embs.tolist(),
            documents=[c.text for c in batch],
            metadatas=[{"letter_id": c.letter_id, "idx": c.idx} for c in batch],
        )


def load_letter_matrix(variant: Variant) -> tuple[list[str], np.ndarray]:
    """Pool chunks back to letter-level. Returns (letter_ids, matrix [n_letters, dim])."""
    client = _client(settings.paths.chroma)
    coll = client.get_collection(name=variant.collection_name)
    data = coll.get(include=["embeddings", "metadatas"])

    by_letter: dict[str, list[np.ndarray]] = {}
    for emb, meta in zip(data["embeddings"], data["metadatas"], strict=True):
        by_letter.setdefault(meta["letter_id"], []).append(np.asarray(emb, dtype=np.float32))

    letter_ids = sorted(by_letter)
    pooled = np.stack([
        # Mean → re-normalise, damit Cosine-Geometrie erhalten bleibt
        _l2(np.mean(np.stack(by_letter[lid]), axis=0)) for lid in letter_ids
    ])
    return letter_ids, pooled


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
