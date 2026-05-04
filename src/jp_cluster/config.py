"""Central configuration. The 6-variant matrix lives here so notebook + app + CLI agree."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


class Paths(BaseModel):
    raw: Path = DATA / "raw"            # TEI XML, original
    interim: Path = DATA / "interim"    # extracted plaintext + metadata
    processed: Path = DATA / "processed"  # normalised + chunked
    chroma: Path = DATA / "processed" / "chroma"
    eval_out: Path = DATA / "processed" / "eval"


NormStage = Literal["raw", "transnormer", "transnormer_lemma"]
ChunkMode = Literal["letter", "sliding_512"]

# Short, Chroma-safe slugs for known model IDs
_MODEL_SLUGS: dict[str, str] = {
    "intfloat/multilingual-e5-large-instruct": "me5l",
    "intfloat/multilingual-e5-base": "me5b",
    "jinaai/jina-embeddings-v3": "jinav3",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet",
    "codefuse-ai/F2LLM-v2-14B": "f2llm",
}

# Models that must be loaded with bfloat16 to fit in GPU memory
BFLOAT16_MODELS: frozenset[str] = frozenset({
    "codefuse-ai/F2LLM-v2-14B",
})


def _model_slug(model_id: str) -> str:
    """Return a short Chroma-safe slug. Falls back to the part after '/' lowercased."""
    if model_id in _MODEL_SLUGS:
        return _MODEL_SLUGS[model_id]
    base = model_id.split("/")[-1].lower()
    # keep only alphanumeric, dots, hyphens, underscores; max 20 chars
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in base)[:20]
    return safe


class Variant(BaseModel):
    """One cell of the variant matrix: norm × chunk × model."""

    id: str
    norm: NormStage
    chunk: ChunkMode
    model: str  # HuggingFace model ID

    @property
    def model_slug(self) -> str:
        return _model_slug(self.model)

    @property
    def collection_name(self) -> str:
        return self.id


def _build_variants(
    norms: list[NormStage],
    chunks: list[ChunkMode],
    models: list[str],
) -> list[Variant]:
    """Generate all norm × chunk × model combinations.

    The variant id equals the Chroma collection name:
    ``jp_<norm>_<chunk>_<model_slug>``, e.g. ``jp_raw_letter_me5l``.
    """
    rows: list[Variant] = []
    for norm in norms:
        for chunk in chunks:
            for model in models:
                vid = f"jp_{norm}_{chunk}_{_model_slug(model)}"
                rows.append(Variant(id=vid, norm=norm, chunk=chunk, model=model))
    return rows


class EmbedCfg(BaseModel):
    # Candidate models to evaluate; first entry is the default / fastest
    models: list[str] = [
        "intfloat/multilingual-e5-large-instruct",
        "intfloat/multilingual-e5-base",
        "jinaai/jina-embeddings-v3",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "codefuse-ai/F2LLM-v2-14B",
    ]
    # Convenience accessor — use models[0] as the single-model default
    @property
    def model(self) -> str:
        return self.models[0]

    batch_size: int = 16
    device: str = "cuda"  # "cpu" als Fallback, CLI-Override
    # E5 verlangt Prefix; "passage:" für indexierte Dokumente, "query:" für Suchanfragen
    passage_prefix: str = "passage: "
    query_prefix: str = "query: "
    max_seq_len: int = 512


class ChunkCfg(BaseModel):
    sliding_tokens: int = 512
    sliding_overlap: int = 64
    min_chunk_chars: int = 80


class ClusterCfg(BaseModel):
    # HDBSCAN-Defaults für ~1–5k Briefe; bei kleinerem Korpus über CLI senken
    hdbscan_min_cluster_size: int = 8
    hdbscan_min_samples: int = 3
    hdbscan_metric: str = "euclidean"  # auf L2-normalisierten Embeddings ≈ cosine
    agglomerative_n_clusters: int = 25
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    random_state: int = 42


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JPC_", env_file=".env", extra="ignore")

    paths: Paths = Paths()
    embed: EmbedCfg = EmbedCfg()
    chunk: ChunkCfg = ChunkCfg()
    cluster: ClusterCfg = ClusterCfg()
    corpus_source: str = "https://www.jeanpaul-edition.de/"  # Platzhalter; via CLI/env überschreibbar

    def ensure_dirs(self) -> None:
        for p in [self.paths.raw, self.paths.interim, self.paths.processed,
                  self.paths.chroma, self.paths.eval_out]:
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()

# Build the variant matrix from the active config at import time
VARIANTS: list[Variant] = _build_variants(
    norms=["raw", "transnormer", "transnormer_lemma"],
    chunks=["letter", "sliding_512"],
    models=settings.embed.models,
)
