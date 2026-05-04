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


class Variant(BaseModel):
    """One row of the variant matrix. ID is stable, used as Chroma collection name."""

    id: str
    norm: NormStage
    chunk: ChunkMode

    @property
    def collection_name(self) -> str:
        return f"jp_{self.norm}_{self.chunk}"


# 3 norm stages × 2 chunk modes = 6 variants
VARIANTS: list[Variant] = [
    Variant(id="v1", norm="raw", chunk="letter"),
    Variant(id="v2", norm="raw", chunk="sliding_512"),
    Variant(id="v3", norm="transnormer", chunk="letter"),
    Variant(id="v4", norm="transnormer", chunk="sliding_512"),
    Variant(id="v5", norm="transnormer_lemma", chunk="letter"),
    Variant(id="v6", norm="transnormer_lemma", chunk="sliding_512"),
]


class EmbedCfg(BaseModel):
    # multilingual-e5 trägt für historisches DE meist robuster als jina-v3 in unseren correspSearch-Tests
    model: str = "intfloat/multilingual-e5-large-instruct"
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
