"""
Takes normalized texts from data/processed/normalized/
and cleans them by removing stopwords and saves them to data/processed/cleaned/,
observing the directory structure of the normalized texts folder.

Usage:
    python -m jp_cluster.clean.cleaner              # cleans all variants
    python -m jp_cluster.clean.cleaner transnormer  # cleans one variant
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Stop word list — modern German + 18th-century orthographic variants.
# Defined here as a module-level constant so lens.py can import it too.
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset({
    # articles
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer", "eines",
    # personal pronouns
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mich", "mir", "dich", "dir", "ihn", "ihm",
    "uns", "euch", "ihnen", "sich",
    # demonstrative / relative
    "dieser", "diese", "dieses", "diesem", "diesen",
    "jener", "jene", "jenes", "jenem", "jenen",
    "welcher", "welche", "welches", "welchem", "welchen",
    # prepositions
    "an", "auf", "aus", "bei", "bey", "bis", "durch",
    "für", "gegen", "in", "mit", "nach", "ohne", "seit",
    "über", "um", "unter", "von", "vor", "zu", "zwischen",
    "vom", "zum", "zur", "im", "ins", "am", "ans", "beim",
    # conjunctions
    "und", "oder", "aber", "denn", "weil", "wenn", "daß",
    "dass", "ob", "als", "wie", "wo", "wer", "was",
    "indem", "indeß", "indessen", "sondern", "jedoch",
    "obwohl", "damit", "sodass", "sowohl", "entweder",
    # auxiliary / modal verbs
    "ist", "sind", "war", "waren", "hat", "haben", "hatte",
    "hatten", "wird", "werden", "wurde", "wurden", "worden",
    "kann", "kan", "können", "konnte", "konnten",
    "muß", "muss", "müssen", "musste",
    "soll", "sollen", "sollte", "wollen", "will", "wollte",
    "darf", "dürfen", "durfte", "mag", "mögen", "möchte",
    "sey", "seyn", "seyne", "seynd", "sei", "seien",
    "bin", "bist", "wäre", "wären",
    # adverbs / particles
    "nicht", "auch", "noch", "schon", "sehr", "viel", "mehr",
    "nur", "so", "ja", "nein", "nun", "dann", "doch", "eben",
    "gleich", "wieder", "immer", "hier", "da", "dort", "wohl",
    "zwar", "jezt", "jetzt", "iezt", "ietzt", "nehmlich",
    "nämlich", "freylich", "freilich", "bereits", "fast",
    "ganz", "gar", "kaum", "leider", "mal", "nie",
    "oft", "stets", "vielleicht", "weit", "wenig",
    # possessive / indefinite determiners
    "kein", "keine", "keinen", "keinem", "keiner", "keines",
    "mein", "meine", "meinen", "meinem", "meiner", "meines",
    "dein", "deine", "deinen", "deinem", "deiner",
    "sein", "seine", "seinen", "seinem", "seiner",
    "ihre", "ihren", "ihrem", "ihrer",
    "unser", "unsere", "unseren", "unserem", "unserer",
    "euer", "eure", "euren", "eurem", "eurer",
    "alles", "alle", "allen", "allem", "aller",
    # other common function words
    "man", "etc", "lassen", "laßen", "macht", "machen",
})

# Matches sequences of word characters including German umlauts and ß
_TOKEN_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, remove stop words. Returns space-joined tokens."""
    stripped = _TOKEN_RE.sub(" ", text.lower())
    tokens = [t for t in stripped.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def clean_file(src: Path, dst: Path) -> None:
    """Clean one normalized text file and write the result to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    dst.write_text(clean_text(text), encoding="utf-8")


def clean_variant(normalized_dir: Path, cleaned_dir: Path, variant: str) -> int:
    """Clean all .txt files under normalized_dir/variant/ → cleaned_dir/variant/.
    Returns the number of files processed."""
    src_dir = normalized_dir / variant
    dst_dir = cleaned_dir / variant
    count = 0
    for src in sorted(src_dir.glob("*.txt")):
        stem = src.name[: -len(".txt")]
        dst = dst_dir / f"{stem}.cleaned.txt"
        clean_file(src, dst)
        count += 1
    return count


def clean_all(normalized_dir: Path, cleaned_dir: Path) -> None:
    """Clean every variant subfolder found under normalized_dir."""
    for variant_dir in sorted(normalized_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        n = clean_variant(normalized_dir, cleaned_dir, variant_dir.name)
        print(f"{variant_dir.name}: {n} files → {cleaned_dir / variant_dir.name}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[4]
    normalized_dir = root / "data" / "processed" / "normalized"
    cleaned_dir = root / "data" / "processed" / "cleaned"

    if len(sys.argv) > 1:
        variant = sys.argv[1]
        n = clean_variant(normalized_dir, cleaned_dir, variant)
        print(f"{variant}: {n} files → {cleaned_dir / variant}")
    else:
        clean_all(normalized_dir, cleaned_dir)
