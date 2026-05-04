"""JSON → Letter records. Reads extracted.json from data/raw/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from jp_cluster.models.data import Letter


def iter_letters(json_path: Path) -> Iterator[Letter]:
    with json_path.open(encoding="utf-8") as f:
        records = json.load(f)
    for rec in records:
        text = rec.get("raw_text", "").strip()
        if not text:
            continue
        yield Letter(
            id=rec["document_id"],
            text_raw=text,
            tei_path=rec.get("xml_path"),
        )
