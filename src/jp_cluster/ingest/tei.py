"""TEI → Letter records.

Stage 0–2h. Schema-Details der Jean-Paul-Edition müssen vor Ort geprüft werden:
  - Wo liegt der Brieftext? Vermutlich //tei:body//tei:div[@type='letter'] oder //tei:body
  - xml:id auf <TEI> oder <text>?
  - Datum: <correspDesc>/<correspAction type='sent'>/<date> ?
  - Korrespondent: <persName ref="#..."> in <correspAction>?
  - Register-Tags: <rs ref="..."> oder <name ref="..."> mit GND-URI

Sobald geklärt, die XPaths unten anpassen — der Rest der Pipeline ist schemafrei.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Iterator

from lxml import etree

from jp_cluster.models import Letter

NS = {"tei": "http://www.tei-c.org/ns/1.0", "xml": "http://www.w3.org/XML/1998/namespace"}


def _text(el: etree._Element | None) -> str | None:
    if el is None:
        return None
    s = " ".join(el.itertext())
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _parse_date(raw: str | None) -> date | None:
    """JP-Briefe haben oft unscharfe Datierungen ('um 1798', '1798/99').
    Konservativ: nur ISO-präfixierte Daten parsen, Rest -> None und date_raw behalten.
    """
    if not raw:
        return None
    m = re.match(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", raw)
    if not m:
        return None
    y = int(m.group(1))
    mo = int(m.group(2) or 1)
    d = int(m.group(3) or 1)
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def parse_letter(xml_path: Path) -> Letter | None:
    """Parse one TEI file. Returns None on schema mismatch (caller logs)."""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    # ---- TODO: an JP-Schema anpassen ----
    letter_id = root.get(f"{{{NS['xml']}}}id") or xml_path.stem

    sender_el = root.find(".//tei:correspAction[@type='sent']/tei:persName", NS)
    addressee_el = root.find(".//tei:correspAction[@type='received']/tei:persName", NS)
    date_el = root.find(".//tei:correspAction[@type='sent']/tei:date", NS)
    place_el = root.find(".//tei:correspAction[@type='sent']/tei:placeName", NS)
    body_el = root.find(".//tei:body", NS)

    date_raw = (date_el.get("when") if date_el is not None else None) or _text(date_el)

    # Register-Tags einsammeln (GND etc.)
    register: list[str] = []
    for el in root.iterfind(".//tei:rs[@ref]", NS):
        register.append(el.get("ref", ""))
    for el in root.iterfind(".//tei:persName[@ref]", NS):
        ref = el.get("ref", "")
        if ref:
            register.append(ref)

    text_raw = _text(body_el)
    if not text_raw:
        return None

    return Letter(
        id=letter_id,
        sender=_text(sender_el),
        addressee=_text(addressee_el),
        date_iso=_parse_date(date_raw),
        date_raw=date_raw,
        place=_text(place_el),
        text_raw=text_raw,
        register_terms=sorted(set(register)),
        tei_path=str(xml_path),
    )


def iter_letters(tei_dir: Path) -> Iterator[Letter]:
    for xml_path in sorted(tei_dir.rglob("*.xml")):
        try:
            letter = parse_letter(xml_path)
        except etree.XMLSyntaxError as e:
            print(f"[skip] {xml_path}: {e}")
            continue
        if letter is not None:
            yield letter
