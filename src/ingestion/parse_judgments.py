"""
HTML -> JSON conversion for the CJEU preliminary-rulings corpus
(Text Mining exam project - Env/Agri).

Each judgment HTML (EUR-Lex "cj-convex" format) is parsed into:
  {
    "text": "<full textual content of the judgment>",
    "metadata": { ... }
  }
"""
import json
import re
import unicodedata
from pathlib import Path
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Reference tables (used for lightweight, rule-based first-pass tagging;
# meant to be refined later, not treated as ground truth)
# ---------------------------------------------------------------------------

INSTRUMENT_PATTERNS = [
    # (regex, canonical label, kind)
    (r"Regulation\s*\(E[CU]\)\s*No\s*\d+/\d{4}", None, "Regulation"),
    (r"Regulation\s*\(EU\)\s*\d{4}/\d+", None, "Regulation"),
    (r"Directive\s*\d{4}/\d+(?:/E[CU])?", None, "Directive"),
    (r"Council\s+Directive\s*\d+/\d+/EEC", None, "Directive"),
]

THEMATIC_KEYWORDS = {
    # environmental
    "Nature & biodiversity": ["habitat", "natura 2000", "wild fauna", "wild flora", "birds directive", "conservation of natura"],
    "Climate, emissions trading & renewable energy": ["emission allowance", "emissions trading", "greenhouse gas", "renewable energy", "biofuel"],
    "EIA/SEA & access to environmental justice": ["environmental impact assessment", "strategic environmental assessment", "access to environmental information", "aarhus"],
    "Waste & circular economy": ["waste", "circular economy", "shipment of waste", "basel convention"],
    "Industrial emissions / air & water quality": ["industrial emissions", "air quality", "water framework", "urban waste water"],
    # agricultural
    "CAP direct payments & cross-compliance": ["direct payment", "cross-compliance", "single area payment", "conditionality"],
    "Rural development (EAFRD)": ["rural development", "eafrd", "young farmer", "start-up aid"],
    "Common organisation of markets": ["common organisation of the market", "wine", "spirit drink"],
    "Organic production": ["organic production", "organic farming", "regulation (ec) no 834/2007", "regulation (eu) 2018/848"],
    "Plant-protection products & food safety": ["plant-protection product", "pesticide", "maximum residue level", "food safety", "general food law"],
    "Quality schemes (PDO/PGI)": ["protected designation of origin", "protected geographical indication", "quality scheme"],
}


def normalize_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def get_case_numbers_from_filename(path: Path):
    """C-116.20.html -> ['C-116/20']; handles joined cases and stray lowercase 'c-'."""
    stem = path.stem
    parts = re.split(r"\s+and\s+|,\s*", stem)
    numbers = []
    for p in parts:
        p = p.strip()
        m = re.search(r"[Cc]?-?\s*(\d+)\.(\d{2})", p)
        if m:
            numbers.append(f"C-{m.group(1)}/{m.group(2)}")
    return numbers or [stem]


def extract_paragraphs(soup):
    """Numbered body paragraphs. Three known EUR-Lex/Curia HTML templates
    are used across the corpus: 'coj-normal' and its unprefixed sibling
    'normal' (table-based numbering), and the newer 'C01PointnumeroteAltN'
    (number inlined in the paragraph text)."""
    for cls in ("coj-normal", "normal"):
        found = soup.find_all("p", class_=cls)
        if found:
            return [normalize_ws(p.get_text(" ")) for p in found if normalize_ws(p.get_text(" "))]
    curia = soup.find_all("p", class_="C01PointnumeroteAltN")
    paras = []
    for p in curia:
        txt = normalize_ws(p.get_text(" "))
        txt = re.sub(r"^\d+\s+", "", txt)  # strip leading inline paragraph number
        if txt:
            paras.append(txt)
    return paras


def extract_header_lines(soup):
    for cls in ("coj-sum-title-1", "sum-title-1"):
        lines = [normalize_ws(p.get_text(" ")) for p in soup.find_all("p", class_=cls)]
        if lines:
            return lines
    # Curia template: title/date lines use 'C19Centre', but so do other
    # short centred lines (e.g. 'v', 'THE COURT (...)') so filter by content.
    out = []
    for p in soup.find_all("p", class_="C19Centre"):
        txt = normalize_ws(p.get_text(" "))
        if re.search(r"JUDGMENT OF THE COURT|ORDER OF THE COURT|\d{4}\s*\(", txt):
            out.append(txt)
    return out


def extract_chamber(header_lines):
    for line in header_lines:
        m = re.search(r"JUDGMENT OF THE COURT\s*(?:\(([^)]+)\))?", line, re.I)
        if m:
            return m.group(1).strip() if m.group(1) else "Full Court"
    return None


def extract_date(header_lines):
    for line in header_lines:
        m = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", line)
        if m:
            raw = m.group(1)
            months = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05",
                      "June": "06", "July": "07", "August": "08", "September": "09",
                      "October": "10", "November": "11", "December": "12"}
            dm = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", raw)
            if dm and dm.group(2) in months:
                return f"{dm.group(3)}-{months[dm.group(2)]}-{int(dm.group(1)):02d}"
            return raw
    return None


def extract_keywords_line(soup):
    for cls in ("coj-index", "index", "C71Indicateur"):
        idx = soup.find("p", class_=cls)
        if idx:
            return normalize_ws(idx.get_text(" | "))
    return None


def extract_case_number_from_text(full_text, fallback):
    m = re.search(r"In Case[s]?\s+(C[\u2011\-‑]\d+/\d+(?:\s+and\s+C[\u2011\-‑]\d+/\d+)*)", full_text)
    if m:
        return re.sub(r"[\u2011‑]", "-", m.group(1)).replace(" ", " ")
    return ", ".join(fallback)


def extract_referring_court(full_text):
    # Group 2 is greedy so it correctly spans nested parentheses, e.g.
    # "(Court of First Instance (Dutch-speaking), Brussels, Belgium)".
    m = re.search(
        r"REQUEST[S]? for a preliminary ruling[s]? under Article\s*267\s*TFEU from (?:the\s+)?(.+?)\s*\((.+)\)\s*,?\s*made by (?:decision|order|judgment)s? of ([^,]+),",
        full_text,
    )
    if not m:
        return {"referring_court": None, "member_state": None, "decision_date": None}
    court, paren, decision_date = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    member_state = paren.split(",")[-1].strip()
    return {"referring_court": court, "member_state": member_state, "decision_date": decision_date}


def extract_language(full_text):
    m = re.search(r"Language of the case\s*:\s*([A-Za-z]+)", full_text)
    return m.group(1) if m else None


def extract_instruments(full_text):
    found = set()
    for pattern, _, kind in INSTRUMENT_PATTERNS:
        for m in re.finditer(pattern, full_text):
            found.add(normalize_ws(m.group(0)))
    return sorted(found)


def extract_operative_part(soup):
    """Text of the numbered rulings after 'hereby rules:', one entry per ruling.
    Handles both the 'coj-normal' (single paragraph per ruling) and the
    'Dispositif' (ruling split across a numbered lead + continuation lines)
    templates by merging chunks until the next numbered ruling starts."""
    marker = None
    for p in soup.find_all("p"):
        if "hereby rules" in p.get_text():
            marker = p
            break
    if not marker:
        return None
    chunks = []
    for p in marker.find_all_next("p"):
        txt = normalize_ws(p.get_text(" "))
        if not txt or txt == "[Signatures]" or "Language of the case" in txt:
            break
        chunks.append(txt)
    if not chunks:
        return None
    rulings, current = [], ""
    for chunk in chunks:
        if re.match(r"^\d+\.", chunk):
            if current:
                rulings.append(current.strip())
            current = chunk
        else:
            current += " " + chunk
    if current:
        rulings.append(current.strip())
    return rulings or chunks


def guess_thematic_areas(full_text_lower):
    hits = []
    for area, kws in THEMATIC_KEYWORDS.items():
        if any(kw in full_text_lower for kw in kws):
            hits.append(area)
    return hits


def parse_judgment(path: Path, domain_hint: str):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    celex = normalize_ws(soup.title.get_text()) if soup.title else None
    celex = celex or None
    header_lines = extract_header_lines(soup)
    keywords_line = extract_keywords_line(soup)
    paragraphs = extract_paragraphs(soup)
    full_text = normalize_ws(soup.get_text("\n"))

    case_numbers = get_case_numbers_from_filename(path)
    case_number_str = extract_case_number_from_text(full_text, case_numbers)
    court_info = extract_referring_court(full_text)
    operative_part = extract_operative_part(soup)

    text_lower = full_text.lower()

    metadata = {
        "case_number": case_number_str,
        "case_numbers": case_numbers,
        "celex_number": celex,
        "date_of_judgment": extract_date(header_lines),
        "chamber": extract_chamber(header_lines),
        "referring_court": court_info["referring_court"],
        "member_state": court_info["member_state"],
        "referring_decision_date": court_info["decision_date"],
        "language_of_the_case": extract_language(full_text),
        "legal_domain": domain_hint,  # patched to "both" later for cross-domain cases
        "thematic_area": guess_thematic_areas(text_lower) or None,
        "eu_instruments": extract_instruments(full_text) or None,
        "keywords": keywords_line,
        "source_file": path.name,
    }

    document = {
        "text": full_text,
        "paragraphs": paragraphs,
        "operative_part": operative_part,
        "metadata": metadata,
    }
    return document


def process_corpus(src_dir: Path, out_dir: Path, domain_hint: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    errors = []
    for f in sorted(src_dir.glob("*.html")):
        try:
            doc = parse_judgment(f, domain_hint)
            out_name = f.stem + ".json"
            (out_dir / out_name).write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            results[f.name] = doc["metadata"]
        except Exception as e:
            errors.append((f.name, str(e)))
    return results, errors


def patch_cross_domain(agri_dir: Path, env_dir: Path):
    """Detect judgments present in both corpora (by celex_number) and mark
    legal_domain='both' in each copy, keeping both sub-corpora copies
    (see Sec.2 of the exam guide: either choice is acceptable if consistent;
    we keep-both + tag, so either routing path can reach them)."""
    def load_index(d):
        idx = {}
        for f in d.glob("*.json"):
            doc = json.loads(f.read_text(encoding="utf-8"))
            key = doc["metadata"].get("case_number")
            idx[key] = f
        return idx

    agri_idx = load_index(agri_dir)
    env_idx = load_index(env_dir)
    shared = set(agri_idx) & set(env_idx)
    shared.discard(None)

    for key in shared:
        for f in (agri_idx[key], env_idx[key]):
            doc = json.loads(f.read_text(encoding="utf-8"))
            doc["metadata"]["legal_domain"] = "both"
            f.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return shared


if __name__ == "__main__":
    base = Path("/home/claude/work")
    agri_meta, agri_err = process_corpus(base / "agri", base / "json" / "agricultural", "agricultural")
    env_meta, env_err = process_corpus(base / "env", base / "json" / "environmental", "environmental")

    print(f"Agricultural: {len(agri_meta)} parsed, {len(agri_err)} errors")
    print(f"Environmental: {len(env_meta)} parsed, {len(env_err)} errors")
    if agri_err:
        print("Agri errors:", agri_err[:5])
    if env_err:
        print("Env errors:", env_err[:5])

    shared = patch_cross_domain(base / "json" / "agricultural", base / "json" / "environmental")
    print(f"Cross-domain judgments (present in both corpora): {len(shared)}")
    print(sorted(shared))

    # Build a flat manifest (one row per judgment) for quick browsing / for
    # feeding the embedding & routing steps later in the pipeline.
    manifest = []
    for sub in ("agricultural", "environmental"):
        for f in sorted((base / "json" / sub).glob("*.json")):
            doc = json.loads(f.read_text(encoding="utf-8"))
            manifest.append({"json_file": f"{sub}/{f.name}", **doc["metadata"]})
    (base / "json" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest written with {len(manifest)} entries.")
