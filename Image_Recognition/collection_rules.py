"""
collection_rules.py -- durable, post-tagging tag harmonization for known design
collections.

Some collections share a visual identity that the per-image tagger can't infer
consistently (e.g. every glyph of the American Flag Font is the same design;
every 12702 piece is free-standing lace). This module encodes those conventions
as prefix rules and re-applies them to the tags CSV after every Gemini Flash run
(wired into run_gemini_flash_tags.py), so manual harmonizations SURVIVE a
--force re-tag instead of being overwritten.

Run standalone to (re)apply to the canonical CSV at any time:
    venv/bin/python Image_Recognition/collection_rules.py
"""
from __future__ import annotations
import csv
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "image_recognition_results.csv"
TAG_COL = "gemini_flash_tags"

# Each rule: design stems beginning with `prefix` get `tags` applied via `mode`.
#   mode="set"          -> replace the tag list entirely with `tags`
#                          (use when every piece should be identical)
#   mode="ensure_front" -> guarantee `tags` are present, at the front, keeping
#                          the design's own remaining tags afterwards
RULES = [
    {
        "name": "American Flag Font (AFF-*)",
        "prefix": "AFF",
        "mode": "set",
        "tags": ["american flag", "patriotic", "stripes", "stars",
                 "red white and blue", "font", "alphabet", "monogram",
                 "letter", "number", "applique", "embroidery", "text"],
    },
    {
        "name": "12702 free-standing lace line",
        "prefix": "12702",
        "mode": "ensure_front",
        "tags": ["free standing lace", "lace", "applique"],
    },
    {
        "name": "E-ASD designer set",
        "prefix": "E-ASD",
        "mode": "ensure_front",
        "tags": ["embroidery"],
    },
    {
        "name": "Anchor Font (Anchor-*)",
        "prefix": "Anchor",
        "mode": "set",
        "tags": ["anchor", "nautical", "rope", "marine", "sea", "ocean",
                 "monogram", "letter", "alphabet", "font", "initial",
                 "embroidery", "text"],
    },
    {
        "name": "Embossed monogram alphabet (Embossed *)",
        "prefix": "Embossed",
        "mode": "set",
        "tags": ["monogram", "letter", "alphabet", "font", "initial", "text",
                 "embossed", "geometric", "pattern", "applique", "embroidery"],
    },
    {
        "name": "Letter font set (Letter *)",
        "prefix": "Letter",
        "mode": "set",
        "tags": ["font", "alphabet", "letter", "monogram", "initial", "text",
                 "embroidery"],
    },
    {
        "name": "Pearl shell monogram alphabet (Pearl-*)",
        "prefix": "Pearl",
        "mode": "set",
        "tags": ["monogram", "letter", "alphabet", "font", "initial", "clam shell",
                 "shell", "pearl", "nautical", "swirl", "scroll", "sea", "ocean",
                 "embroidery", "text"],
    },
    {
        # Varied themed set: all Christmas-tree quilting, but each tree differs.
        # Guarantee the shared theme tags; keep each design's own extras.
        "name": "E7 Christmas-tree quilting set (E7*)",
        "prefix": "E7",
        "mode": "ensure_front",
        "tags": ["quilting", "christmas tree", "christmas", "pine tree", "tree",
                 "trees", "holiday", "winter"],
    },
    {
        # Varied themed set: different motifs (stars / clam-shell / grid) within
        # one edge-to-edge quilting collection. Guarantee the shared tags only.
        "name": "OAB edge-to-edge quilting set (OAB*)",
        "prefix": "OAB",
        "mode": "ensure_front",
        "tags": ["quilting", "edge-to-edge"],
    },
]


def _split(s: str) -> list:
    return [t.strip() for t in (s or "").split(",") if t.strip()]


def _join(tags: list) -> str:
    return ", ".join(tags)


def _match(stem: str):
    for rule in RULES:
        if stem.startswith(rule["prefix"]):
            return rule
    return None


def apply_rules(stem: str, tags: list) -> list:
    """Return the harmonized tag list for one design stem."""
    rule = _match(stem)
    if not rule:
        return tags
    if rule["mode"] == "set":
        return list(rule["tags"])
    # ensure_front: required tags first (deduped, case-insensitive), then the rest
    req = rule["tags"]
    req_lower = {r.lower() for r in req}
    rest = [t for t in tags if t.lower() not in req_lower]
    return req + rest


def apply_to_csv(csv_path=CSV_PATH) -> int:
    """Apply all collection rules to the tags CSV in place. Returns rows changed."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"collection_rules: CSV not found: {csv_path}")
        return 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    fieldnames = list(rows[0].keys())
    if TAG_COL not in fieldnames:
        print(f"collection_rules: column {TAG_COL!r} not in CSV; nothing to do.")
        return 0

    changed = 0
    changed_by = {}
    matched_by = {}
    for r in rows:
        stem = (r.get("image_file") or "").rsplit(".", 1)[0]
        rule = _match(stem)
        if rule:
            matched_by[rule["name"]] = matched_by.get(rule["name"], 0) + 1
        before = r.get(TAG_COL) or ""
        after = _join(apply_rules(stem, _split(before)))
        if after != before:
            r[TAG_COL] = after
            changed += 1
            if rule:
                changed_by[rule["name"]] = changed_by.get(rule["name"], 0) + 1

    if changed:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"collection_rules: {csv_path.name} -> {changed} row(s) changed.")
    for rule in RULES:
        m = matched_by.get(rule["name"], 0)
        c = changed_by.get(rule["name"], 0)
        print(f"  - {rule['name']}: {m} matched, {c} updated")
    return changed


if __name__ == "__main__":
    apply_to_csv()
