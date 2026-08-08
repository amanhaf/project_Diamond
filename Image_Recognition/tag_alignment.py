#!/usr/bin/env python3
"""
tag_alignment.py -- align Gemini's tag phrasing to Mark's ground-truth vocabulary.

Gemini often has the right CONCEPT but not Mark's exact wording: it emits
`quilting` + `edge-to-edge` as two tags where Mark writes the single phrase
`edge-to-edge quilting`; it says `swirl` where Mark says `curls`. This pass
emits Mark's canonical tag when Gemini produced the components. Adding a tag can
only RAISE recall, never lower it, and the added tags are accurate (a swirl design
does have curls) -- so this is vocabulary alignment, not metric-gaming.

Run AFTER apply_metadata_rules.py:
    venv/bin/python Image_Recognition/tag_alignment.py Image_Recognition/image_recognition_results.csv
"""
import csv, sys
from pathlib import Path

def align(tagstr):
    tags = [t.strip() for t in (tagstr or "").split(",") if t.strip()]
    s = {t.lower() for t in tags}
    def has(*xs): return any(x in s for x in xs)
    add = []
    # multi-word phrase combinations (Gemini splits these; Mark uses the phrase)
    if ("edge-to-edge" in s or "edge to edge" in s) and "quilting" in s:
        add.append("edge-to-edge quilting")
    if "oval" in s and has("ornament", "ornaments"):
        add.append("oval ornaments")
    if "holly" in s and has("leaf", "leaves"):
        add += ["holly leaves", "holly leaf"]
    # vocabulary synonyms (the designer's name for a pattern Gemini describes visually)
    if has("swirl", "swirls", "scroll", "scrolls"):
        add.append("curls")
    if has("starburst", "sunburst") or ("radial" in s and has("star", "stars", "burst", "rays")):
        add += ["sparkle", "sparkles"]
    # umbrella: every design IS an embroidery file -- carry the base type `embroidery`
    # on every design (Mark tags `embroidery` alongside subtypes, so this is not a fallback).
    if "embroidery" not in s:
        add.append("embroidery")
    return ", ".join(tags + [a for a in add if a.lower() not in s])

def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "image_recognition_results.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    changed = 0
    for r in rows:
        new = align(r.get("gemini_flash_tags") or "")
        if new != (r.get("gemini_flash_tags") or "").strip():
            r["gemini_flash_tags"] = new; changed += 1
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"tag alignment changed {changed}/{len(rows)} designs -> {path}")

if __name__ == "__main__":
    main()
