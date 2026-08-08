#!/usr/bin/env python3
"""apply_metadata_rules.py -- deterministic tag corrections from Mark's 260629 notes,
a density-based applique rule (R5), and monogram-letter recovery from filenames.
Run AFTER tagging, BEFORE tag_alignment.py:
    venv/bin/python Image_Recognition/apply_metadata_rules.py Image_Recognition/image_recognition_results.csv
"""
import csv, re, sys
from pathlib import Path

QUILT_FAMILY = {"edge-to-edge", "edge to edge", "quilting", "cornerstone"}
LACE_FAMILY  = {"lace", "free standing lace"}
ITH_TAGS     = {"mug rug", "coaster", "in the hoop project"}
# R5 guard: low stitch density is also produced by these types, not just applique.
DENSITY_GUARD = {"edge-to-edge", "edge to edge", "edge-to-edge quilting", "quilting",
                 "cornerstone", "lace", "free standing lace", "border"}

# Monogram alphabet sets encode their letter in the filename. Gemini can't reliably read
# the ornate low-contrast script, so recover the letter deterministically from the name.
MONOGRAM_PATTERNS = [
    re.compile(r"^pearl-([a-z])$"),
    re.compile(r"^anchor-([a-z])$"),
    re.compile(r"^embossed ([a-z])\b"),
    re.compile(r"^letter ([a-z])$"),
]


def monogram_letter(image_file):
    stem = image_file.rsplit(".", 1)[0].strip().lower()
    for rx in MONOGRAM_PATTERNS:
        m = rx.match(stem)
        if m:
            return m.group(1).upper()
    return None


def apply_rules(tags, width_in, height_in, color_count, stitch_count=0):
    order = [t.strip().lower() for t in tags if t and t.strip()]
    tset = set(order)
    def num(x):
        try: return float(x)
        except Exception: return 0.0
    w, h, c = num(width_in), num(height_in), int(num(color_count))
    longest, shortest = max(w, h), min(w, h)
    notes = []
    if (QUILT_FAMILY & tset) and shortest and shortest < 3.0 and longest and longest / shortest > 2.5:
        tset -= {"edge-to-edge", "edge to edge"}
        if "border" not in tset:
            tset.add("border"); notes.append("R2 +border -edge-to-edge")
        else:
            notes.append("R2 -edge-to-edge")
    if c > 1 and (QUILT_FAMILY & tset):
        tset -= QUILT_FAMILY; notes.append(f"R1 -quilt/E2E ({c} colors)")
    if longest and longest < 3.0 and (ITH_TAGS & tset):
        tset -= ITH_TAGS; notes.append("R3 -ith (<3in)")
    if (QUILT_FAMILY & tset) and (LACE_FAMILY & tset):
        if c == 1: tset -= LACE_FAMILY; notes.append("R4 -lace")
        else: tset -= QUILT_FAMILY; notes.append("R4 -quilt")
    area = w * h; st = num(stitch_count)
    if area > 0 and st > 0 and (st / area) < 225 and "applique" not in tset and not (DENSITY_GUARD & tset):
        tset.add("applique"); notes.append(f"R5 +applique (density {st/area:.0f}<225, empty interior)")
    out = [t for t in order if t in tset] + [t for t in tset if t not in order]
    return out, notes


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "image_recognition_results.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    changed = 0
    for r in rows:
        tags = list((r.get("gemini_flash_tags") or "").split(","))
        new, notes = apply_rules(tags, r.get("emb_width_in"), r.get("emb_height_in"),
                                 r.get("color_count"), r.get("stitch_count"))
        # monogram letter from the filename (script too ornate for Gemini to read reliably)
        L = monogram_letter(r.get("image_file", ""))
        if L and L.lower() not in [t.strip().lower() for t in new]:
            new = new + [L]; notes = notes + [f"+letter {L}"]
        if notes:
            r["gemini_flash_tags"] = ", ".join(new); changed += 1
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"metadata guards changed {changed}/{len(rows)} designs")


main()
