"""
Build the multi-label tag vocabulary from the Gemini tags in
image_recognition_results.csv.

Keeps tags that appear in at least MIN_TAG_OCCURRENCES images, so we have
enough examples per tag to actually train.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition")
RESULTS_CSV = PROJECT_ROOT / "image_recognition_results.csv"
OUT_DIR = PROJECT_ROOT / "trained_model"
VOCAB_FILE = OUT_DIR / "tag_vocabulary.json"
TAG_COUNTS_FILE = OUT_DIR / "tag_counts.json"

# Tuning knobs
MIN_TAG_OCCURRENCES = 10       # tag must appear in at least this many images
MAX_VOCAB_SIZE = 150           # cap the vocab; keep the top-N most frequent
LOWERCASE = True               # treat "Butterfly" and "butterfly" as the same tag
STRIP_PLURAL_S = False         # set True to collapse "butterfly" / "butterflies"

# Tags to always exclude (junk / noise / too generic)
EXCLUDE_TAGS = {
    "",
    "image",
    "picture",
    "photo",
    "design",
    "pattern",
    "illustration",
    "graphic",
    # Too common — appears on nearly every image, not discriminative
    "embroidery",
    "machine embroidery",
    # Handled by Model A (craft classifier), avoid overlap
    "applique",
    "quilting",
    "pantograph",
}

def normalize_tag(tag: str) -> str:
    tag = tag.strip()
    if LOWERCASE:
        tag = tag.lower()
    if STRIP_PLURAL_S and tag.endswith("s") and len(tag) > 3:
        # Crude plural stripping. Good enough for our purposes.
        tag = tag[:-1]
    return tag


def parse_tags(tag_string: str) -> list[str]:
    if not tag_string:
        return []
    parts = [normalize_tag(t) for t in tag_string.split(",")]
    return [t for t in parts if t and t not in EXCLUDE_TAGS]


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {RESULTS_CSV}")

    print(f"[READ] {RESULTS_CSV}")
    counter: Counter[str] = Counter()
    row_count = 0
    tagged_row_count = 0

    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            tag_string = row.get("gemini_tags", "") or ""
            tags = parse_tags(tag_string)
            if tags:
                tagged_row_count += 1
            for t in set(tags):  # use set so a tag doesn't double-count within one image
                counter[t] += 1

    print(f"[STATS] Scanned {row_count} rows ({tagged_row_count} had tags)")
    print(f"[STATS] Unique raw tags: {len(counter)}")

    # Filter by frequency
    filtered = [(tag, count) for tag, count in counter.items() if count >= MIN_TAG_OCCURRENCES]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:MAX_VOCAB_SIZE]

    print(f"[STATS] Tags surviving filter (>= {MIN_TAG_OCCURRENCES}, top {MAX_VOCAB_SIZE}): {len(filtered)}")

    # Preview the head and tail of the surviving vocabulary
    print("\n[PREVIEW] Top 20 tags by frequency:")
    for tag, count in filtered[:20]:
        print(f"  {count:4d}  {tag}")

    print("\n[PREVIEW] Bottom 10 tags in vocabulary:")
    for tag, count in filtered[-10:]:
        print(f"  {count:4d}  {tag}")

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = [tag for tag, _ in filtered]
    with VOCAB_FILE.open("w") as f:
        json.dump(vocab, f, indent=2)
    print(f"\n[WROTE] {VOCAB_FILE}  ({len(vocab)} tags)")

    counts_dict = {tag: count for tag, count in filtered}
    with TAG_COUNTS_FILE.open("w") as f:
        json.dump(counts_dict, f, indent=2)
    print(f"[WROTE] {TAG_COUNTS_FILE}")

    print("\n[DONE] Tag vocabulary built. Next step: generate multi-label training data.")


if __name__ == "__main__":
    main()