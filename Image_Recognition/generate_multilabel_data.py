"""
For every image in clarifai_train/, look up its gemini_tags from
image_recognition_results.csv and generate a multi-label training index.

Output: trained_model/multilabel_labels.json with schema:
{
  "vocab": ["text", "line art", "monogram", ...],
  "samples": [
    {
      "filepath": "/absolute/path/to/img.png",
      "filename": "img.png",
      "craft_class": "applique",
      "tag_vector": [0, 1, 0, 1, ...]  # length 109, one per vocab term
    },
    ...
  ]
}

Also reports how many images had no tag match (they'd still train for
craft classification but be useless for multi-label).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition")
TRAIN_ROOT = PROJECT_ROOT / "clarifai_train"
RESULTS_CSV = PROJECT_ROOT / "image_recognition_results.csv"
OUT_DIR = PROJECT_ROOT / "trained_model"
VOCAB_FILE = OUT_DIR / "tag_vocabulary.json"
LABELS_FILE = OUT_DIR / "multilabel_labels.json"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOWERCASE = True


def normalize_tag(tag: str) -> str:
    tag = tag.strip()
    if LOWERCASE:
        tag = tag.lower()
    return tag


def parse_tags(tag_string: str) -> set[str]:
    if not tag_string:
        return set()
    return {normalize_tag(t) for t in tag_string.split(",") if t.strip()}


def load_vocab() -> list[str]:
    if not VOCAB_FILE.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {VOCAB_FILE}. Run build_tag_vocabulary.py first.")
    with VOCAB_FILE.open("r") as f:
        return json.load(f)


def build_filename_to_tags_map() -> dict[str, set[str]]:
    """
    Read the Gemini CSV and build filename -> set of tags.
    """
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}")

    mapping: dict[str, set[str]] = {}
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("image_file") or "").strip()
            if not filename:
                continue
            tags = parse_tags(row.get("gemini_tags") or "")
            mapping[filename] = tags

    return mapping


def collect_training_images() -> list[tuple[Path, str]]:
    """
    Walk clarifai_train/{craft_class}/ and return (filepath, craft_class) tuples.
    """
    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_ROOT}")

    results: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in TRAIN_ROOT.iterdir() if p.is_dir()):
        craft_class = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                results.append((img_path, craft_class))
    return results


def main():
    vocab = load_vocab()
    vocab_index = {tag: i for i, tag in enumerate(vocab)}
    print(f"[LOAD] Vocabulary size: {len(vocab)}")

    filename_to_tags = build_filename_to_tags_map()
    print(f"[LOAD] Gemini CSV rows with filenames: {len(filename_to_tags)}")

    images = collect_training_images()
    print(f"[LOAD] Training images found: {len(images)}")

    samples: list[dict] = []
    matched = 0
    unmatched_filenames: list[str] = []
    zero_vector_count = 0
    craft_counts: dict[str, int] = {}

    for img_path, craft_class in images:
        filename = img_path.name
        tags = filename_to_tags.get(filename, set())

        if not tags:
            unmatched_filenames.append(filename)

        # Build the tag vector (1 if tag is in vocab and present on this image)
        vector = [0] * len(vocab)
        for tag in tags:
            if tag in vocab_index:
                vector[vocab_index[tag]] = 1

        if sum(vector) == 0:
            zero_vector_count += 1
        else:
            matched += 1

        samples.append({
            "filepath": str(img_path.resolve()),
            "filename": filename,
            "craft_class": craft_class,
            "tag_vector": vector,
        })

        craft_counts[craft_class] = craft_counts.get(craft_class, 0) + 1

    # Report
    print(f"\n[STATS] Samples with at least one vocab tag: {matched}/{len(images)}")
    print(f"[STATS] Samples with no vocab tags (tag_vector all zeros): {zero_vector_count}")

    print("\n[STATS] Craft class distribution:")
    for cls, count in sorted(craft_counts.items()):
        print(f"  {cls}: {count}")

    # Vocab coverage: how often each tag appears across the training set
    tag_coverage = [0] * len(vocab)
    for s in samples:
        for i, v in enumerate(s["tag_vector"]):
            tag_coverage[i] += v

    print("\n[STATS] Tag coverage (training images per vocab term):")
    print(f"  mean: {sum(tag_coverage) / len(vocab):.1f}")
    print(f"  min: {min(tag_coverage)}")
    print(f"  max: {max(tag_coverage)}")

    zero_coverage_tags = [vocab[i] for i, c in enumerate(tag_coverage) if c == 0]
    if zero_coverage_tags:
        print(f"[WARN] {len(zero_coverage_tags)} vocab tags have 0 occurrences in training data.")
        print(f"       (These tags were in the Gemini CSV but not on any clarifai_train image.)")
        if len(zero_coverage_tags) <= 20:
            print(f"       Examples: {zero_coverage_tags}")

    # Write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "vocab": vocab,
        "samples": samples,
    }
    with LABELS_FILE.open("w") as f:
        json.dump(output, f)
    print(f"\n[WROTE] {LABELS_FILE}")
    print(f"        ({len(samples)} samples, {len(vocab)}-dim tag vectors)")

    if zero_vector_count > 10:
        print(
            f"\n[CAUTION] {zero_vector_count} training images have no matching tags. "
            f"They will still contribute to craft classification but not to tag learning."
        )

    print("\n[DONE] Multi-label data generated. Next step: train both models.")


if __name__ == "__main__":
    main()