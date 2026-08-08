from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------
# PATHS — EDIT IF NEEDED
# --------------------------------------------------
LABELS_CSV = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition/labels_template.csv")
SPLIT_CSV = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition/train_val_split.csv")

TRAIN_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition/clarifai_train")
VAL_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition/clarifai_val")

# If True, existing folders are deleted and rebuilt
RESET_OUTPUT_FOLDERS = True

# If True, overwrite train_val_split.csv every run
REBUILD_SPLIT_CSV = True

# Split ratio
TRAIN_RATIO = 0.80

# Random seed for reproducibility
RANDOM_SEED = 42

# Only include rendered_images rows for Clarifai training
ONLY_RENDERED_IMAGES = True


# --------------------------------------------------
# CONTROLLED PRIMARY CLASSES
# Ordered by priority
# --------------------------------------------------
PRIMARY_CLASSES = [
    "applique",
    "quilting",
    "pantograph",
    "embroidery",
]

# Fallback class if nothing matches
DEFAULT_CLASS = "embroidery"


def normalize_tags(tag_string: str) -> List[str]:
    if not tag_string:
        return []
    return [t.strip().lower() for t in tag_string.split(",") if t.strip()]


def choose_primary_class(tags: List[str]) -> str:
    """
    Choose one dominant class from the tag list.
    Priority order is defined in PRIMARY_CLASSES.
    """
    tag_set = set(tags)

    if "applique" in tag_set or "appliqué" in tag_set:
        return "applique"

    if (
        "pantograph" in tag_set
        or "edge_to_edge" in tag_set
        or "edge to edge" in tag_set
        or "e2e" in tag_set
    ):
        return "pantograph"

    if "quilting" in tag_set or "longarm quilting" in tag_set or "longarm" in tag_set:
        return "quilting"

    if "embroidery" in tag_set or "machine embroidery" in tag_set:
        return "embroidery"

    return DEFAULT_CLASS


def load_labels_rows(labels_csv: Path) -> List[dict]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return rows


def build_split_csv_from_labels(labels_csv: Path, split_csv: Path) -> None:
    """
    Builds train_val_split.csv from labels_template.csv using filename.
    """
    rows = load_labels_rows(labels_csv)

    usable_rows = []
    seen = set()

    for row in rows:
        dataset = (row.get("dataset") or "").strip().lower()
        filename = (row.get("filename") or "").strip()
        filepath = (row.get("filepath") or "").strip()

        if not filename or not filepath:
            continue

        if ONLY_RENDERED_IMAGES and dataset != "rendered_images":
            continue

        # Avoid duplicate filenames in split file
        if filename in seen:
            continue
        seen.add(filename)
        usable_rows.append(row)

    if not usable_rows:
        raise ValueError("No usable rows found in labels CSV to create split file.")

    random.seed(RANDOM_SEED)
    filenames = [r["filename"].strip() for r in usable_rows]
    random.shuffle(filenames)

    split_idx = int(len(filenames) * TRAIN_RATIO)
    train_files = filenames[:split_idx]
    val_files = filenames[split_idx:]

    split_csv.parent.mkdir(parents=True, exist_ok=True)
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split"])
        for fn in train_files:
            writer.writerow([fn, "train"])
        for fn in val_files:
            writer.writerow([fn, "val"])

    print(f"[WROTE] {split_csv}")
    print(f"  Train rows: {len(train_files)}")
    print(f"  Val rows:   {len(val_files)}")


def load_split_map(split_csv: Path) -> Dict[str, str]:
    """
    Returns: filename -> split ('train' or 'val')
    """
    split_map: Dict[str, str] = {}

    with split_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        filename_col = "filename" if "filename" in cols else cols[0]
        split_col = "split" if "split" in cols else None

        if split_col is None:
            raise ValueError(
                f"Could not find a 'split' column in {split_csv}. Columns found: {cols}"
            )

        for row in reader:
            filename = (row.get(filename_col) or "").strip()
            split = (row.get(split_col) or "").strip().lower()

            if not filename:
                continue
            if split not in {"train", "val"}:
                continue

            split_map[filename] = split

    return split_map


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

    # Build or rebuild split file automatically
    if REBUILD_SPLIT_CSV or not SPLIT_CSV.exists():
        build_split_csv_from_labels(LABELS_CSV, SPLIT_CSV)

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"Split CSV not found: {SPLIT_CSV}")

    split_map = load_split_map(SPLIT_CSV)

    if RESET_OUTPUT_FOLDERS:
        reset_dir(TRAIN_ROOT)
        reset_dir(VAL_ROOT)
    else:
        TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        VAL_ROOT.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": {c: 0 for c in PRIMARY_CLASSES},
        "val": {c: 0 for c in PRIMARY_CLASSES},
        "skipped": 0,
    }

    rows = load_labels_rows(LABELS_CSV)

    for row in rows:
        dataset = (row.get("dataset") or "").strip().lower()
        filename = (row.get("filename") or "").strip()
        filepath = (row.get("filepath") or "").strip()
        tags_raw = (row.get("tags") or "").strip()

        if ONLY_RENDERED_IMAGES and dataset != "rendered_images":
            counts["skipped"] += 1
            continue

        if not filename or not filepath:
            counts["skipped"] += 1
            continue

        src = Path(filepath)
        if not src.exists():
            print(f"[SKIP] Missing file: {src}")
            counts["skipped"] += 1
            continue

        split = split_map.get(filename)
        if split not in {"train", "val"}:
            print(f"[SKIP] No split found for: {filename}")
            counts["skipped"] += 1
            continue

        tags = normalize_tags(tags_raw)
        primary_class = choose_primary_class(tags)

        if primary_class not in PRIMARY_CLASSES:
            primary_class = DEFAULT_CLASS

        if split == "train":
            dst = TRAIN_ROOT / primary_class / filename
            copy_image(src, dst)
            counts["train"][primary_class] += 1
        else:
            dst = VAL_ROOT / primary_class / filename
            copy_image(src, dst)
            counts["val"][primary_class] += 1

    print("\nDone building Clarifai training folders.\n")

    print("Train counts:")
    for cls, n in counts["train"].items():
        print(f"  {cls}: {n}")

    print("\nValidation counts:")
    for cls, n in counts["val"].items():
        print(f"  {cls}: {n}")

    print(f"\nSkipped: {counts['skipped']}")

    print("\nFolders created:")
    print(f"  {TRAIN_ROOT}")
    print(f"  {VAL_ROOT}")
    print(f"\nSplit file used:")
    print(f"  {SPLIT_CSV}")


if __name__ == "__main__":
    main()