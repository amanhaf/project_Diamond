from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


# -----------------------------
# EDIT THESE TWO PATHS
# -----------------------------
SAMPLES_DIR = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition/samples")
RENDERED_DIR = Path("/Users/amans/Desktop/Mark-VLM/Metadata_Capture/rendered_images")

# Where to write CSV outputs (in your project root)
OUT_DIR = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition")


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


@dataclass(frozen=True)
class ImageRow:
    dataset: str
    filename: str
    filepath: str


def iter_images(folder: Path, dataset_name: str) -> List[ImageRow]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: List[ImageRow] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
            rows.append(ImageRow(
                dataset=dataset_name,
                filename=p.name,
                filepath=str(p.resolve()),
            ))
    return rows


def write_labels_template(rows: List[ImageRow], out_path: Path) -> None:
    """
    Creates a CSV you can fill in:
      - tags: your ground truth, comma-separated lowercase (e.g., "rabbit, animal, wreath, embroidery")
      - notes: optional
    """
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "filename", "filepath", "tags", "notes"])
        for r in rows:
            w.writerow([r.dataset, r.filename, r.filepath, "", ""])


def write_compare_outputs(rows: List[ImageRow], out_path: Path) -> None:
    """
    Creates a CSV where your pipeline can store outputs from each provider.
    If it already exists, we do NOT overwrite it.
    """
    if out_path.exists():
        print(f"[OK] compare_outputs.csv already exists: {out_path}")
        return

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            "filename",
            "filepath",

            # Ground truth
            "gt_tags",

            # Provider outputs (comma-separated tags)
            "gemini_tags",
            "azure_tags",
            "aws_tags",
            "clarifai_tags",

            # Optional scoring fields (you can fill later)
            "craft_terms_found",
            "notes",
        ])
        for r in rows:
            w.writerow([r.dataset, r.filename, r.filepath, "", "", "", "", "", "", ""])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    rows += iter_images(SAMPLES_DIR, "samples")
    rows += iter_images(RENDERED_DIR, "rendered_images")

    print(f"Found {len(rows)} image(s) total:")
    print(f" - samples: {sum(1 for r in rows if r.dataset == 'samples')}")
    print(f" - rendered_images: {sum(1 for r in rows if r.dataset == 'rendered_images')}")

    labels_path = OUT_DIR / "labels_template.csv"
    compare_path = OUT_DIR / "compare_outputs.csv"

    write_labels_template(rows, labels_path)
    print(f"[WROTE] {labels_path}")

    write_compare_outputs(rows, compare_path)
    if compare_path.exists():
        print(f"[OK] {compare_path}")

    print("\nNext:")
    print("1) Open labels_template.csv and fill the 'tags' column for each image.")
    print("2) Copy those tags into compare_outputs.csv -> gt_tags (or I can give you a script to sync them).")


if __name__ == "__main__":
    main()