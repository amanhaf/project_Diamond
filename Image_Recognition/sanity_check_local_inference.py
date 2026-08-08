"""
Run local_predict on a sample of images from each craft folder
and report confusion. Helps decide if the model is actually usable.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from local_inference import local_predict, TAG_CONF_THRESHOLD

TRAIN_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition/clarifai_train")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SAMPLES_PER_CLASS = 10
RANDOM_SEED = 42


def main():
    rng = random.Random(RANDOM_SEED)

    # Gather
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in TRAIN_ROOT.iterdir() if p.is_dir()):
        imgs = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        ]
        rng.shuffle(imgs)
        by_class[class_dir.name] = imgs[:SAMPLES_PER_CLASS]

    print(f"Sampling up to {SAMPLES_PER_CLASS} images per class from {TRAIN_ROOT}\n")

    confusion = defaultdict(Counter)  # confusion[true_class][predicted_class]
    total_correct = 0
    total_n = 0
    tag_examples = []

    for true_class, images in by_class.items():
        if not images:
            print(f"[SKIP] {true_class}: no images")
            continue

        print(f"=== {true_class} ({len(images)} images) ===")
        n_correct = 0
        for p in images:
            result = local_predict(p.read_bytes())
            predicted = result["craft"]
            conf = result["craft_confidence"]

            mark = "OK" if predicted == true_class else "WRONG"
            print(f"  [{mark}] {p.name}: predicted={predicted} ({conf:.0%})")

            confusion[true_class][predicted] += 1
            if predicted == true_class:
                n_correct += 1
                total_correct += 1
            total_n += 1

            # Save a few tag examples for later display
            if len(tag_examples) < 4:
                tag_examples.append((p.name, true_class, result))

        acc = n_correct / len(images) if images else 0.0
        print(f"  Per-class accuracy: {acc:.0%} ({n_correct}/{len(images)})\n")

    # Summary confusion matrix
    print("=== Confusion matrix (rows = true class, cols = predicted) ===")
    classes = sorted(by_class.keys())
    col_width = max(max(len(c) for c in classes), 10) + 2
    header = " " * 12 + "".join(c.rjust(col_width) for c in classes)
    print(header)
    for true_cls in classes:
        row = true_cls.ljust(12)
        for pred_cls in classes:
            count = confusion[true_cls][pred_cls]
            row += str(count).rjust(col_width)
        print(row)

    overall = total_correct / total_n if total_n else 0.0
    print(f"\nOverall sample accuracy: {overall:.0%} ({total_correct}/{total_n})")

    # Show tag output on a few examples so we can see if tags are sensible
    print("\n=== Tag predictions on sample images ===")
    for fname, true_cls, result in tag_examples:
        print(f"\n{fname} (true={true_cls}, pred={result['craft']} @ {result['craft_confidence']:.0%})")
        for tag, conf in result["tags"][:8]:
            print(f"  {tag:25s} {conf:.0%}")


if __name__ == "__main__":
    main()