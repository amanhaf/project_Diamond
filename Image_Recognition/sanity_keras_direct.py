"""
Diagnostic: run inference through Keras directly, using the SAME
preprocessing the training script used. Bypass ONNX entirely.

Compares against the "collapse" we're seeing in local_inference.py.

If this also shows total collapse -> the model itself is bad.
If this shows varied predictions -> local_inference.py's preprocessing is wrong.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from io import BytesIO

PROJECT_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition")
TRAIN_ROOT = PROJECT_ROOT / "clarifai_train"
OUT_DIR = PROJECT_ROOT / "trained_model"
CRAFT_MODEL_FILE = OUT_DIR / "craft_classifier.keras"
TAG_MODEL_FILE = OUT_DIR / "tag_generator.keras"
CRAFT_LABELS_FILE = OUT_DIR / "craft_labels.json"
TAG_VOCAB_FILE = OUT_DIR / "tag_vocabulary.json"

IMG_SIZE = 224
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SAMPLES_PER_CLASS = 10
RANDOM_SEED = 42
TAG_THRESHOLD = 0.3


def preprocess_like_training(img_bytes: bytes) -> np.ndarray:
    """
    Match exactly what decode_and_augment does in the training script
    with augment=False:
      - Decode via tf.io
      - Resize to 224x224 via tf.image.resize
      - Cast to float32
      - Apply Keras EfficientNet preprocess_input
    """
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img.numpy()[np.newaxis, ...]


def main():
    with CRAFT_LABELS_FILE.open("r") as f:
        craft_labels = json.load(f)
    with TAG_VOCAB_FILE.open("r") as f:
        tag_vocab = json.load(f)

    print("[LOAD] Keras models...")
    craft_model = tf.keras.models.load_model(str(CRAFT_MODEL_FILE), compile=False)
    tag_model = tf.keras.models.load_model(str(TAG_MODEL_FILE), compile=False)

    rng = random.Random(RANDOM_SEED)
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in TRAIN_ROOT.iterdir() if p.is_dir()):
        imgs = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
        rng.shuffle(imgs)
        by_class[class_dir.name] = imgs[:SAMPLES_PER_CLASS]

    print(f"[SAMPLE] {SAMPLES_PER_CLASS} images per class from {TRAIN_ROOT}\n")

    confusion = defaultdict(Counter)
    total_correct = 0
    total_n = 0
    tag_examples = []

    for true_class, images in by_class.items():
        print(f"=== {true_class} ===")
        n_correct = 0
        for p in images:
            x = preprocess_like_training(p.read_bytes())
            craft_probs = craft_model.predict(x, verbose=0)[0]
            tag_probs = tag_model.predict(x, verbose=0)[0]

            idx = int(np.argmax(craft_probs))
            predicted = craft_labels[idx]
            conf = float(craft_probs[idx])

            mark = "OK" if predicted == true_class else "WRONG"
            # Also print the full prob distribution so we can see if it's truly collapsed
            probs_str = "  ".join(f"{craft_labels[i]}:{craft_probs[i]:.2f}" for i in range(len(craft_labels)))
            print(f"  [{mark}] {p.name}: pred={predicted} conf={conf:.2f}  [{probs_str}]")

            confusion[true_class][predicted] += 1
            if predicted == true_class:
                n_correct += 1
                total_correct += 1
            total_n += 1

            if len(tag_examples) < 4:
                top_tags = sorted(
                    [(tag_vocab[i], float(tag_probs[i])) for i in range(len(tag_vocab))],
                    key=lambda x: x[1],
                    reverse=True,
                )[:8]
                tag_examples.append((p.name, true_class, predicted, conf, top_tags))

        acc = n_correct / len(images) if images else 0.0
        print(f"  Per-class accuracy: {acc:.0%} ({n_correct}/{len(images)})\n")

    print("=== Confusion matrix (rows = true, cols = predicted) ===")
    classes = sorted(by_class.keys())
    col_w = max(max(len(c) for c in classes), 10) + 2
    header = " " * 12 + "".join(c.rjust(col_w) for c in classes)
    print(header)
    for tc in classes:
        row = tc.ljust(12)
        for pc in classes:
            row += str(confusion[tc][pc]).rjust(col_w)
        print(row)

    overall = total_correct / total_n if total_n else 0.0
    print(f"\nOverall: {overall:.0%} ({total_correct}/{total_n})")

    print("\n=== Tag predictions ===")
    for fname, true_cls, pred_cls, conf, tags in tag_examples:
        print(f"\n{fname} (true={true_cls}, pred={pred_cls} @ {conf:.0%})")
        for tag, c in tags:
            print(f"  {tag:25s} {c:.0%}")


if __name__ == "__main__":
    main()