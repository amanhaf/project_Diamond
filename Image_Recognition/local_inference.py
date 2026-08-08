"""
Lightweight inference using the ONNX-converted craft & tag models.

IMPORTANT: EfficientNet in Keras uses a Rescaling layer INSIDE the model.
Inputs must be in [0, 255] range (NOT normalized). Use OpenCV's resize
to match tf.image.resize behavior — Pillow's BILINEAR is close but not
identical, enough to flip predictions on dense line-pattern images.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent / "trained_model"
CRAFT_ONNX = MODEL_DIR / "craft_classifier.onnx"
TAG_ONNX = MODEL_DIR / "tag_generator.onnx"
CRAFT_LABELS_FILE = MODEL_DIR / "craft_labels.json"
TAG_VOCAB_FILE = MODEL_DIR / "tag_vocabulary.json"

IMG_SIZE = 224
TAG_CONF_THRESHOLD = 0.3
MAX_TAGS = 20

# Tags the model is unreliable at predicting (high false-positive rate against Gemini).
# Filtered out at inference time so they never appear in output.
# Source: empirical analysis on 347-image comparison CSV — these tags fired with
# confidence > 0.5 but Gemini disagreed in >40% of cases.
SUPPRESSED_TAGS = {
    "filigree",     # 65% wrong — over-fires on any fine linework
    "swirl",        # 47% wrong — too generic
    "intricate",    # 46% wrong — same pattern
    "initial",      # 43% wrong — fires on any letter-shape
    "marine",       # 43% wrong — over-fires on circular/scroll patterns
    "shape",        # 33% wrong — too generic
    "christmas",    # 28% wrong — training set was holiday-heavy
    "winter",       # 28% wrong — same as above
    "satin stitch", # 75% wrong — fires on any solid embroidery fill
}


class LocalInferencer:
    def __init__(self):
        for p in (CRAFT_ONNX, TAG_ONNX, CRAFT_LABELS_FILE, TAG_VOCAB_FILE):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing expected file: {p}\n"
                    f"Did you run convert_to_onnx.py?"
                )

        with CRAFT_LABELS_FILE.open("r") as f:
            self.craft_labels: list[str] = json.load(f)

        with TAG_VOCAB_FILE.open("r") as f:
            self.tag_vocab: list[str] = json.load(f)

        providers = ["CPUExecutionProvider"]
        self.craft_sess = ort.InferenceSession(str(CRAFT_ONNX), providers=providers)
        self.tag_sess = ort.InferenceSession(str(TAG_ONNX), providers=providers)

        self.craft_input_name = self.craft_sess.get_inputs()[0].name
        self.tag_input_name = self.tag_sess.get_inputs()[0].name

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Training-time preprocessing match:
          - Decode to RGB
          - Resize to 224x224 using OpenCV INTER_LINEAR (closer to tf.image.resize
            than Pillow BILINEAR, matters on dense pattern images)
          - Float32 in [0, 255] range (EfficientNet has internal Rescaling)
        """
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        arr = arr.astype(np.float32)
        return arr[np.newaxis, ...]

    def predict(self, image_bytes: bytes) -> dict:
        x = self._preprocess(image_bytes)

        craft_probs = self.craft_sess.run(None, {self.craft_input_name: x})[0][0]
        tag_probs = self.tag_sess.run(None, {self.tag_input_name: x})[0][0]

        craft_idx = int(np.argmax(craft_probs))
        craft_label = self.craft_labels[craft_idx]
        craft_conf = float(craft_probs[craft_idx])

        scored = [
            (self.tag_vocab[i], float(tag_probs[i]))
            for i in range(len(self.tag_vocab))
            if tag_probs[i] >= TAG_CONF_THRESHOLD
            and self.tag_vocab[i] not in SUPPRESSED_TAGS
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:MAX_TAGS]

        return {
            "craft": craft_label,
            "craft_confidence": craft_conf,
            "tags": scored,
        }


_inferencer: Optional[LocalInferencer] = None


def _get_inferencer() -> LocalInferencer:
    global _inferencer
    if _inferencer is None:
        _inferencer = LocalInferencer()
    return _inferencer


def local_predict(image_bytes: bytes) -> dict:
    return _get_inferencer().predict(image_bytes)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python local_inference.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    result = local_predict(img_path.read_bytes())

    print(f"\nImage: {img_path.name}")
    print(f"Craft:  {result['craft']} ({result['craft_confidence']:.2%})")
    print(f"Tags ({len(result['tags'])} above threshold {TAG_CONF_THRESHOLD}):")
    for tag, conf in result["tags"]:
        print(f"  {tag:25s} {conf:.2%}")