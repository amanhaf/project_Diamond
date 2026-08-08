"""
Convert the trained Keras models to ONNX for lightweight inference.

Run this in tf-venv (where TF is installed). Outputs:
  trained_model/craft_classifier.onnx
  trained_model/tag_generator.onnx

Then runtime inference can happen in any venv with onnxruntime installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort

# -----------------------------
# PATHS
# -----------------------------
PROJECT_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition")
OUT_DIR = PROJECT_ROOT / "trained_model"

CRAFT_KERAS = OUT_DIR / "craft_classifier.keras"
TAG_KERAS = OUT_DIR / "tag_generator.keras"

CRAFT_ONNX = OUT_DIR / "craft_classifier.onnx"
TAG_ONNX = OUT_DIR / "tag_generator.onnx"

IMG_SIZE = 224
OPSET = 15  # broadly compatible across onnxruntime versions


def keras_to_onnx(keras_path: Path, onnx_path: Path) -> None:
    if not keras_path.exists():
        raise FileNotFoundError(f"Missing Keras model: {keras_path}")

    print(f"[LOAD] {keras_path}")
    model = tf.keras.models.load_model(str(keras_path), compile=False)

    # Wrap in a concrete function with a fixed input signature.
    # tf2onnx can't always infer shapes from the Keras SavedModel format directly.
    input_signature = [
        tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name="image")
    ]

    print(f"[CONVERT] {keras_path.name} -> ONNX (opset {OPSET})")
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=OPSET,
        output_path=str(onnx_path),
    )
    print(f"[WROTE] {onnx_path}")


def verify_onnx_matches_keras(keras_path: Path, onnx_path: Path) -> None:
    """
    Sanity check: run the same random-noise input through both and make sure
    the predictions match to ~4 decimal places.
    """
    print(f"\n[VERIFY] {keras_path.name} vs {onnx_path.name}")

    keras_model = tf.keras.models.load_model(str(keras_path), compile=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Build a consistent test input (preprocessed ImageNet range)
    rng = np.random.default_rng(42)
    test_input = rng.standard_normal((1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)

    keras_out = keras_model.predict(test_input, verbose=0)
    if isinstance(keras_out, list):
        keras_out = keras_out[0]

    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: test_input})[0]

    diff = np.abs(keras_out - onnx_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print(f"  Keras output shape: {keras_out.shape}")
    print(f"  ONNX output shape:  {onnx_out.shape}")
    print(f"  Max absolute diff:  {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print(f"  [OK] Outputs match.")
    else:
        print(
            f"  [WARN] Outputs differ by more than 0.001 on the max dimension. "
            f"This usually still works in practice but flags a conversion issue."
        )


def main():
    keras_to_onnx(CRAFT_KERAS, CRAFT_ONNX)
    keras_to_onnx(TAG_KERAS, TAG_ONNX)

    verify_onnx_matches_keras(CRAFT_KERAS, CRAFT_ONNX)
    verify_onnx_matches_keras(TAG_KERAS, TAG_ONNX)

    print("\n[DONE] ONNX models ready:")
    print(f"  {CRAFT_ONNX}")
    print(f"  {TAG_ONNX}")
    print("\nNext step: install onnxruntime + Pillow in your main venv, then wire into run_compare_aws_azure_clarifai.py.")


if __name__ == "__main__":
    main()