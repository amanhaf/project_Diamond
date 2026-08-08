"""
Train an EfficientNetB0-backed dual-head classifier.

After a previous run, class-balanced oversampling collapsed the craft head
onto one class. This version uses per-sample weights on the craft loss
instead, which amplifies minority-class gradients without duplicating images.

Keeps:
  - Mixup augmentation
  - Label smoothing on craft head
  - Cosine LR schedule
  - Gradient clipping
  - Best-checkpoint-by-val-loss
  - Strong augmentation
  - EfficientNetB0 backbone
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition")
OUT_DIR = PROJECT_ROOT / "trained_model"
LABELS_FILE = OUT_DIR / "multilabel_labels.json"

CRAFT_MODEL_FILE = OUT_DIR / "craft_classifier.keras"
TAG_MODEL_FILE = OUT_DIR / "tag_generator.keras"
CRAFT_LABELS_FILE = OUT_DIR / "craft_labels.json"
METRICS_FILE = OUT_DIR / "training_metrics.json"
BEST_HEAD_CKPT = OUT_DIR / "best_head_phase.weights.h5"
BEST_FT_CKPT = OUT_DIR / "best_finetune_phase.weights.h5"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_HEAD = 20
EPOCHS_FINETUNE = 4
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-6
UNFREEZE_FROM_LAYER = 200
DROPOUT_RATE = 0.5
VAL_SPLIT = 0.15
RANDOM_SEED = 42
GRADIENT_CLIP_NORM = 1.0
LABEL_SMOOTHING_CRAFT = 0.1

LOSS_WEIGHT_CRAFT = 1.0
LOSS_WEIGHT_TAGS = 0.5

# Sample weighting: amplify minority class gradients without oversampling.
# Cap prevents extreme weights on very small classes from destabilizing training.
USE_SAMPLE_WEIGHTING = True
MAX_SAMPLE_WEIGHT = 3.0

USE_MIXUP = True
MIXUP_ALPHA = 0.2

REPORT_THRESHOLDS = (0.3, 0.5)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset():
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

    with LABELS_FILE.open("r") as f:
        data = json.load(f)

    vocab = data["vocab"]
    samples = data["samples"]
    craft_classes = sorted({s["craft_class"] for s in samples})
    craft_index = {c: i for i, c in enumerate(craft_classes)}

    print(f"[DATA] Samples: {len(samples)}")
    print(f"[DATA] Craft classes: {craft_classes}")
    print(f"[DATA] Vocab size: {len(vocab)}")
    return samples, vocab, craft_classes, craft_index


def split_train_val(samples, val_split):
    rng = random.Random(RANDOM_SEED)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    print(f"[SPLIT] Train: {len(train)}, Val: {len(val)}")
    return train, val


def compute_per_sample_craft_weights(train_samples, craft_classes, cap):
    """
    Return a list of per-sample scalar weights aligned with train_samples.
    Uses inverse-frequency weighting (capped at `cap`) per craft class.
    """
    counts = {c: 0 for c in craft_classes}
    for s in train_samples:
        counts[s["craft_class"]] += 1

    total = sum(counts.values())
    n_cls = len(craft_classes)
    raw_weight = {c: total / (n_cls * max(counts[c], 1)) for c in craft_classes}
    # Cap so tiny classes don't blow up gradients
    weight = {c: min(raw_weight[c], cap) for c in craft_classes}

    print("[WEIGHTS] Per-sample craft weights:")
    for c in craft_classes:
        print(f"  {c}: n={counts[c]}  raw={raw_weight[c]:.3f}  used={weight[c]:.3f}")

    return [float(weight[s["craft_class"]]) for s in train_samples]


def decode_and_augment(filepath_tensor, augment: bool):
    img_bytes = tf.io.read_file(filepath_tensor)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)

        scale = tf.random.uniform([], 0.8, 1.0)
        new_size = tf.cast(tf.cast(IMG_SIZE, tf.float32) * scale, tf.int32)
        img = tf.image.random_crop(img, [new_size, new_size, 3])
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k=k)

    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


def smooth_craft_labels(craft_id_int, n_classes, smoothing):
    one_hot = tf.one_hot(craft_id_int, depth=n_classes, dtype=tf.float32)
    return one_hot * (1.0 - smoothing) + smoothing / float(n_classes)


def build_base_dataset(samples, craft_index, sample_weights, augment, n_craft_classes):
    """
    Build a tf.data pipeline yielding (image, (craft_soft, tag_vector), (craft_weight, tag_weight))
    so Keras can apply per-sample weights to the craft loss.
    """
    filepaths = [s["filepath"] for s in samples]
    craft_ids = [craft_index[s["craft_class"]] for s in samples]
    tag_vectors = [s["tag_vector"] for s in samples]

    if sample_weights is None:
        sample_weights = [1.0] * len(samples)

    fp_ds = tf.data.Dataset.from_tensor_slices(filepaths)
    craft_ds = tf.data.Dataset.from_tensor_slices(tf.constant(craft_ids, dtype=tf.int32))
    tag_ds = tf.data.Dataset.from_tensor_slices(tf.constant(tag_vectors, dtype=tf.float32))
    craft_w_ds = tf.data.Dataset.from_tensor_slices(tf.constant(sample_weights, dtype=tf.float32))
    tag_w_ds = tf.data.Dataset.from_tensor_slices(tf.ones([len(samples)], dtype=tf.float32))

    img_ds = fp_ds.map(
        lambda fp: decode_and_augment(fp, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    smoothing = LABEL_SMOOTHING_CRAFT if augment else 0.0
    craft_smooth_ds = craft_ds.map(
        lambda c: smooth_craft_labels(c, n_craft_classes, smoothing),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # ((image), (craft_label, tag_label), (craft_weight, tag_weight))
    ds = tf.data.Dataset.zip((
        img_ds,
        (craft_smooth_ds, tag_ds),
        (craft_w_ds, tag_w_ds),
    ))

    if augment:
        ds = ds.shuffle(buffer_size=min(1024, len(samples)), seed=RANDOM_SEED)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


def mixup_batch(images, labels_tuple, weights_tuple, alpha):
    """
    Mix a batch with itself (shuffled). Works on soft (one-hot) labels.
    Also blends sample weights so weighted samples in one slot still
    contribute correctly when paired with lighter samples.
    """
    craft_labels, tag_labels = labels_tuple
    craft_w, tag_w = weights_tuple
    batch_size = tf.shape(images)[0]

    ga = tf.random.gamma(shape=[1], alpha=alpha)
    gb = tf.random.gamma(shape=[1], alpha=alpha)
    lam = ga / (ga + gb)
    lam = tf.squeeze(lam)
    lam = tf.maximum(lam, 1.0 - lam)

    perm = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, perm)
    craft_shuffled = tf.gather(craft_labels, perm)
    tag_shuffled = tf.gather(tag_labels, perm)
    craft_w_shuffled = tf.gather(craft_w, perm)
    tag_w_shuffled = tf.gather(tag_w, perm)

    mixed_images = lam * images + (1.0 - lam) * images_shuffled
    mixed_craft = lam * craft_labels + (1.0 - lam) * craft_shuffled
    mixed_tags = lam * tag_labels + (1.0 - lam) * tag_shuffled
    mixed_craft_w = lam * craft_w + (1.0 - lam) * craft_w_shuffled
    mixed_tag_w = lam * tag_w + (1.0 - lam) * tag_w_shuffled

    return mixed_images, (mixed_craft, mixed_tags), (mixed_craft_w, mixed_tag_w)


def apply_mixup(ds, alpha):
    return ds.map(
        lambda imgs, labels, weights: mixup_batch(imgs, labels, weights, alpha),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def build_model(n_craft_classes, n_tags):
    base = EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    craft_out = layers.Dense(n_craft_classes, activation="softmax", name="craft")(x)
    tag_out = layers.Dense(n_tags, activation="sigmoid", name="tags")(x)

    return tf.keras.Model(inputs, [craft_out, tag_out]), base


def compile_model(model, lr, total_epochs, steps_per_epoch):
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=max(1, total_epochs * steps_per_epoch),
        alpha=0.01,
    )
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=schedule,
            clipnorm=GRADIENT_CLIP_NORM,
        ),
        loss={
            "craft": tf.keras.losses.CategoricalCrossentropy(),
            "tags": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={
            "craft": LOSS_WEIGHT_CRAFT,
            "tags": LOSS_WEIGHT_TAGS,
        },
        metrics={
            "craft": [tf.keras.metrics.CategoricalAccuracy(name="acc")],
            "tags": [
                tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
                tf.keras.metrics.Precision(name="prec", thresholds=0.5),
                tf.keras.metrics.Recall(name="rec", thresholds=0.5),
            ],
        },
    )


def fit(model, train_ds, val_ds, epochs, phase_name, ckpt_path):
    print(f"\n[FIT] Phase: {phase_name} | epochs={epochs}")
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=[ckpt_cb],
    )


def evaluate_tag_head(model, val_samples, vocab):
    filepaths = [s["filepath"] for s in val_samples]
    tag_vectors = np.array([s["tag_vector"] for s in val_samples], dtype=np.float32)

    fp_ds = tf.data.Dataset.from_tensor_slices(filepaths).map(
        lambda fp: decode_and_augment(fp, augment=False),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(BATCH_SIZE)

    preds = model.predict(fp_ds, verbose=0)
    tag_preds = preds[1]

    results_by_threshold = {}
    for thr in REPORT_THRESHOLDS:
        tag_preds_bin = (tag_preds > thr).astype(np.int32)
        tp = (tag_preds_bin * tag_vectors).sum(axis=0)
        fp_arr = (tag_preds_bin * (1 - tag_vectors)).sum(axis=0)
        fn = ((1 - tag_preds_bin) * tag_vectors).sum(axis=0)

        per_tag = []
        for i, tag in enumerate(vocab):
            prec = tp[i] / max(tp[i] + fp_arr[i], 1)
            rec = tp[i] / max(tp[i] + fn[i], 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            per_tag.append((tag, int(tag_vectors[:, i].sum()), float(prec), float(rec), float(f1)))

        per_tag.sort(key=lambda x: x[4], reverse=True)
        results_by_threshold[thr] = per_tag

        total_tp = tp.sum()
        total_fp = fp_arr.sum()
        total_fn = fn.sum()
        micro_prec = total_tp / max(total_tp + total_fp, 1)
        micro_rec = total_tp / max(total_tp + total_fn, 1)
        micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-8)

        print(f"\n[TAG HEAD @ thr={thr}] Micro: prec={micro_prec:.2f} rec={micro_rec:.2f} f1={micro_f1:.2f}")
        print(f"[TAG HEAD @ thr={thr}] Top 15 tags by val F1:")
        for tag, n, p, r, f in per_tag[:15]:
            print(f"  {tag:20s} n={n:3d}  prec={p:.2f}  rec={r:.2f}  f1={f:.2f}")

    return results_by_threshold


def evaluate_craft_head(model, val_samples, craft_classes, craft_index):
    filepaths = [s["filepath"] for s in val_samples]
    craft_ids = np.array([craft_index[s["craft_class"]] for s in val_samples])

    fp_ds = tf.data.Dataset.from_tensor_slices(filepaths).map(
        lambda fp: decode_and_augment(fp, augment=False),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(BATCH_SIZE)

    preds = model.predict(fp_ds, verbose=0)
    craft_preds = preds[0]
    pred_ids = np.argmax(craft_preds, axis=1)

    print("\n[CRAFT HEAD] Per-class validation accuracy:")
    per_class_acc = {}
    for i, cls in enumerate(craft_classes):
        mask = craft_ids == i
        n = int(mask.sum())
        if n == 0:
            print(f"  {cls}: no val samples")
            per_class_acc[cls] = None
            continue
        acc = float((pred_ids[mask] == craft_ids[mask]).mean())
        per_class_acc[cls] = acc
        print(f"  {cls}: {acc:.2%} ({int((pred_ids[mask] == craft_ids[mask]).sum())}/{n})")

    overall = float((pred_ids == craft_ids).mean())
    print(f"  overall: {overall:.2%}")

    # Prediction distribution (catches collapse like we just saw)
    from collections import Counter
    pred_labels = [craft_classes[i] for i in pred_ids]
    print("\n[CRAFT HEAD] Val prediction distribution:")
    for cls, count in sorted(Counter(pred_labels).items()):
        print(f"  {cls}: {count}")

    return overall, per_class_acc


def export_single_head_model(backbone_model, head_name, save_path):
    single = tf.keras.Model(
        inputs=backbone_model.input,
        outputs=backbone_model.get_layer(head_name).output,
    )
    single.save(str(save_path))
    print(f"[EXPORT] Saved {head_name} head to: {save_path}")


def main():
    set_seeds(RANDOM_SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[GPU] Using GPU: {gpus}")
    else:
        print("[GPU] No GPU detected, training on CPU")

    samples, vocab, craft_classes, craft_index = load_dataset()
    train_samples, val_samples = split_train_val(samples, VAL_SPLIT)

    # No oversampling this time. Use sample weights instead.
    sample_weights = None
    if USE_SAMPLE_WEIGHTING:
        sample_weights = compute_per_sample_craft_weights(train_samples, craft_classes, MAX_SAMPLE_WEIGHT)

    train_ds = build_base_dataset(
        train_samples, craft_index, sample_weights,
        augment=True, n_craft_classes=len(craft_classes),
    )
    val_ds = build_base_dataset(
        val_samples, craft_index, None,
        augment=False, n_craft_classes=len(craft_classes),
    )

    if USE_MIXUP:
        print(f"[MIXUP] Enabled with alpha={MIXUP_ALPHA}")
        train_ds = apply_mixup(train_ds, MIXUP_ALPHA)

    steps_per_epoch = len(train_samples) // BATCH_SIZE

    model, base = build_model(len(craft_classes), len(vocab))
    model.summary()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[PHASE 1] Training heads with frozen backbone...")
    compile_model(model, LEARNING_RATE_HEAD, EPOCHS_HEAD, steps_per_epoch)
    fit(model, train_ds, val_ds, EPOCHS_HEAD, "head-only", BEST_HEAD_CKPT)

    if BEST_HEAD_CKPT.exists():
        print(f"[CKPT] Loading best head-phase weights from {BEST_HEAD_CKPT}")
        model.load_weights(str(BEST_HEAD_CKPT))

    print(f"\n[PHASE 2] Fine-tuning (unfreezing backbone from layer {UNFREEZE_FROM_LAYER})...")
    base.trainable = True
    for layer in base.layers[:UNFREEZE_FROM_LAYER]:
        layer.trainable = False
    compile_model(model, LEARNING_RATE_FINETUNE, EPOCHS_FINETUNE, steps_per_epoch)
    fit(model, train_ds, val_ds, EPOCHS_FINETUNE, "fine-tune", BEST_FT_CKPT)

    if BEST_FT_CKPT.exists():
        print(f"[CKPT] Loading best fine-tune weights from {BEST_FT_CKPT}")
        model.load_weights(str(BEST_FT_CKPT))

    print("\n[EVAL] Running validation on BEST checkpoint...")
    overall_acc, per_class_acc = evaluate_craft_head(model, val_samples, craft_classes, craft_index)
    results_by_threshold = evaluate_tag_head(model, val_samples, vocab)

    with CRAFT_LABELS_FILE.open("w") as f:
        json.dump(craft_classes, f, indent=2)
    print(f"\n[SAVE] Craft labels: {CRAFT_LABELS_FILE}")

    metrics = {
        "backbone": "EfficientNetB0",
        "techniques": [
            "mixup", "label_smoothing", "cosine_lr", "gradient_clip",
            "sample_weighting", "best_ckpt",
        ],
        "craft_overall_val_accuracy": overall_acc,
        "craft_per_class_val_accuracy": per_class_acc,
        "tag_head_by_threshold": {
            str(thr): [
                {"tag": t, "n": n, "prec": p, "rec": r, "f1": f}
                for t, n, p, r, f in per_tag[:20]
            ]
            for thr, per_tag in results_by_threshold.items()
        },
    }
    with METRICS_FILE.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics: {METRICS_FILE}")

    export_single_head_model(model, "craft", CRAFT_MODEL_FILE)
    export_single_head_model(model, "tags", TAG_MODEL_FILE)

    print("\n[DONE] Training complete.")
    print(f"  Craft model: {CRAFT_MODEL_FILE}")
    print(f"  Tag model:   {TAG_MODEL_FILE}")


if __name__ == "__main__":
    main()