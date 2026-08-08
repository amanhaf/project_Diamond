import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root if present, then fall back to normal dotenv search
load_dotenv(dotenv_path=Path("/Users/amans/Desktop/Mark-VLM/.env"))
load_dotenv()

from clarifai.client.user import User
from clarifai.client.app import App
from clarifai.client.model import Model
from clarifai.client.input import Inputs

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# -----------------------------
# CONFIG
# -----------------------------
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
CLARIFAI_USER_ID = os.getenv("CLARIFAI_USER_ID")
CLARIFAI_TRAIN_USER_ID = os.getenv("CLARIFAI_TRAIN_USER_ID") or CLARIFAI_USER_ID
CLARIFAI_TRAIN_APP_ID = os.getenv("CLARIFAI_TRAIN_APP_ID", "diamond-training")

MODEL_ID = os.getenv("CLARIFAI_MODEL_ID", "diamond-craft-classifier-v6")
DATASET_ID = os.getenv("CLARIFAI_DATASET_ID", "embroidery-prototype-dataset-v4")
VAL_DATASET_ID = os.getenv("CLARIFAI_VAL_DATASET_ID", "embroidery-prototype-val-v4")

TRAIN_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition/clarifai_train")
VAL_ROOT = Path("/Users/amans/Desktop/Mark-VLM/Image_Recognition/clarifai_val")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".webp"}
MAX_UPLOAD_BATCH_SIZE = 128

# -----------------------------
# INFERENCE TEST SETTINGS
# -----------------------------
TRAINING_POLL_INTERVAL = 30        # seconds between status checks
MAX_TRAINING_WAIT_SECONDS = 1800   # give up after 30 minutes
MIN_INFERENCE_CONF = 0.05          # show any concept with at least 5% confidence

# How long to wait after uploading inputs before adding them to the dataset
POST_UPLOAD_WAIT_SECONDS = 10

# How long to retry inference after training finishes (deployment can lag)
INFERENCE_MAX_WAIT_SECONDS = 300
INFERENCE_RETRY_INTERVAL = 20


def count_images_in_folder_tree(folder: Path) -> int:
    total = 0
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            total += 1
    return total


def get_label_dirs(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_dir()])


def get_label_names(folder: Path):
    return [p.name for p in get_label_dirs(folder)]


def upload_folder_to_dataset(
    user_id: str,
    app_id: str,
    pat: str,
    dataset_id: str,
    folder: Path,
    batch_size: int = MAX_UPLOAD_BATCH_SIZE,
) -> int:
    """
    Upload labeled images AND explicitly add them to the specified dataset.
    """
    if not folder.exists():
        print(f"[WARN] Folder does not exist: {folder}")
        return 0

    class_dirs = get_label_dirs(folder)
    if not class_dirs:
        print(f"[WARN] No label subfolders found inside: {folder}")
        return 0

    all_input_protos = []

    print(f"Scanning label folders in: {folder}")
    for class_dir in class_dirs:
        image_files = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not image_files:
            print(f"[WARN] No supported images found in label folder: {class_dir}")
            continue

        print(f"Preparing {len(image_files)} image(s) for label: {class_dir.name}")

        input_protos = Inputs.get_image_inputs_from_folder(
            folder_path=str(class_dir),
            dataset_id=dataset_id,
            labels=True,
        )

        if not input_protos:
            print(f"[WARN] No input protos created from label folder: {class_dir}")
            continue

        all_input_protos.extend(input_protos)

    if not all_input_protos:
        print(f"[WARN] No input protos created from any subfolder in: {folder}")
        return 0

    print(f"Prepared {len(all_input_protos)} total input proto(s) from {folder}")

    input_obj = Inputs(user_id=user_id, app_id=app_id, pat=pat)

    total_uploaded = 0
    uploaded_input_ids = []
    total_batches = (len(all_input_protos) + batch_size - 1) // batch_size

    for i in range(0, len(all_input_protos), batch_size):
        batch = all_input_protos[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        print(
            f"Uploading batch {batch_num}/{total_batches} "
            f"({len(batch)} input proto(s))..."
        )

        input_job_id, response = input_obj.upload_inputs(batch, show_log=True)
        print(f"Upload job id: {input_job_id}")

        for proto in batch:
            uploaded_input_ids.append(proto.id)

        total_uploaded += len(batch)

    print(f"Uploaded {total_uploaded} input proto(s) total from {folder}")

    # -----------------------------
    # Explicitly add inputs to the dataset
    # -----------------------------
    print(f"Waiting {POST_UPLOAD_WAIT_SECONDS}s for Clarifai to index uploads...")
    time.sleep(POST_UPLOAD_WAIT_SECONDS)

    print(f"Adding {len(uploaded_input_ids)} input(s) to dataset '{dataset_id}'...")

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", "Key " + pat),)

    added = 0
    for i in range(0, len(uploaded_input_ids), batch_size):
        batch_ids = uploaded_input_ids[i:i + batch_size]
        dataset_inputs = [
            resources_pb2.DatasetInput(
                input=resources_pb2.Input(id=input_id)
            )
            for input_id in batch_ids
        ]

        req = service_pb2.PostDatasetInputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            dataset_id=dataset_id,
            dataset_inputs=dataset_inputs,
        )
        resp = stub.PostDatasetInputs(req, metadata=metadata)

        if resp.status.code == status_code_pb2.SUCCESS:
            added += len(batch_ids)
            print(f"  Added {len(batch_ids)} to dataset (running total: {added})")
        else:
            print(
                f"  [WARN] Failed to add batch to dataset: "
                f"{resp.status.description} | {resp.status.details}"
            )

    print(f"Finished adding {added} input(s) to dataset '{dataset_id}'")
    return total_uploaded


def create_concepts_if_needed(app: App, concept_ids: list[str]) -> None:
    if not concept_ids:
        raise ValueError("No concept IDs provided.")

    try:
        app.create_concepts(concept_ids=concept_ids)
        print(f"Created concepts: {concept_ids}")
    except Exception as e:
        print(f"Using existing concepts (or partially existing): {concept_ids}")
        print(f"[INFO] create_concepts returned: {e}")


def ensure_dataset_exists(app: App, dataset_id: str) -> None:
    try:
        app.create_dataset(dataset_id=dataset_id)
        print(f"Created dataset: {dataset_id}")
    except Exception:
        try:
            app.dataset(dataset_id=dataset_id)
            print(f"Using existing dataset: {dataset_id}")
        except Exception as e:
            raise RuntimeError(f"Could not create or access dataset '{dataset_id}': {e}")


def create_model_if_needed_sdk(
    user_id: str,
    app_id: str,
    pat: str,
    model_id: str,
) -> None:
    """
    Create the model via the SDK as a visual-classifier.
    """
    app = App(user_id=user_id, app_id=app_id, pat=pat)

    try:
        app.create_model(
            model_id=model_id,
            model_type_id="visual-classifier",
        )
        print(f"Created model via SDK: {model_id}")
    except Exception as e:
        print(f"Model already exists or could not be created: {model_id}")
        print(f"[INFO] create_model returned: {e}")


def train_model_via_sdk(
    user_id: str,
    app_id: str,
    pat: str,
    model_id: str,
    concept_ids: list[str],
    dataset_id: str,
) -> str:
    """
    Train the model using Clarifai's high-level SDK.

    Uses MMClassification_EfficientNet, which is Clarifai's documented template
    for the visual-classifier model type (per their official image-classification
    training notebook). Clarifai_ResNext is for visual-embedder, not this type.
    """
    print(f"Loading model via SDK: {model_id}")
    model = Model(
        user_id=user_id,
        app_id=app_id,
        model_id=model_id,
        pat=pat,
    )

    # List templates available for this model type so we know what's valid.
    try:
        print("Available training templates for this model type:")
        templates = model.list_training_templates()
        for t in templates:
            print(f"  - {t}")
    except Exception as e:
        print(f"[WARN] Could not list training templates: {e}")
        templates = []

    # MMClassification_EfficientNet is Clarifai's documented template for
    # visual-classifier (from their official training notebook).
    template_to_try = "MMClassification_EfficientNet"
    print(f"Fetching training params for template: {template_to_try}")

    try:
        model.get_params(template=template_to_try)
    except Exception as e:
        print(f"[WARN] get_params failed for {template_to_try}: {e}")
        print("Listing available templates and falling back to first option...")
        if not templates:
            templates = model.list_training_templates()
        if not templates:
            raise RuntimeError("No training templates available for this model type.")
        template_to_try = templates[0]
        print(f"Falling back to template: {template_to_try}")
        model.get_params(template=template_to_try)

    # Stage the dataset + concepts on the model.
    print(f"Updating params: dataset_id={dataset_id}, concepts={concept_ids}")
    model.update_params(
        dataset_id=dataset_id,
        concepts=concept_ids,
    )

    # Print the resolved training params for transparency.
    try:
        print("Resolved training params:")
        print(model.training_params)
    except Exception:
        pass

    # Kick off training.
    print("Starting training...")
    model_version_id = model.train()

    print(f"Training kicked off. Model version: {model_version_id}")
    return model_version_id


def get_model_training_status_grpc(
    user_id: str,
    app_id: str,
    pat: str,
    model_id: str,
    version_id: str,
):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", "Key " + pat),)

    request = service_pb2.GetModelVersionRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
        model_id=model_id,
        version_id=version_id,
    )

    response = stub.GetModelVersion(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        raise RuntimeError(response.status)

    return response.model_version.status


def wait_for_training_to_finish(
    user_id: str,
    app_id: str,
    pat: str,
    model_id: str,
    version_id: str,
    poll_interval: int = TRAINING_POLL_INTERVAL,
    max_wait: int = MAX_TRAINING_WAIT_SECONDS,
) -> bool:
    """
    Poll model version status until training is done, failed, or timed out.

    Clarifai status codes of interest:
      21100 = MODEL_TRAINED (done, success)
      21101 = MODEL_TRAINING
      21102 = MODEL_QUEUED_FOR_TRAINING
      21106 = MODEL_TRAINING_FAILED
    """
    print(f"\nWaiting for training to complete (checking every {poll_interval}s)...")
    elapsed = 0

    while elapsed < max_wait:
        try:
            status = get_model_training_status_grpc(
                user_id=user_id,
                app_id=app_id,
                pat=pat,
                model_id=model_id,
                version_id=version_id,
            )
        except Exception as e:
            print(f"[WARN] Could not fetch training status: {e}")
            time.sleep(poll_interval)
            elapsed += poll_interval
            continue

        code = status.code
        description = status.description or ""
        print(f"  [{elapsed}s] status code={code} description={description}")

        if code == 21100:
            print("Training finished successfully.")
            return True

        if code not in (21101, 21102):
            print(f"[ERROR] Training did not succeed. Final status: {status}")
            return False

        time.sleep(poll_interval)
        elapsed += poll_interval

    print(f"[WARN] Training still not finished after {max_wait}s. Giving up on waiting.")
    return False


def find_sample_image_per_class(folder: Path) -> list[Path]:
    """
    Return one sample image per class subfolder.
    """
    samples = []
    for class_dir in get_label_dirs(folder):
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
                samples.append(p)
                break
    return samples


def test_inference_on_sample_grpc(
    user_id: str,
    app_id: str,
    pat: str,
    model_id: str,
    model_version_id: str,
    sample_image_path: Path,
) -> None:
    """
    Run inference on a single sample image using the newly trained model.
    Retries a few times because deployment after training can lag.
    """
    if not sample_image_path.exists():
        print(f"[WARN] Sample image not found: {sample_image_path}")
        return

    expected_label = sample_image_path.parent.name
    print(f"\n--- Inference test ---")
    print(f"Image:          {sample_image_path.name}")
    print(f"Expected label: {expected_label}")

    image_bytes = sample_image_path.read_bytes()

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", "Key " + pat),)

    elapsed = 0
    last_status = None
    response = None

    while elapsed < INFERENCE_MAX_WAIT_SECONDS:
        request = service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
            model_id=model_id,
            version_id=model_version_id,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(base64=image_bytes)
                    )
                )
            ],
        )

        response = stub.PostModelOutputs(request, metadata=metadata)
        last_status = response.status

        if response.status.code == status_code_pb2.SUCCESS:
            break

        print(f"[WARN] Inference not ready yet: {response.status}")
        time.sleep(INFERENCE_RETRY_INTERVAL)
        elapsed += INFERENCE_RETRY_INTERVAL

    if last_status is None or last_status.code != status_code_pb2.SUCCESS:
        print(f"[ERROR] Inference failed after waiting {INFERENCE_MAX_WAIT_SECONDS}s: {last_status}")
        return

    concepts = (response.outputs[0].data.concepts or []) if response.outputs else []

    if not concepts:
        print("[WARN] No concepts returned from inference.")
        return

    sorted_concepts = sorted(
        concepts, key=lambda c: float(c.value or 0.0), reverse=True
    )

    print("Predicted tags:")
    top_label = None
    top_conf = 0.0
    for concept in sorted_concepts:
        name = (concept.name or "").strip().lower()
        conf = float(concept.value or 0.0)

        if conf >= MIN_INFERENCE_CONF:
            print(f"  {name:<15} {conf:.2%}")

        if conf > top_conf:
            top_conf = conf
            top_label = name

    if top_label:
        match = "CORRECT" if top_label == expected_label.lower() else "WRONG"
        print(f"Top prediction: {top_label} ({top_conf:.2%}) --> {match}")


def run_inference_tests(
    user_id: str,
    app_id: str,
    pat: str,
    model_id: str,
    model_version_id: str,
    train_root: Path,
) -> None:
    """
    Run inference on one sample per class from the training folder.
    """
    samples = find_sample_image_per_class(train_root)

    if not samples:
        print("[WARN] No sample images found for inference testing.")
        return

    print(f"\n============================================")
    print(f"Running inference test on {len(samples)} sample image(s),")
    print(f"one per class, using model '{model_id}' version '{model_version_id}'")
    print(f"============================================")

    for sample in samples:
        test_inference_on_sample_grpc(
            user_id=user_id,
            app_id=app_id,
            pat=pat,
            model_id=model_id,
            model_version_id=model_version_id,
            sample_image_path=sample,
        )


def main():
    if not CLARIFAI_PAT:
        raise ValueError("CLARIFAI_PAT not set")
    if not CLARIFAI_TRAIN_USER_ID:
        raise ValueError("CLARIFAI_TRAIN_USER_ID or CLARIFAI_USER_ID not set")
    if not CLARIFAI_TRAIN_APP_ID:
        raise ValueError("CLARIFAI_TRAIN_APP_ID not set")
    if not MODEL_ID:
        raise ValueError("CLARIFAI_MODEL_ID not set")
    if not DATASET_ID:
        raise ValueError("CLARIFAI_DATASET_ID not set")
    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_ROOT}")

    # Safety check against known-broken legacy IDs
    if MODEL_ID in {
        "diamond-craft-classifier",
        "diamond-craft-classifier-v3",
        "diamond-craft-classifier-v4",
        "diamond-craft-classifier-v5",
        "diamond-craft-classifier-v6",
    }:
        raise ValueError(
            f"MODEL_ID '{MODEL_ID}' is a known-broken legacy model. "
            f"Bump to a fresh ID like 'diamond-craft-classifier-v7' in your .env."
        )
    if DATASET_ID in {
        "embroidery-prototype-dataset",
        "embroidery-prototype-dataset-v2",
        "embroidery-prototype-dataset-v3",
    }:
        raise ValueError(
            f"DATASET_ID '{DATASET_ID}' is a legacy dataset. "
            f"Bump to a fresh ID like 'embroidery-prototype-dataset-v4' in your .env."
        )

    print(f"Using Clarifai user: {CLARIFAI_TRAIN_USER_ID}")
    print(f"Using app id:        {CLARIFAI_TRAIN_APP_ID}")
    print(f"Using model id:      {MODEL_ID}")
    print(f"Using dataset id:    {DATASET_ID}")

    label_names = get_label_names(TRAIN_ROOT)
    if not label_names:
        raise ValueError(f"No label subfolders found in {TRAIN_ROOT}")

    print(f"Discovered training labels: {label_names}")

    train_disk_count = count_images_in_folder_tree(TRAIN_ROOT)
    print(f"Found {train_disk_count} training image(s) on disk under {TRAIN_ROOT}")

    if train_disk_count == 0:
        raise ValueError(
            f"No supported training images found under {TRAIN_ROOT}."
        )

    user = User(user_id=CLARIFAI_TRAIN_USER_ID, pat=CLARIFAI_PAT)

    try:
        app = user.create_app(app_id=CLARIFAI_TRAIN_APP_ID, base_workflow="Universal")
        print(f"Created app: {CLARIFAI_TRAIN_APP_ID}")
    except Exception:
        app = App(
            user_id=CLARIFAI_TRAIN_USER_ID,
            app_id=CLARIFAI_TRAIN_APP_ID,
            pat=CLARIFAI_PAT,
        )
        print(f"Using existing app: {CLARIFAI_TRAIN_APP_ID}")

    create_concepts_if_needed(app, label_names)

    ensure_dataset_exists(app, DATASET_ID)

    print("Uploading training data...")
    train_count = upload_folder_to_dataset(
        user_id=CLARIFAI_TRAIN_USER_ID,
        app_id=CLARIFAI_TRAIN_APP_ID,
        pat=CLARIFAI_PAT,
        dataset_id=DATASET_ID,
        folder=TRAIN_ROOT,
    )
    print("Training upload complete.")

    if train_count == 0:
        raise ValueError("No training images were uploaded. Cannot continue.")

    val_dataset_id = None
    if VAL_ROOT.exists():
        val_disk_count = count_images_in_folder_tree(VAL_ROOT)
        print(f"Found {val_disk_count} validation image(s) on disk under {VAL_ROOT}")

        if val_disk_count > 0:
            val_dataset_id = VAL_DATASET_ID
            ensure_dataset_exists(app, val_dataset_id)

            print("Uploading validation data...")
            val_count = upload_folder_to_dataset(
                user_id=CLARIFAI_TRAIN_USER_ID,
                app_id=CLARIFAI_TRAIN_APP_ID,
                pat=CLARIFAI_PAT,
                dataset_id=val_dataset_id,
                folder=VAL_ROOT,
            )
            print("Validation upload complete.")

            if val_count == 0:
                print("[WARN] No validation images were uploaded.")
                val_dataset_id = None

    create_model_if_needed_sdk(
        user_id=CLARIFAI_TRAIN_USER_ID,
        app_id=CLARIFAI_TRAIN_APP_ID,
        pat=CLARIFAI_PAT,
        model_id=MODEL_ID,
    )

    print("Starting training via Clarifai SDK...")
    model_version_id = train_model_via_sdk(
        user_id=CLARIFAI_TRAIN_USER_ID,
        app_id=CLARIFAI_TRAIN_APP_ID,
        pat=CLARIFAI_PAT,
        model_id=MODEL_ID,
        concept_ids=label_names,
        dataset_id=DATASET_ID,
    )

    trained_ok = wait_for_training_to_finish(
        user_id=CLARIFAI_TRAIN_USER_ID,
        app_id=CLARIFAI_TRAIN_APP_ID,
        pat=CLARIFAI_PAT,
        model_id=MODEL_ID,
        version_id=model_version_id,
    )

    if trained_ok:
        run_inference_tests(
            user_id=CLARIFAI_TRAIN_USER_ID,
            app_id=CLARIFAI_TRAIN_APP_ID,
            pat=CLARIFAI_PAT,
            model_id=MODEL_ID,
            model_version_id=model_version_id,
            train_root=TRAIN_ROOT,
        )

        print("\n============================================")
        print("NEXT STEP")
        print("============================================")
        print("To use this trained model in run_compare_aws_azure_clarifai.py,")
        print("update your .env with these values:\n")
        print(f"  CLARIFAI_USER_ID={CLARIFAI_TRAIN_USER_ID}")
        print(f"  CLARIFAI_APP_ID={CLARIFAI_TRAIN_APP_ID}")
        print(f"  CLARIFAI_MODEL_ID={MODEL_ID}")
        print(f"  CLARIFAI_MODEL_VERSION_ID={model_version_id}")
    else:
        print("\n[WARN] Skipping inference test because training did not complete.")
        print(f"Model version ID (still usable later): {model_version_id}")

    if val_dataset_id:
        print(f"\n[INFO] Validation dataset is ready: {val_dataset_id}")


if __name__ == "__main__":
    main()