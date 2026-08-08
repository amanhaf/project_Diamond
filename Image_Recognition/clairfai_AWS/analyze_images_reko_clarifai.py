import os
from pathlib import Path
from typing import List, Tuple, Set
from io import BytesIO
import csv
import time
import warnings

from dotenv import load_dotenv

# AWS
import boto3

# PIL
from PIL import Image

# Clarifai gRPC
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2


# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_IMAGES_DIR = Path("/Users/amans/Desktop/Mark-VLM/Metadata_Capture/rendered_images")

MAX_AWS_LABELS = 25
AWS_MIN_CONF = 60.0
AWS_MAX_DIM = 10000  # Rekognition max dimension

MAX_CLARIFAI_CONCEPTS = 25
CLARIFAI_MIN_CONF = 0.20

# Clarifai stability controls
USE_CLARIFAI = True
CLARIFAI_SLEEP_SECONDS = 1.0                 # slow down calls to reduce rate/quota hits
CLARIFAI_CACHE_FILE = Path("clarifai_cache.csv")  # saved in your current working directory

# Debug switches
DEBUG_PRINT_CLARIFAI_APPS = False
DEBUG_PRINT_CLARIFAI_MODELS = False


# -----------------------------
# ENV LOADING
# -----------------------------
def load_env():
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"

    if not env_path.exists():
        raise FileNotFoundError(f".env not found at {env_path}")

    load_dotenv(dotenv_path=env_path)

    required = [
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "CLARIFAI_PAT",
        "CLARIFAI_USER_ID",
        "CLARIFAI_APP_ID",
        "CLARIFAI_MODEL_ID",
    ]

    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

    return {
        "aws_region": os.getenv("AWS_REGION"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "clarifai_pat": os.getenv("CLARIFAI_PAT"),
        "clarifai_user_id": os.getenv("CLARIFAI_USER_ID"),
        "clarifai_app_id": os.getenv("CLARIFAI_APP_ID"),
        "clarifai_model_id": os.getenv("CLARIFAI_MODEL_ID"),
    }


# -----------------------------
# CLARIFAI CACHE
# -----------------------------
def load_clarifai_cache(path: Path) -> dict[str, str]:
    """
    cache key: image filename
    cache value: comma-separated concepts
    """
    cache: dict[str, str] = {}
    if not path.exists():
        return cache

    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            img = (row.get("image") or "").strip()
            concepts = (row.get("concepts") or "").strip()
            if img:
                cache[img] = concepts
    return cache


def append_clarifai_cache(path: Path, image_name: str, concepts: List[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["image", "concepts"])
        w.writerow([image_name, ",".join(concepts)])


# -----------------------------
# IMAGE HELPERS
# -----------------------------
def ensure_rekognition_size(image_bytes: bytes) -> bytes:
    """
    Resize image if it exceeds AWS Rekognition max dimension.
    NOTE: this does NOT guarantee <= 5MB payload; we'll address AWS payload errors next.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        img = Image.open(BytesIO(image_bytes))

    w, h = img.size

    if w <= AWS_MAX_DIM and h <= AWS_MAX_DIM:
        return image_bytes

    scale = min(AWS_MAX_DIM / w, AWS_MAX_DIM / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)

    out = BytesIO()
    img.save(out, format="PNG", optimize=False)
    return out.getvalue()


def list_images(images_dir: Path) -> List[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.webp"]
    files: List[Path] = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
    return sorted(files)


# -----------------------------
# AWS Rekognition
# -----------------------------
def aws_labels(image_bytes: bytes, cfg: dict) -> List[Tuple[str, float]]:
    client = boto3.client(
        "rekognition",
        region_name=cfg["aws_region"],
        aws_access_key_id=cfg["aws_access_key_id"],
        aws_secret_access_key=cfg["aws_secret_access_key"],
    )

    resp = client.detect_labels(
        Image={"Bytes": image_bytes},
        MaxLabels=MAX_AWS_LABELS,
        MinConfidence=AWS_MIN_CONF,
    )

    out: List[Tuple[str, float]] = []
    for lab in resp.get("Labels", []):
        name = (lab.get("Name") or "").strip().lower()
        conf = float(lab.get("Confidence") or 0.0)
        if name:
            out.append((name, conf))
    return out


# -----------------------------
# Clarifai
# -----------------------------
def clarifai_list_apps(pat: str):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", f"Key {pat}"),)

    req = service_pb2.ListAppsRequest(page=1, per_page=100)
    resp = stub.ListApps(req, metadata=metadata)

    if resp.status.code != status_code_pb2.SUCCESS:
        raise RuntimeError(f"Clarifai list apps error: {resp.status.description}")

    print("Apps visible to this PAT:")
    for a in resp.apps:
        owner = ""
        if hasattr(a, "user_id") and a.user_id:
            owner = f" (owner user_id={a.user_id})"
        elif hasattr(a, "owner") and hasattr(a.owner, "id") and a.owner.id:
            owner = f" (owner id={a.owner.id})"
        print(" -", a.id, "|", a.name, owner)


def clarifai_list_models(pat: str, user_id: str, app_id: str):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", f"Key {pat}"),)

    req = service_pb2.ListModelsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
        page=1,
        per_page=50,
    )

    resp = stub.ListModels(req, metadata=metadata)

    if resp.status.code != status_code_pb2.SUCCESS:
        raise RuntimeError(f"Clarifai list models error: {resp.status.description}")

    print(f"Models in {user_id}/{app_id}:")
    for m in resp.models:
        print(" -", m.id, "|", getattr(m, "name", ""))


def clarifai_concepts(
    image_bytes: bytes,
    pat: str,
    user_id: str,
    app_id: str,
    model_id: str,
) -> List[Tuple[str, float]]:
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", f"Key {pat}"),)

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id),
        model_id=model_id,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=image_bytes)
                )
            )
        ],
    )

    response = stub.PostModelOutputs(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        raise RuntimeError(f"Clarifai error: {response.status.description}")

    out: List[Tuple[str, float]] = []
    concepts = response.outputs[0].data.concepts or []

    for c in concepts[:MAX_CLARIFAI_CONCEPTS]:
        name = (c.name or "").strip().lower()
        val = float(c.value or 0.0)
        if name and val >= CLARIFAI_MIN_CONF:
            out.append((name, val))

    return out


# -----------------------------
# Merge
# -----------------------------
def merge_keywords(aws: List[Tuple[str, float]], cla: List[Tuple[str, float]]) -> List[str]:
    seen: Set[str] = set()
    merged: List[str] = []

    for name, _ in cla:
        if name not in seen:
            seen.add(name)
            merged.append(name)

    for name, _ in aws:
        if name not in seen:
            seen.add(name)
            merged.append(name)

    return merged


# -----------------------------
# MAIN
# -----------------------------
def main():
    global USE_CLARIFAI

    cfg = load_env()

    # Debug outputs
    if DEBUG_PRINT_CLARIFAI_APPS:
        clarifai_list_apps(cfg["clarifai_pat"])
        return

    if DEBUG_PRINT_CLARIFAI_MODELS:
        clarifai_list_models(cfg["clarifai_pat"], cfg["clarifai_user_id"], cfg["clarifai_app_id"])
        return

    clarifai_cache = load_clarifai_cache(CLARIFAI_CACHE_FILE)

    images = list_images(DEFAULT_IMAGES_DIR)
    if not images:
        print(f"No images found in {DEFAULT_IMAGES_DIR}")
        return

    print(f"Found {len(images)} image(s) to analyze.")
    print("=" * 60)

    for img_path in images:
        print(f"\nAnalyzing: {img_path.name}")
        print("-" * 40)

        try:
            b = img_path.read_bytes()

            # AWS resize protection (dimension-only)
            b_aws = ensure_rekognition_size(b)
            aws = aws_labels(b_aws, cfg)

            # Clarifai with cache + throttle + disable-on-quota
            cla: List[Tuple[str, float]] = []

            if USE_CLARIFAI:
                if img_path.name in clarifai_cache:
                    cached = clarifai_cache[img_path.name].strip()
                    if cached:
                        cla = [(c.strip(), 1.0) for c in cached.split(",") if c.strip()]
                else:
                    try:
                        time.sleep(CLARIFAI_SLEEP_SECONDS)

                        cla = clarifai_concepts(
                            b,
                            cfg["clarifai_pat"],
                            cfg["clarifai_user_id"],
                            cfg["clarifai_app_id"],
                            cfg["clarifai_model_id"],
                        )

                        cla_names = [name for name, _ in cla]
                        append_clarifai_cache(CLARIFAI_CACHE_FILE, img_path.name, cla_names)
                        clarifai_cache[img_path.name] = ",".join(cla_names)

                    except Exception as e:
                        msg = str(e).lower()
                        if "account limits exceeded" in msg:
                            print("Clarifai skipped: Account limits exceeded (quota). Using AWS only for the rest of this run.")
                            USE_CLARIFAI = False
                        else:
                            print(f"Clarifai skipped: {e}")

            keywords = merge_keywords(aws, cla)
            print(", ".join(keywords) if keywords else "(no keywords)")

        except Exception as e:
            print(f"Error: {e}")

        print("=" * 60)


if __name__ == "__main__":
    main()