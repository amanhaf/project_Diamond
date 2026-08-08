import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from io import BytesIO

import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageFile

# --------------------------------------------------
# PIL SAFETY
# --------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 250_000_000
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --------------------------------------------------
# Ensure project root is on sys.path
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# IMPORTANT: your folder is "Azure" (capital A)
from Image_Recognition.Azure.azure_vision import (
    azure_analyze_image,
    azure_describe_image,
    azure_keywords_from_analyze,
    azure_caption_from_describe,
)

# AWS
import boto3

# Gemini
from google import genai
from google.genai import types

# Clarifai (optional)
CLARIFAI_AVAILABLE = True
try:
    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
    from clarifai_grpc.grpc.api.status import status_code_pb2
except Exception:
    CLARIFAI_AVAILABLE = False


# --------------------------------------------------
# PATHS
# --------------------------------------------------
ENV_PATH = PROJECT_ROOT / ".env"
RENDER_DIR = PROJECT_ROOT / "Metadata_Capture" / "rendered_images"
METADATA_CSV = PROJECT_ROOT / "Metadata_Capture" / "metadata_results.csv"
OUT_CSV = PROJECT_ROOT / "Image_Recognition" / "image_recognition_results.csv"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
MAX_DIM_FOR_APIS = 2048
AWS_MAX_BYTES = 5 * 1024 * 1024
AWS_MIN_CONF = 60.0
MAX_AWS_LABELS = 25

AZ_MIN_CONF = 0.20
AZ_MAX_TAGS = 25

CL_MIN_CONF = 0.20
CL_MAX_CONCEPTS = 25

GEMINI_MODEL_DEFAULT = "gemini-2.5-pro"

PROMPT = """You are an expert at analyzing images, particularly designs intended for crafts like embroidery, appliqué, and quilting. Your task is to generate a concise, comma-separated list of relevant keywords for the provided image.

Follow these rules carefully:

Identify Core Subjects: List all primary objects, animals, plants, and distinct shapes in the image.
Example: rabbit, carrot, rose, heart, butterfly.

Include General Categories: Add broader categories for the identified subjects.
Example: If you see a rabbit, add animal. If you see a rose, add flower.

Include Thematic Categories: If applicable, add a thematic description.
Example: For a rabbit, woodland creature is a good thematic keyword.

Identify Text: If the image contains any words or letters, simply add the keyword text. Do not write out the actual words.

Identify Craft Style: If the image is clearly a pattern or design for a specific craft, include that craft as a keyword.
Example: If the image is a continuous line drawing meant for a longarm sewing machine, add quilting.

Formatting: The final output must be a single line of lowercase, comma-separated keywords. Do not use any other formatting.

Now, analyze the following image and generate the keywords.
"""


# --------------------------------------------------
# ENV
# --------------------------------------------------
def load_cfg() -> Dict[str, str]:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f".env not found at {ENV_PATH}")

    load_dotenv(ENV_PATH)

    required = [
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_VISION_ENDPOINT",
        "AZURE_VISION_KEY",
        "GEMINI_API_KEY",
    ]

    missing = [k for k in required if not (os.getenv(k) or "").strip()]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

    cfg = {
        "aws_region": os.getenv("AWS_REGION").strip(),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY").strip(),
        "azure_endpoint": os.getenv("AZURE_VISION_ENDPOINT").strip().rstrip("/"),
        "azure_key": (os.getenv("AZURE_VISION_KEY") or "").strip().replace("\ufeff", ""),
        "gemini_api_key": os.getenv("GEMINI_API_KEY").strip(),
        "gemini_model": os.getenv("GEMINI_MODEL", GEMINI_MODEL_DEFAULT).strip(),
    }

    for k in ["CLARIFAI_PAT", "CLARIFAI_USER_ID", "CLARIFAI_APP_ID", "CLARIFAI_MODEL_ID"]:
        v = os.getenv(k)
        if v and v.strip():
            cfg[k.lower()] = v.strip()

    return cfg


def list_images(images_dir: Path) -> List[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.webp"]
    files: List[Path] = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
    return sorted(files)


# --------------------------------------------------
# IMAGE PREP
# --------------------------------------------------
def prepare_image_bytes_for_apis(
    image_bytes: bytes,
    max_dim: int = MAX_DIM_FOR_APIS,
    max_bytes: int = AWS_MAX_BYTES,
) -> tuple[bytes, str]:
    """
    Safely open, downscale, and compress an image:
      - Downscale so max(width,height) <= max_dim
      - Ensure <= max_bytes by JPEG compression if needed
    Returns (bytes, mime_type)
    """
    img = Image.open(BytesIO(image_bytes))
    img.load()

    w, h = img.size
    longest = max(w, h)

    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img = img.convert("RGB")

    # Start with JPEG since Gemini code already uses JPEG and AWS is size-limited
    for quality in [88, 80, 70, 60, 50, 40]:
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        b = out.getvalue()
        if len(b) <= max_bytes:
            return b, "image/jpeg"

    # Fallback: return last attempt
    return b, "image/jpeg"


# --------------------------------------------------
# AWS
# --------------------------------------------------
def aws_labels(image_bytes: bytes, cfg: Dict[str, str]) -> List[Tuple[str, float]]:
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


# --------------------------------------------------
# GEMINI
# --------------------------------------------------
def gemini_client(cfg: Dict[str, str]) -> genai.Client:
    return genai.Client(api_key=cfg["gemini_api_key"])


def gemini_tags(client: genai.Client, model_name: str, image_bytes: bytes, mime_type: str) -> str:
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=PROMPT),
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
    ]

    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
    )

    return (resp.text or "").strip() or "(empty response)"


# --------------------------------------------------
# CLARIFAI
# --------------------------------------------------
def clarifai_concepts(image_bytes: bytes, cfg: Dict[str, str]) -> List[Tuple[str, float]]:
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", f"Key {cfg['clarifai_pat']}"),)

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(
            user_id=cfg["clarifai_user_id"],
            app_id=cfg["clarifai_app_id"],
        ),
        model_id=cfg["clarifai_model_id"],
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(image=resources_pb2.Image(base64=image_bytes))
            )
        ],
    )

    response = stub.PostModelOutputs(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        raise RuntimeError(f"Clarifai error: {response.status.description}")

    out: List[Tuple[str, float]] = []
    concepts = (response.outputs[0].data.concepts or []) if response.outputs else []
    for c in concepts[:CL_MAX_CONCEPTS]:
        name = (c.name or "").strip().lower()
        val = float(c.value or 0.0)
        if name and val >= CL_MIN_CONF:
            out.append((name, val))
    return out


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    cfg = load_cfg()
    g_client = gemini_client(cfg)
    gemini_model_name = cfg["gemini_model"]

    images = list_images(RENDER_DIR)
    if not images:
        print(f"No images found in: {RENDER_DIR}")
        return

    print(f"Found {len(images)} images.")

    meta_df = None
    if METADATA_CSV.exists():
        meta_df = pd.read_csv(METADATA_CSV)
        meta_df["render_file"] = meta_df["render_file"].astype(str)

    clarifai_enabled = (
        CLARIFAI_AVAILABLE
        and all(k in cfg for k in ["clarifai_pat", "clarifai_user_id", "clarifai_app_id", "clarifai_model_id"])
    )

    if not CLARIFAI_AVAILABLE:
        print("Clarifai not installed; skipping.")
    elif not clarifai_enabled:
        print("Clarifai env not fully set; skipping.")

    rows: List[Dict[str, Any]] = []

    for img_path in images:
        print(f"\nAnalyzing: {img_path.name}")

        raw = img_path.read_bytes()
        prepared, mime_type = prepare_image_bytes_for_apis(raw)

        # --------------------
        # Gemini
        # --------------------
        gemini_out = None
        gemini_err = None
        try:
            gemini_out = gemini_tags(g_client, gemini_model_name, prepared, mime_type)
        except Exception as e:
            gemini_err = str(e)

        # --------------------
        # Azure
        # --------------------
        az_tags: List[Tuple[str, float]] = []
        az_caption: Optional[Tuple[str, float]] = None
        az_err = None

        try:
            an = azure_analyze_image(prepared, cfg["azure_endpoint"], cfg["azure_key"])
            az_tags = azure_keywords_from_analyze(an, min_conf=AZ_MIN_CONF, max_tags=AZ_MAX_TAGS)

            desc = azure_describe_image(prepared, cfg["azure_endpoint"], cfg["azure_key"])
            az_caption = azure_caption_from_describe(desc)
        except Exception as e:
            az_err = str(e)

        # --------------------
        # AWS
        # --------------------
        aws_out: List[Tuple[str, float]] = []
        aws_err = None

        try:
            aws_out = aws_labels(prepared, cfg)
        except Exception as e:
            aws_err = str(e)

        # --------------------
        # Clarifai
        # --------------------
        cla_out: List[Tuple[str, float]] = []
        cla_err = None

        if clarifai_enabled:
            try:
                cla_out = clarifai_concepts(prepared, cfg)
            except Exception as e:
                cla_err = str(e)
                if "quota" in cla_err.lower() or "limit" in cla_err.lower():
                    print("Clarifai quota/limit hit; disabling Clarifai for rest of run.")
                    clarifai_enabled = False

        row: Dict[str, Any] = {
            "image_file": img_path.name,

            "gemini_tags": gemini_out,
            "azure_tags": ", ".join([t[0] for t in az_tags]),
            "azure_caption": az_caption[0] if az_caption else None,
            "azure_caption_conf": az_caption[1] if az_caption else None,
            "aws_labels": ", ".join([t[0] for t in aws_out]),
            "clarifai_concepts": ", ".join([t[0] for t in cla_out]),

            "gemini_error": gemini_err,
            "azure_error": az_err,
            "aws_error": aws_err,
            "clarifai_error": cla_err,

            "prepared_bytes_len": len(prepared),
        }

        if meta_df is not None:
            m = meta_df[meta_df["render_file"] == img_path.name]
            if not m.empty:
                m0 = m.iloc[0].to_dict()
                row.update({
                    "original_file": m0.get("original_file"),
                    "normalized_file": m0.get("normalized_file"),
                    "converted": m0.get("converted"),
                    "emb_width_in": m0.get("width_in"),
                    "emb_height_in": m0.get("height_in"),
                    "stitch_count": m0.get("stitch_count"),
                    "color_count": m0.get("color_count"),
                })

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to: {OUT_CSV}")


if __name__ == "__main__":
    main()