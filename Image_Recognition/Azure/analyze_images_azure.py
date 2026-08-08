import os
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
from dotenv import load_dotenv

# -----------------------------
# CONFIG
# -----------------------------
# Point this at your rendered previews from metadata capture
DEFAULT_IMAGES_DIR = Path("/Users/amans/Desktop/Mark-VLM/Metadata_Capture/rendered_images")

# Azure Image Analysis 4.0 endpoint (recommended)
# Docs: https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/call-analyze-image-40 :contentReference[oaicite:6]{index=6}
#
# You will set these in .env:
#   AZURE_VISION_ENDPOINT="https://<your-resource-name>.cognitiveservices.azure.com"
#   AZURE_VISION_KEY="..."
#
# NOTE: Some caption features are region-dependent in 4.0. :contentReference[oaicite:7]{index=7}
# If you run into region limitations, you can fall back to the older v3.2 "describe" endpoint. :contentReference[oaicite:8]{index=8}


def load_environment() -> Tuple[str, str]:
    load_dotenv()
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")
    if not endpoint or not key:
        raise ValueError("Missing AZURE_VISION_ENDPOINT or AZURE_VISION_KEY in .env")
    return endpoint.rstrip("/"), key


def list_images(images_dir: Path) -> List[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"]
    files = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
    return sorted(files)


def normalize_keywords(raw: List[Dict[str, Any]]) -> List[str]:
    """
    Convert Azure tag objects into lowercase keywords.
    Deduplicate and keep stable ordering.
    """
    out = []
    seen = set()
    for t in raw:
        name = str(t.get("name", "")).strip().lower()
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def analyze_image_azure(endpoint: str, key: str, image_path: Path) -> Dict[str, Any]:
    """
    Calls Azure Image Analysis 4.0 Analyze API with:
      - caption
      - tags

    API guide: :contentReference[oaicite:9]{index=9}
    Caption concepts: :contentReference[oaicite:10]{index=10}
    """
    url = f"{endpoint}/computervision/imageanalysis:analyze"

    params = {
        # Request caption + tags.
        # Azure Image Analysis supports multiple visual features; these are the two you want for “Describe + keywords”.
        "features": "caption,tags",
        # Optional: "language": "en"
    }

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
    }

    img_bytes = image_path.read_bytes()
    resp = requests.post(url, params=params, headers=headers, data=img_bytes, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Azure error {resp.status_code}: {resp.text}")

    return resp.json()


def extract_caption_and_tags(payload: Dict[str, Any]) -> Tuple[str, float, List[Dict[str, Any]]]:
    caption = ""
    confidence = None
    tags = []

    # Image Analysis 4.0 returns caption in "captionResult"
    cap = payload.get("captionResult") or {}
    caption = (cap.get("text") or "").strip().lower()
    confidence = cap.get("confidence", None)

    # tags in "tagsResult"
    tr = payload.get("tagsResult") or {}
    tags = tr.get("values", []) or []

    return caption, confidence, tags


def main():
    endpoint, key = load_environment()

    images_dir = DEFAULT_IMAGES_DIR
    image_files = list_images(images_dir)

    if not image_files:
        print(f"No images found in: {images_dir}")
        return

    print(f"Found {len(image_files)} image(s) to analyze from {images_dir}")
    print("=" * 60)

    for image_path in image_files:
        print(f"\nAnalyzing: {image_path.name}")
        print("-" * 40)

        try:
            payload = analyze_image_azure(endpoint, key, image_path)
            caption, conf, raw_tags = extract_caption_and_tags(payload)
            keywords = normalize_keywords(raw_tags)

            # Match your demo format: a single comma-separated line
            print(f"Caption: {caption} (conf={conf})")
            print(f"Keywords: {', '.join(keywords)}")

        except Exception as e:
            print(f"Error: {e}")

        print("=" * 60)


if __name__ == "__main__":
    main()
