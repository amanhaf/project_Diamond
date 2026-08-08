import os
from pathlib import Path
from dotenv import load_dotenv

from azure_vision import (
    azure_analyze_image,
    azure_describe_image,
    azure_keywords_from_analyze,
    azure_caption_from_describe,
)

# Load .env from project root (Mark-VLM/.env)
root = Path(__file__).resolve().parents[2]
load_dotenv(root / ".env")

endpoint = (os.getenv("AZURE_VISION_ENDPOINT") or "").strip().rstrip("/")
key = (os.getenv("AZURE_VISION_KEY") or "").strip().replace("\ufeff", "")

print("Endpoint loaded:", endpoint)
print("Endpoint repr:", repr(endpoint))
print("Key loaded:", bool(key))
print("Key length:", len(key))

if not endpoint or not key:
    raise RuntimeError("Azure endpoint or key not loaded from .env")

img_path = Path(
    "/Users/amans/Desktop/Mark-VLM/Metadata_Capture/rendered_images/E-AGD-Ih-da1.png"
)
img = img_path.read_bytes()

an = azure_analyze_image(img, endpoint, key)
tags = azure_keywords_from_analyze(an)
print("\nTAGS:", [t[0] for t in tags])

desc = azure_describe_image(img, endpoint, key)
cap = azure_caption_from_describe(desc)
print("\nCAPTION:", cap)