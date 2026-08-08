from __future__ import annotations

from pathlib import Path
from io import BytesIO
import hashlib
import uuid

import pandas as pd
from PIL import Image, ImageChops, ImageOps
import imagehash

from uuid_config import DIAMOND_NAMESPACE, PHASH_SIZE, CANON_SIZE, MARGIN_PX


def _crop_to_content(img: Image.Image) -> Image.Image:
    """
    Crop away blank background. Your renders are line art on white background,
    so we treat 'near-white' as background and crop to bounding box of ink.
    """
    # Ensure RGB then invert to make "ink" bright
    rgb = img.convert("RGB")

    # Convert to grayscale and invert
    gray = ImageOps.grayscale(rgb)
    inv = ImageOps.invert(gray)

    # Threshold: keep only strong ink-ish pixels
    # (line art is often faint; using a low threshold helps)
    bw = inv.point(lambda p: 255 if p > 12 else 0)

    bbox = bw.getbbox()
    if not bbox:
        return rgb  # no content detected; fallback

    left, top, right, bottom = bbox
    left = max(0, left - MARGIN_PX)
    top = max(0, top - MARGIN_PX)
    right = min(rgb.width, right + MARGIN_PX)
    bottom = min(rgb.height, bottom + MARGIN_PX)

    return rgb.crop((left, top, right, bottom))


def _normalize_render(img: Image.Image) -> Image.Image:
    """
    Make the render stable across machines:
    - crop to content (removes giant whitespace / border)
    - convert to grayscale (more stable for line art)
    - resize to fixed square using padding to preserve aspect ratio
    """
    cropped = _crop_to_content(img)

    gray = ImageOps.grayscale(cropped)

    # Preserve aspect ratio by letterboxing into CANON_SIZE x CANON_SIZE
    gray.thumbnail((CANON_SIZE, CANON_SIZE), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (CANON_SIZE, CANON_SIZE), color=255)  # white
    x = (CANON_SIZE - gray.width) // 2
    y = (CANON_SIZE - gray.height) // 2
    canvas.paste(gray, (x, y))
    return canvas


def _canonical_png_bytes(img: Image.Image) -> bytes:
    """
    Serialize deterministically so sha256 doesn't change due to encoder quirks.
    """
    buf = BytesIO()
    # Keep parameters stable
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def compute_uuid_from_png(png_path: Path) -> dict:
    """
    Returns uuid + fingerprints for a rendered preview PNG.
    """
    img = Image.open(png_path)
    norm = _normalize_render(img)

    # Exact identity (of normalized preview)
    png_bytes = _canonical_png_bytes(norm)
    sha256 = hashlib.sha256(png_bytes).hexdigest()

    # Visual identity
    ph = imagehash.phash(norm, hash_size=PHASH_SIZE)  # hex string-able
    phash_str = str(ph)

    combined = f"sha256:{sha256}|phash:{phash_str}"
    duuid = uuid.uuid5(DIAMOND_NAMESPACE, combined)

    return {
        "render_file": png_path.name,
        "uuid": str(duuid),
        "sha256_render": sha256,
        "phash_render": phash_str,
        "combined_signature": combined,
    }


def main():
    # Adjust these to your project paths
    project_root = Path(__file__).resolve().parents[1]
    meta_csv = project_root / "Metadata_Capture" / "metadata_results.csv"
    renders_dir = project_root / "Metadata_Capture" / "rendered_images"

    if not meta_csv.exists():
        raise FileNotFoundError(f"metadata_results.csv not found: {meta_csv}")
    if not renders_dir.exists():
        raise FileNotFoundError(f"rendered_images folder not found: {renders_dir}")

    df = pd.read_csv(meta_csv)

    # Ensure column exists
    if "render_file" not in df.columns:
        raise ValueError("metadata_results.csv must contain a 'render_file' column")

    # Compute UUIDs for every render_file that exists
    results = []
    missing = 0

    for rf in df["render_file"].dropna().unique():
        png_path = renders_dir / str(rf)
        if not png_path.exists():
            missing += 1
            continue
        results.append(compute_uuid_from_png(png_path))

    uuid_df = pd.DataFrame(results)

    # Merge back into metadata results
    merged = df.merge(uuid_df, on="render_file", how="left")

    out_csv = project_root / "UUID_generation" / "uuid_results_merged.csv"
    out_csv.parent.mkdir(exist_ok=True)
    merged.to_csv(out_csv, index=False)

    print(f"Saved merged results → {out_csv}")
    if missing:
        print(f"Warning: {missing} render_file(s) missing in rendered_images/")

    # Also save a slim UUID registry file (handy for client caching)
    registry_csv = project_root / "UUID_generation" / "uuid_registry.csv"
    merged[["uuid", "render_file", "sha256_render", "phash_render"]].dropna().drop_duplicates().to_csv(
        registry_csv, index=False
    )
    print(f"Saved UUID registry → {registry_csv}")


if __name__ == "__main__":
    main()
