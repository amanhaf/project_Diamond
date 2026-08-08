#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import imagehash
import pyembroidery
import requests
from PIL import Image, ImageDraw, ImageOps
from pyembroidery import STITCH

# Local VIP reader (pyembroidery has no VIP reader of its own).
from vip_reader import read_vip

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RENDER_DIR = PROJECT_ROOT / "Metadata_Capture" / "rendered_images"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

CONVERT_DIR = PROJECT_ROOT / "Metadata_Capture" / "vip+art_converted"
CONVERT_DIR.mkdir(parents=True, exist_ok=True)

# .vip is now read natively (see load_pattern); only .art still needs the
# external converter.
SUPPORTED = (".pes", ".dst", ".exp", ".jef", ".vp3", ".hus", ".vip")
NEEDS_CONVERSION = (".art",)

CONVERTER_URL = "http://127.0.0.1:8000/convert"
CONVERTER_TIMEOUT_SECONDS = 180

# UUID settings — must stay stable forever once shipped
DIAMOND_NAMESPACE = uuid.UUID("2b5c2a8c-8b0c-4b2d-9d1c-3f2f5c2a8c10")
PHASH_SIZE = 16
CANON_SIZE = 512
MARGIN_PX = 8


# -----------------------------
# CONVERSION
# -----------------------------
def convert_to_pes(src_path: Path) -> tuple[Path | None, str | None]:
    dst_path = CONVERT_DIR / f"{src_path.stem}.pes"
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return dst_path, None
    try:
        with open(src_path, "rb") as f:
            response = requests.post(
                CONVERTER_URL,
                files={"file": (src_path.name, f, "application/octet-stream")},
                data={"target_format": "pes"},
                timeout=CONVERTER_TIMEOUT_SECONDS,
            )
        if response.status_code != 200:
            return None, f"Converter HTTP {response.status_code}: {response.text}"
        with open(dst_path, "wb") as out:
            out.write(response.content)
        if not dst_path.exists() or dst_path.stat().st_size == 0:
            return None, "Converter returned empty PES file"
        return dst_path, None
    except Exception as e:
        return None, f"Converter request failed: {e}"


# -----------------------------
# READ WITH FALLBACK
# -----------------------------
def load_pattern(file_path: Path):
    suffix = file_path.suffix.lower()

    # pyembroidery has no VIP reader, so handle .vip with our own reader.
    # VIP is read natively here and never goes through the converter.
    if suffix == ".vip":
        try:
            pattern = read_vip(file_path)
            if pattern and len(pattern.stitches) > 0:
                return pattern, file_path, False, None
            return None, file_path, False, "VIP read produced no stitches"
        except Exception as e:
            return None, file_path, False, f"VIP read failed: {e}"

    direct_error = None
    conversion_error = None
    try:
        pattern = pyembroidery.read(str(file_path))
        if pattern and len(pattern.stitches) > 0:
            return pattern, file_path, False, None
    except Exception as e:
        direct_error = str(e)
    if suffix in NEEDS_CONVERSION:
        converted, conversion_error = convert_to_pes(file_path)
        if converted is not None:
            try:
                pattern = pyembroidery.read(str(converted))
                if pattern and len(pattern.stitches) > 0:
                    return pattern, converted, True, None
                conversion_error = "Converted PES was read but contained no stitches"
            except Exception as e:
                conversion_error = f"Converted PES could not be read: {e}"
    final_error = conversion_error or direct_error or "Could not read embroidery"
    return None, file_path, False, final_error


# -----------------------------
# RENDERING
#
# Bug 2 fix: only draw STITCH commands. Non-stitch commands (JUMP, TRIM,
# COLOR_CHANGE) break line continuity. Different formats encode the same
# design with different JUMP placements, so drawing them would render the
# same design differently per format.
# -----------------------------
def render_pattern(pattern, output_path: Path, scale: int = 4) -> bool:
    stitches = pattern.stitches
    if not stitches:
        return False

    min_x, min_y, max_x, max_y = pattern.bounds()
    width = int((max_x - min_x) * scale)
    height = int((max_y - min_y) * scale)
    if width <= 0 or height <= 0:
        return False

    img = Image.new("RGB", (width + 20, height + 20), "white")
    draw = ImageDraw.Draw(img)

    last = None
    drew_anything = False
    for x, y, cmd in stitches:
        if cmd == STITCH:
            if last is not None:
                draw.line(
                    (
                        (last[0] - min_x) * scale,
                        (last[1] - min_y) * scale,
                        (x - min_x) * scale,
                        (y - min_y) * scale,
                    ),
                    fill="black",
                    width=1,
                )
                drew_anything = True
            last = (x, y)
        else:
            last = None

    if not drew_anything:
        return False
    img.save(output_path)
    return True


# -----------------------------
# METADATA
# -----------------------------
def extract_metadata(pattern):
    min_x, min_y, max_x, max_y = pattern.bounds()
    width_in = ((max_x - min_x) / 25.4) / 10
    height_in = ((max_y - min_y) / 25.4) / 10
    stitch_count = sum(1 for _, _, cmd in pattern.stitches if cmd == STITCH)
    threads = getattr(pattern, "threads", getattr(pattern, "threadlist", []))
    color_count = len(threads) if threads else None
    return width_in, height_in, stitch_count, color_count


# -----------------------------
# NORMALIZATION
# -----------------------------
def crop_to_content(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    gray = ImageOps.grayscale(rgb)
    inv = ImageOps.invert(gray)
    bw = inv.point(lambda p: 255 if p > 12 else 0)
    bbox = bw.getbbox()
    if not bbox:
        return rgb
    l, t, r, b = bbox
    l = max(0, l - MARGIN_PX)
    t = max(0, t - MARGIN_PX)
    r = min(rgb.width, r + MARGIN_PX)
    b = min(rgb.height, b + MARGIN_PX)
    return rgb.crop((l, t, r, b))


def normalize_render(img: Image.Image) -> Image.Image:
    cropped = crop_to_content(img)
    gray = ImageOps.grayscale(cropped)
    gray.thumbnail((CANON_SIZE, CANON_SIZE))
    canvas = Image.new("L", (CANON_SIZE, CANON_SIZE), color=255)
    x = (CANON_SIZE - gray.width) // 2
    y = (CANON_SIZE - gray.height) // 2
    canvas.paste(gray, (x, y))
    return canvas


# -----------------------------
# UUID GENERATION
#
# Bug 1 fix: pHash-only, no SHA gate. Two designs with the same perceptual
# hash get the same UUID, regardless of whether their pixel data is byte-
# identical. This is what makes cross-format matching work at all.
# -----------------------------
def uuid_from_render_png(png_path: Path):
    img = Image.open(png_path)
    norm = normalize_render(img)
    ph = imagehash.phash(norm, hash_size=PHASH_SIZE)
    phash_str = str(ph)
    duuid = uuid.uuid5(DIAMOND_NAMESPACE, f"phash:{phash_str}")
    return str(duuid), phash_str


# -----------------------------
# MAIN
# -----------------------------
def main():
    if len(sys.argv) != 2:
        print(json.dumps({"status": "error", "error": "Usage: process_file.py <path>"}))
        sys.exit(2)
    original = Path(sys.argv[1])
    try:
        if not original.exists():
            print(json.dumps({"status": "error", "original_path": str(original), "error": "File not found"}))
            sys.exit(1)
        pattern, used_path, was_converted, read_error = load_pattern(original)
        if not pattern:
            print(json.dumps({"status": "error", "original_path": str(original), "error": read_error or "Could not read embroidery"}))
            sys.exit(1)
        width, height, stitches, colors = extract_metadata(pattern)
        render_name = f"{used_path.stem}.png"
        render_path = RENDER_DIR / render_name
        rendered = render_pattern(pattern, render_path)
        if not rendered:
            print(json.dumps({"status": "error", "original_path": str(original), "error": "Render failed (no stitches)"}))
            sys.exit(1)
        duuid, phash = uuid_from_render_png(render_path)
        out = {
            "status": "ok",
            "original_path": str(original),
            "normalized_path": str(used_path),
            "converted": bool(was_converted),
            "render_path": str(render_path),
            "phash_render": phash,
            "uuid": duuid,
            "width_in": round(width, 2),
            "height_in": round(height, 2),
            "stitch_count": int(stitches),
            "color_count": int(colors) if colors is not None else None,
        }
        print(json.dumps(out))
    except Exception as e:
        print(json.dumps({"status": "error", "original_path": str(original), "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()