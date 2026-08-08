#!/usr/bin/env python3
"""
Re-render all embroidery files to COLOR PNGs using their thread color metadata.

These color renders are intended for the vision-tagging pipeline only. The B&W
canonical renders produced by process_file.py remain the input to UUID
generation and must NOT be changed — cross-format UUID matching depends on
their format-invariance.

Output: <PROJECT_ROOT>/Metadata_Capture/rendered_images_color/<stem>.png

Run once:
    python rerender_color.py

Then point run_compare_aws_azure_clarifai.py's RENDER_DIR at the new folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyembroidery
from PIL import Image, ImageDraw
from pyembroidery import STITCH, COLOR_CHANGE, NEEDLE_SET

# VIP files cannot be decoded by pyembroidery; use the project's native VIP
# reader (the same one the main worker uses) so VIP designs get a color render.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vip_reader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOT = PROJECT_ROOT / "Metadata_Capture"
CONVERT_DIR = SCAN_ROOT / "vip+art_converted"
OUT_DIR = SCAN_ROOT / "rendered_images_color"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {".pes", ".dst", ".exp", ".jef", ".vp3", ".hus", ".vip", ".art"}
SCALE = 4
STROKE_WIDTH = 1


def _thread_rgb(thread) -> tuple[int, int, int]:
    """Pull (r,g,b) out of a pyembroidery EmbThread however it stores it."""
    # Most builds expose .red / .green / .blue
    try:
        return (int(thread.red), int(thread.green), int(thread.blue))
    except Exception:
        pass
    # Some builds expose get_color() returning a 0xRRGGBB int
    try:
        c = int(thread.get_color())
        return ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
    except Exception:
        pass
    # Some expose .color as 0xRRGGBB
    try:
        c = int(thread.color)
        return ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
    except Exception:
        pass
    return (0, 0, 0)


def render_colored(pattern, output_path: Path) -> bool:
    stitches = pattern.stitches
    if not stitches:
        return False

    min_x, min_y, max_x, max_y = pattern.bounds()
    width = int((max_x - min_x) * SCALE)
    height = int((max_y - min_y) * SCALE)
    if width <= 0 or height <= 0:
        return False

    img = Image.new("RGB", (width + 20, height + 20), "white")
    draw = ImageDraw.Draw(img)

    threads = getattr(pattern, "threadlist", None) or getattr(pattern, "threads", []) or []

    def color_for(idx: int):
        if not threads:
            return (0, 0, 0)
        return _thread_rgb(threads[idx % len(threads)])

    color_idx = 0
    current_color = color_for(color_idx)

    last = None
    drew_anything = False

    for x, y, cmd in stitches:
        if cmd == STITCH:
            if last is not None:
                draw.line(
                    (
                        (last[0] - min_x) * SCALE,
                        (last[1] - min_y) * SCALE,
                        (x - min_x) * SCALE,
                        (y - min_y) * SCALE,
                    ),
                    fill=current_color,
                    width=STROKE_WIDTH,
                )
                drew_anything = True
            last = (x, y)
        elif cmd in (COLOR_CHANGE, NEEDLE_SET):
            color_idx += 1
            current_color = color_for(color_idx)
            last = None  # break line continuity at color boundary
        else:
            # JUMP, TRIM, STOP, END — break line continuity
            last = None

    if not drew_anything:
        return False
    img.save(output_path)
    return True


def find_embroidery_files() -> list[Path]:
    """Walk SCAN_ROOT for embroidery source files. Prefer the converted PES
    when present (same logic as process_file.py)."""
    files: list[Path] = []
    seen_stems: set[str] = set()

    # First pass: converted PES files take precedence
    if CONVERT_DIR.exists():
        for p in CONVERT_DIR.iterdir():
            if p.suffix.lower() == ".pes" and p.is_file():
                files.append(p)
                seen_stems.add(p.stem)

    # Second pass: original embroidery files not already covered by conversion
    for p in SCAN_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        # Skip files under the converted dir (already added above)
        try:
            if CONVERT_DIR in p.parents:
                continue
        except Exception:
            pass
        if p.stem in seen_stems:
            continue
        files.append(p)
        seen_stems.add(p.stem)

    return sorted(files)


def main():
    files = find_embroidery_files()
    print(f"Found {len(files)} embroidery files to re-render in color.")
    print(f"Output: {OUT_DIR}")

    ok = 0
    fail = 0
    skipped = 0

    for i, f in enumerate(files, 1):
        out_path = OUT_DIR / f"{f.stem}.png"
        if out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            if i % 50 == 0:
                print(f"[{i}/{len(files)}] (cached) {f.name}")
            continue
        try:
            if f.suffix.lower() == ".vip":
                pattern = vip_reader.read_vip(str(f))
            else:
                pattern = pyembroidery.read(str(f))
            if not pattern or not pattern.stitches:
                fail += 1
                print(f"[{i}/{len(files)}] NO STITCHES: {f.name}")
                continue
            if render_colored(pattern, out_path):
                ok += 1
                print(f"[{i}/{len(files)}] {f.name}")
            else:
                fail += 1
                print(f"[{i}/{len(files)}] RENDER FAILED: {f.name}")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(files)}] ERROR: {f.name}  -- {e}")

    print()
    print(f"Rendered: {ok}   Skipped (cached): {skipped}   Failed: {fail}")
    print(f"Color renders are in: {OUT_DIR}")


if __name__ == "__main__":
    main()