#!/usr/bin/env python3
"""
run_gemini_flash_tags.py
========================
Strictly-Gemini-Flash tagger for Project Diamond.

Calls ONLY Gemini Flash -- no Pro, no Clarifai, no ONNX -- over the rendered
design images and writes the keyword tags into
Image_Recognition/image_recognition_results.csv (the "gemini_flash_tags"
column the results gallery reads).

It MERGES into that CSV: any existing columns (Gemini Pro tokens, Clarifai,
metadata, etc.) produced by run_compare_aws_azure_clarifai.py are preserved
untouched. By default it only tags images that don't already have Flash tags,
so re-runs are cheap and a demo "run all" won't re-bill 400+ images. Use
--force to re-tag everything.

All Flash logic (model, prompt, JSON schema, parsing, image prep) is imported
from run_compare_aws_azure_clarifai.py, so the two never drift apart.

Usage (project venv):
    venv/bin/python Image_Recognition/run_gemini_flash_tags.py
    venv/bin/python Image_Recognition/run_gemini_flash_tags.py --limit 5     # cheap test
    venv/bin/python Image_Recognition/run_gemini_flash_tags.py --force       # re-tag all
    venv/bin/python Image_Recognition/run_gemini_flash_tags.py --workers 6
"""
import argparse
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Image_Recognition"))
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# Reuse the exact Flash pipeline from the benchmark script (no duplication).
import run_compare_aws_azure_clarifai as R

FLASH_COLS = [
    "gemini_flash_model", "gemini_flash_tags", "gemini_flash_confidences",
    "gemini_flash_input_tokens", "gemini_flash_output_tokens",
    "gemini_flash_total_tokens", "gemini_flash_error",
]


def tag_one(img_path: Path, client) -> dict:
    """One Flash call -> dict of the gemini_flash_* columns for this image."""
    try:
        raw = img_path.read_bytes()
        prepared = R.prepare_image_bytes_for_apis(raw)
        tags, conf, in_tok, out_tok, total_tok = R.gemini_tags_with_usage(
            client, prepared, R.GEMINI_FLASH_MODEL
        )
        return {
            "gemini_flash_model": R.GEMINI_FLASH_MODEL,
            "gemini_flash_tags": tags,
            "gemini_flash_confidences": conf,
            "gemini_flash_input_tokens": in_tok,
            "gemini_flash_output_tokens": out_tok,
            "gemini_flash_total_tokens": total_tok,
            "gemini_flash_error": None,
        }
    except Exception as e:
        return {
            "gemini_flash_model": R.GEMINI_FLASH_MODEL,
            "gemini_flash_tags": None,
            "gemini_flash_confidences": None,
            "gemini_flash_input_tokens": None,
            "gemini_flash_output_tokens": None,
            "gemini_flash_total_tokens": None,
            "gemini_flash_error": str(e),
        }


def main():
    ap = argparse.ArgumentParser(description="Gemini Flash-only tagger (merges into the results CSV).")
    ap.add_argument("--limit", type=int, default=None, help="tag at most N images (cheap test)")
    ap.add_argument("--force", action="store_true", help="re-tag images that already have Flash tags")
    ap.add_argument("--workers", type=int, default=4, help="parallel Flash calls (default 4)")
    args = ap.parse_args()

    cfg = R.load_cfg()                      # requires GEMINI_API_KEY in .env
    client = R.gemini_client(cfg)
    print(f"Gemini Flash model: {R.GEMINI_FLASH_MODEL}")

    render_dir = R.resolve_render_dir()
    images = R.list_images(render_dir)
    if not images:
        print(f"No images found in: {render_dir}")
        return
    print(f"Found {len(images)} images.")

    out_csv = R.OUT_CSV
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        if "image_file" not in df.columns:
            df["image_file"] = pd.Series(dtype=object)
        df = df.set_index("image_file", drop=True)
        print(f"Merging into existing {out_csv.name} ({len(df)} rows).")
    else:
        df = pd.DataFrame().rename_axis("image_file")
        print(f"Creating new {out_csv.name}.")

    # ensure flash columns exist
    for c in FLASH_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    def needs(img: Path) -> bool:
        if args.force:
            return True
        name = img.name
        if name not in df.index:
            return True
        val = df.at[name, "gemini_flash_tags"]
        return pd.isna(val) or str(val).strip() == ""

    todo = [im for im in images if needs(im)]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"Tagging {len(todo)} images "
          f"({'forced' if args.force else 'new/untagged only'}); "
          f"{len(images) - len(todo)} skipped / already tagged.")

    if not todo:
        print("Nothing to tag. CSV already current.")
        return

    # add rows for any brand-new images (keep existing order, append new at end)
    existing = list(df.index)
    new_names = [im.name for im in todo if im.name not in df.index]
    if new_names:
        df = df.reindex(existing + new_names)

    t0 = time.time()
    done = 0
    in_tot = out_tot = tot_tot = n_tok = 0
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(tag_one, im, client): im for im in todo}
        for fut in as_completed(futs):
            im = futs[fut]
            res = fut.result()
            results[im.name] = res
            done += 1
            if res.get("gemini_flash_total_tokens") is not None:
                in_tot += res["gemini_flash_input_tokens"] or 0
                out_tot += res["gemini_flash_output_tokens"] or 0
                tot_tot += res["gemini_flash_total_tokens"] or 0
                n_tok += 1
            preview = (res.get("gemini_flash_tags") or res.get("gemini_flash_error") or "")[:60]
            avg = (time.time() - t0) / done
            eta = avg * (len(todo) - done) / 60
            print(f"[{done}/{len(todo)}] {im.name}  ({avg:.1f}s/img, ETA {eta:.1f}min)  {preview}", flush=True)

    # merge results back, preserving every other column
    for name, res in results.items():
        for c in FLASH_COLS:
            df.at[name, c] = res[c]

    df.reset_index().to_csv(out_csv, index=False)
    print(f"\nSaved Flash tags to: {out_csv}")

    # Re-apply durable collection conventions (AFF / 12702 / E-ASD) so manual
    # harmonizations survive a --force re-tag. See collection_rules.py.
    try:
        import collection_rules
        collection_rules.apply_to_csv(out_csv)
    except Exception as e:
        print(f"  (collection rules step skipped: {e})")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")

    if n_tok > 0:
        print(f"\n=== Gemini Flash token summary ({n_tok} images this run) ===")
        print(f"  Mean input  tokens / image: {in_tot / n_tok:.0f}")
        print(f"  Mean output tokens / image: {out_tot / n_tok:.0f}")
        print(f"  Mean total  tokens / image: {tot_tot / n_tok:.0f}")
        cost = (in_tot * 0.30 + out_tot * 2.50) / 1_000_000
        print(f"  Est. cost this run (@$0.30/M in, $2.50/M out): ${cost:.4f}")


if __name__ == "__main__":
    main()
