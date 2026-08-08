#!/usr/bin/env python3
"""
run_locate_and_tag.py
=====================
Web-path mirror of run_scan_and_tag.py.

The file path is a three-stage pipeline: the Rust scanner FINDS embroidery
files and (via process_file.py) renders + perceptual-hashes each one into a
deterministic UUID, then run_gemini_flash_tags.py TAGS the renders. Find ->
UUID -> Tag.

The web path was only doing stage one: web_locator.py LOCATES design images on
a site and writes a manifest, then stops. This script carries that manifest
through the SAME two remaining stages so the web intake mirrors the file intake:

    LOCATE  -> web_locator.py  (find candidate design images on a site)
    UUID    -> SAME normalize_render + pHash + namespace as process_file.py
    TAG     -> SAME Gemini Flash call as run_gemini_flash_tags.py

The UUID and tagging logic are IMPORTED from the existing modules (not copied),
so the web and file paths can never drift apart. Images are fetched in memory
and never written to disk (preserves the designer-copyright posture).

Usage (project venv):
    venv/bin/python run_locate_and_tag.py https://example-store.com
    venv/bin/python run_locate_and_tag.py example-store.com --limit 5   # cheap test
    venv/bin/python run_locate_and_tag.py example-store.com --max-pages 8 --workers 6
"""
import argparse
import csv
import io
import subprocess
import sys
import tempfile
import time
import uuid as uuidlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / "venv" / "bin" / "python"
sys.path.insert(0, str(ROOT / "Image_Recognition"))
sys.path.insert(0, str(ROOT / "worker_python"))
sys.path.insert(0, str(ROOT))

import requests
from PIL import Image
import imagehash

# SAME Gemini Flash pipeline the file path uses (model, prompt, schema, parsing).
import run_compare_aws_azure_clarifai as R
# SAME normalization + pHash + UUID namespace the file path uses.
import process_file as pf

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

WEB_CSV = ROOT / "Image_Recognition" / "web_recognition_results.csv"

COLS = [
    "image_url", "source_page", "alt", "domain",
    "uuid", "phash_render",
    "gemini_flash_model", "gemini_flash_tags", "gemini_flash_confidences",
    "gemini_flash_input_tokens", "gemini_flash_output_tokens",
    "gemini_flash_total_tokens", "gemini_flash_error",
]


def log(m=""):
    print(m, flush=True)


# --- STAGE 1: LOCATE -------------------------------------------------------
def locate(url, max_pages, timeout):
    """Run web_locator.py as the pure 'find' stage; stream it; return its rows."""
    tmp = Path(tempfile.gettempdir()) / "_locate_manifest.csv"
    if tmp.exists():
        tmp.unlink()
    cmd = [str(VENV_PY), str(ROOT / "web_locator.py"), url,
           "--max-pages", str(max_pages), "--out", str(tmp)]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        log("[locate] " + line.rstrip("\n"))
    proc.wait()
    rows = []
    if tmp.exists():
        with open(tmp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


# --- STAGE 2: UUID (identical logic to the file path) ----------------------
def uuid_for_image(raw_bytes):
    """Deterministic content UUID for an image, via the SAME steps process_file
    uses on a render: normalize_render -> pHash(size 16) -> uuid5(namespace)."""
    img = Image.open(io.BytesIO(raw_bytes))
    norm = pf.normalize_render(img)
    ph = imagehash.phash(norm, hash_size=pf.PHASH_SIZE)
    return str(uuidlib.uuid5(pf.DIAMOND_NAMESPACE, f"phash:{ph}")), str(ph)


# --- STAGES 2+3 per image: fetch -> UUID -> TAG ----------------------------
def process_image(row, client, timeout):
    url = row.get("image_url", "")
    out = dict(row)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        r.raise_for_status()
        raw = r.content                      # bytes held in memory only
        duuid, phash = uuid_for_image(raw)   # STAGE 2 (same as files)
        prepared = R.prepare_image_bytes_for_apis(raw)
        tags, conf, it, ot, tt = R.gemini_tags_with_usage(
            client, prepared, R.GEMINI_FLASH_MODEL)   # STAGE 3 (same as files)
        out.update({
            "uuid": duuid, "phash_render": phash,
            "gemini_flash_model": R.GEMINI_FLASH_MODEL,
            "gemini_flash_tags": tags, "gemini_flash_confidences": conf,
            "gemini_flash_input_tokens": it, "gemini_flash_output_tokens": ot,
            "gemini_flash_total_tokens": tt, "gemini_flash_error": None,
        })
    except Exception as e:
        out.update({
            "uuid": None, "phash_render": None,
            "gemini_flash_model": R.GEMINI_FLASH_MODEL,
            "gemini_flash_tags": None, "gemini_flash_confidences": None,
            "gemini_flash_input_tokens": None, "gemini_flash_output_tokens": None,
            "gemini_flash_total_tokens": None, "gemini_flash_error": str(e),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Web path: locate -> UUID -> tag (mirror of scan/uuid/tag).")
    ap.add_argument("url", help="site URL (https:// optional)")
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="UUID/tag at most N images (cheap test)")
    ap.add_argument("--workers", type=int, default=4, help="parallel fetch+tag (default 4)")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="results CSV path")
    a = ap.parse_args()

    log("=" * 56)
    log("Web pipeline:  LOCATE -> UUID -> TAG   (mirror of scan/uuid/tag)")
    log("=" * 56)

    # STAGE 1
    rows = locate(a.url, a.max_pages, a.timeout)
    if a.limit is not None:
        rows = rows[: a.limit]
    if not rows:
        log("No candidate images located; nothing to UUID/tag.")
        return
    log(f"Found {len(rows)} candidate images")   # dashboard reads this as the total

    # STAGES 2 + 3
    cfg = R.load_cfg()                            # needs GEMINI_API_KEY in .env
    client = R.gemini_client(cfg)
    log(f"UUID + tag with {R.GEMINI_FLASH_MODEL}  (images held in memory, never stored)")

    results = []
    done = 0
    it = ot = tt = ntok = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        futs = {pool.submit(process_image, row, client, a.timeout): row for row in rows}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            if res.get("gemini_flash_total_tokens") is not None:
                it += res["gemini_flash_input_tokens"] or 0
                ot += res["gemini_flash_output_tokens"] or 0
                tt += res["gemini_flash_total_tokens"] or 0
                ntok += 1
            u = (res.get("uuid") or "--------")[:8]
            prev = (res.get("gemini_flash_tags") or res.get("gemini_flash_error") or "")[:48]
            log(f"  [{done}/{len(rows)}] {res.get('image_url','')}  {u}  {prev}")

    out_path = Path(a.out) if a.out else WEB_CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for res in results:
            w.writerow({k: res.get(k) for k in COLS})

    log("-" * 56)
    log(f"Wrote {len(results)} rows -> {out_path}")
    if ntok:
        cost = (it * 0.30 + ot * 2.50) / 1_000_000
        log(f"Flash tokens: {tt} total over {ntok} images; est cost ${cost:.4f} "
            f"(@$0.30/M in, $2.50/M out)")
    log(f"Runtime: {(time.time() - t0) / 60:.1f} min")
    log("Done.")


if __name__ == "__main__":
    main()
