#!/usr/bin/env python3
"""
run_diamond_demo.py
===================
End-to-end demo of the Diamond cataloging pipeline. It reuses the EXACT
Worker_python pipeline (process_file.py + the native VIP reader), so results
match the real scanner.

For each embroidery design file under Metadata_Capture it:
  - renders the design (reusing an existing render if one is already on disk),
  - extracts metadata (size, stitch count, colours),
  - computes the content-derived UUID (pHash, 7-bit tolerance),
then groups designs by visual identity and attaches the Gemini Flash tags
already produced by the tagging service
(Image_Recognition/image_recognition_results.csv, "gemini_flash_tags" column).

It writes a self-contained, tabbed results gallery:
    _diamond_demo/index.html   -- tabs: UUID & cross-format / Metadata / Image tagging
    _diamond_demo/metadata.csv -- one row per file
    _diamond_demo/renders/     -- one small JPEG per design group (format-copies share it)
    _diamond_demo.zip          -- the whole thing, ready to share

Run with the project venv:
    /Users/amans/Desktop/Mark-VLM/venv/bin/python run_diamond_demo.py
    /Users/amans/Desktop/Mark-VLM/venv/bin/python run_diamond_demo.py Test_Set_1   # quick partial

Image tagging is strictly Gemini Flash, read from the tagging service's CSV.
This script never calls a tagging API itself; refresh the CSV with the tagging
pipeline first if you want the newest tags.
"""

from __future__ import annotations

import base64
import csv
import html
import os
import re
import shutil
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 300_000_000
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --- locate the project + reuse the real worker -------------------------------
PROJECT_ROOT = Path("/Users/amans/Desktop/Mark-VLM")
WORKER_DIR = PROJECT_ROOT / "Worker_python"
SCAN_ROOT = PROJECT_ROOT / "Metadata_Capture"
OUT_DIR = PROJECT_ROOT / "_diamond_demo"
RENDER_OUT = OUT_DIR / "renders"
FLASH_CSV = PROJECT_ROOT / "Image_Recognition" / "image_recognition_results.csv"

sys.path.insert(0, str(WORKER_DIR))
try:
    import process_file as W
except Exception as e:  # pragma: no cover
    sys.exit(
        f"Could not import the worker ({e}).\n"
        f"Run this with the project venv:\n"
        f"  {PROJECT_ROOT}/venv/bin/python {Path(__file__).name}"
    )

EXTS = {".pes", ".dst", ".exp", ".jef", ".vp3", ".hus", ".vip", ".art", ".xxx"}
SKIP_DIRS = {
    "rendered_images", "rendered_images_2", "rendered_images_3", "rendered_images_4",
    "rendered_images_5", "rendered_images_6", "rendered_images_color",
    "rendered_images_color_BACKUP_20260614", "vip+art_converted", "__pycache__",
    ".ipynb_checkpoints", ".venv", "venv", "tf-venv", ".git", "_diamond_demo",
}
TOLERANCE_BITS = 7
TARGET_RENDER_PX = 3000   # render the design this big for an accurate pHash (then discard)
DISPLAY_PX = 640          # long edge of the small JPEG shipped in the gallery + zip
DISPLAY_QUALITY = 80      # JPEG quality for the shipped display images


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _make_display(src_png: Path, dst_jpg: Path, max_dim: int, quality: int) -> None:
    """Write a small JPEG copy of a render for the gallery/zip (one per design group)."""
    im = Image.open(src_png)
    im.load()
    w, h = im.size
    if max(w, h) > max_dim:
        sc = max_dim / float(max(w, h))
        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.Resampling.LANCZOS)
    im.convert("RGB").save(dst_jpg, format="JPEG", quality=quality, optimize=True)
    im.close()



def find_files(root: Path):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith("rendered_images")]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() in EXTS:
                out.append(p)
    return sorted(out)


def hamming_hex(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def cluster_by_identity(rows):
    """Union files whose pHashes are within TOLERANCE_BITS of each other."""
    ok = [r for r in rows if r["phash"]]
    n = len(ok)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    ints = [int(r["phash"], 16) for r in ok]
    for i in range(n):
        ai = ints[i]
        for j in range(i + 1, n):
            if hamming_hex(ai, ints[j]) <= TOLERANCE_BITS:
                union(i, j)

    groups = defaultdict(list)
    for idx, r in enumerate(ok):
        groups[find(idx)].append(r)
    return list(groups.values())


# --- Gemini Flash tags from the tagging service CSV ---------------------------
def load_flash_tags() -> dict:
    """Map design stem -> list of Gemini Flash keywords.

    Source: Image_Recognition/image_recognition_results.csv, produced by the
    tagging pipeline. Keyed on the render stem ("image_file" without extension),
    which equals the design's file stem -- so tags fan out to every format copy
    of that design. Falls back to the Pro column only if a row has no Flash tags.
    """
    mapping = {}
    if not FLASH_CSV.exists():
        return mapping
    try:
        with open(FLASH_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = (row.get("image_file") or "").strip()
                if not img:
                    continue
                stem = Path(img).stem
                raw = (row.get("gemini_flash_tags") or "").strip()
                if not raw:
                    raw = (row.get("gemini_tags") or "").strip()
                tags = [t.strip() for t in raw.split(",") if t.strip()]
                if tags:
                    mapping[stem] = tags
    except Exception as e:
        print(f"  (could not read Flash tags from {FLASH_CSV.name}: {e})")
    return mapping


def load_flash_token_usage() -> dict:
    """Aggregate Gemini Flash token usage from the tags CSV (latest tagging run).

    Sums the per-design gemini_flash_input/output/total_tokens columns and
    estimates cost at the Flash rates used by the benchmark ($0.30/M input,
    $2.50/M output). Returns {} when no usage is recorded.
    """
    IN_RATE, OUT_RATE = 0.30, 2.50  # USD per 1M tokens, Gemini 2.5 Flash
    if not FLASH_CSV.exists():
        return {}
    def _int(v):
        try:
            return int(float(v))
        except Exception:
            return 0
    in_tot = out_tot = tot_tot = n = 0
    model = "gemini-2.5-flash"
    try:
        with open(FLASH_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = _int(row.get("gemini_flash_total_tokens"))
                if t <= 0:
                    continue
                n += 1
                in_tot += _int(row.get("gemini_flash_input_tokens"))
                out_tot += _int(row.get("gemini_flash_output_tokens"))
                tot_tot += t
                m = (row.get("gemini_flash_model") or "").strip()
                if m:
                    model = m
    except Exception as e:
        print(f"  (could not read token usage from {FLASH_CSV.name}: {e})")
        return {}
    if n == 0:
        return {}
    cost = (in_tot * IN_RATE + out_tot * OUT_RATE) / 1_000_000
    return {"model": model, "n": n, "in_tok": in_tot, "out_tok": out_tot,
            "total_tok": tot_tot, "avg_tok": tot_tot / n, "cost": cost,
            "in_rate": IN_RATE, "out_rate": OUT_RATE}


GROUND_TRUTH = {}


def load_ground_truth():
    """stem -> Mark's PST ground-truth tag string, from his 260628 analysis workbook
    (searched in Downloads and Dropbox). Empty dict if not found."""
    import glob
    cands = (glob.glob(str(Path.home() / "Downloads" / "*260628*.xlsm"))
             + glob.glob(str(Path.home() / "Library" / "CloudStorage" / "Dropbox" / "**" / "*260628*.xlsm"),
                         recursive=True))
    gt = {}
    if not cands:
        return gt
    try:
        from openpyxl import load_workbook
        ws = load_workbook(cands[0], read_only=True, data_only=True)["Combined Test Sets"]
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i < 2:
                continue
            row = list(row) + [None] * 30
            name, pst = row[4], row[18]
            if name and pst:
                gt[str(name).rsplit(".", 1)[0].strip()] = str(pst)
    except Exception as e:
        print(f"  (could not read ground truth: {e})")
    return gt


def _match(tags, pst):
    """(recall of PST tags by `tags`, list of missed PST tags). (None, []) if no GT."""
    import re as _re
    if not pst:
        return None, []
    P = [t.strip() for t in str(pst).split(",") if t.strip()]
    if not P:
        return None, []
    def _n(t):
        return _re.sub(r"(es|s)$", "", t.strip().lower())
    Fn = {_n(t) for t in tags}
    miss = [p for p in P if _n(p) not in Fn]
    return (len(P) - len(miss)) / len(P), miss


def _match_badge_html(sc):
    if sc is None:
        return ""
    if sc >= 0.999: bg, fg = "#fff4cc", "#8a6d00"
    elif sc >= 0.75: bg, fg = "#e7f6ec", "#1c7a3e"
    elif sc >= 0.5: bg, fg = "#fff0e0", "#a5590a"
    else: bg, fg = "#fdeaea", "#b02a2a"
    return (" <span style='font:600 10px ui-monospace,Menlo,monospace;background:%s;"
            "color:%s;border-radius:999px;padding:1px 7px;margin-left:6px'>%.0f%%</span>"
            % (bg, fg, sc * 100))


def build_excel(rows, groups, out_dir, token_usage=None):
    """Write per-tab and combined .xlsx files into out_dir/downloads/ (real Excel)."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
    except Exception as e:
        print(f"  (openpyxl unavailable, skipping Excel export: {e})")
        return []

    dl = out_dir / "downloads"
    dl.mkdir(parents=True, exist_ok=True)
    HFILL = PatternFill("solid", fgColor="203090")
    HFONT = Font(bold=True, color="FFFFFF")

    def _style(ws, cols, widths):
        for ci, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(ci)].width = w
        for ci in range(1, len(cols) + 1):
            c = ws.cell(row=1, column=ci)
            c.fill = HFILL; c.font = HFONT; c.alignment = Alignment(vertical="center")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}1"

    def _fill_meta(ws):
        ws.title = "Metadata"
        cols = ["Test set", "File", "Format", "Width (in)", "Height (in)", "Stitches", "Colors", "UUID"]
        ws.append(cols)
        for r in rows:
            ws.append([r["test_set"], r["file"], r["fmt"], r["width"], r["height"],
                       r["stitches"], r["colors"], r["uuid"]])
        _style(ws, cols, [13, 26, 8, 11, 11, 10, 8, 38])

    def _fill_uuid(ws):
        ws.title = "UUID & cross-format"
        cols = ["UUID", "Distinct formats", "Formats", "Files", "Cross-format", "Member files"]
        ws.append(cols)
        for g in sorted(groups, key=lambda g: (-len({r["fmt"] for r in g}), -len(g))):
            fmts = sorted({r["fmt"] for r in g})
            ws.append([g[0]["uuid"], len(fmts), ", ".join(fmts), len(g),
                       "Yes" if len(fmts) > 1 else "No",
                       "; ".join(sorted(r["file"] for r in g))])
        _style(ws, cols, [38, 15, 22, 7, 13, 64])

    def _rep(g):
        return next((r for r in sorted(g, key=lambda r: (r["fmt"], r["file"])) if r.get("render")), g[0])

    def _fill_tags(ws):
        ws.title = "Image tagging"
        cols = ["Design", "UUID", "Formats", "Tags", "Match %", "Missed (ground-truth tags)"]
        ws.append(cols)
        def taglist_of(g):
            for r in g:
                if r.get("tags"):
                    return r["tags"]
            return []
        for g in sorted(groups, key=lambda g: (0 if any(r.get("tags") for r in g) else 1)):
            tg = taglist_of(g)
            sc, miss = _match(tg, GROUND_TRUTH.get(Path(_rep(g)["file"]).stem))
            ws.append([_rep(g)["file"], g[0]["uuid"], ", ".join(sorted({r["fmt"] for r in g})),
                       ", ".join(tg), sc, ", ".join(miss)])
            if sc is not None:
                ws.cell(row=ws.max_row, column=5).number_format = "0%"
        _style(ws, cols, [26, 38, 22, 70, 9, 40])

    def _fill_tokens(ws):
        ws.title = "Token usage"
        tu = token_usage
        data = [
            ["Gemini Flash tagging - token usage (latest run)", ""],
            ["Model", tu["model"]],
            ["Designs tagged", tu["n"]],
            ["Input tokens", tu["in_tok"]],
            ["Output tokens", tu["out_tok"]],
            ["Total tokens", tu["total_tok"]],
            ["Avg tokens / design", round(tu["avg_tok"], 1)],
            [f"Est. cost (@ ${tu['in_rate']:.2f}/M in, ${tu['out_rate']:.2f}/M out)", round(tu["cost"], 4)],
            ["Avg cost / design (USD)", round(tu["cost"] / tu["n"], 6) if tu["n"] else 0],
        ]
        for d in data:
            ws.append(d)
        ws.column_dimensions["A"].width = 48
        ws.column_dimensions["B"].width = 22
        ws.merge_cells("A1:B1")
        h = ws.cell(row=1, column=1); h.fill = HFILL; h.font = HFONT
        ws.cell(row=1, column=2).fill = HFILL

    written = []
    wb = Workbook(); _fill_meta(wb.active)
    wb.save(dl / "Project_Diamond_Metadata.xlsx"); written.append("Project_Diamond_Metadata.xlsx")
    wb = Workbook(); _fill_uuid(wb.active)
    wb.save(dl / "Project_Diamond_UUID.xlsx"); written.append("Project_Diamond_UUID.xlsx")
    wb = Workbook(); _fill_tags(wb.active)
    if token_usage:
        _fill_tokens(wb.create_sheet())
    wb.save(dl / "Project_Diamond_Tags.xlsx"); written.append("Project_Diamond_Tags.xlsx")
    wb = Workbook(); _fill_meta(wb.active); _fill_uuid(wb.create_sheet()); _fill_tags(wb.create_sheet())
    if token_usage:
        _fill_tokens(wb.create_sheet())
    wb.save(dl / "Project_Diamond_Results_All.xlsx"); written.append("Project_Diamond_Results_All.xlsx")
    print(f"  Excel: wrote {len(written)} files to {dl}")
    return written


def main():
    global SCAN_ROOT, GROUND_TRUTH
    t0 = time.time()
    full = "--full" in sys.argv   # render at production scale=4 (matches 449 baseline; slow, same small zip)
    if full:
        print("--full: rendering at production scale=4 -- highest-fidelity grouping, slower build")

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sub = SCAN_ROOT / sys.argv[1]
        if sub.is_dir():
            SCAN_ROOT = sub
            print(f"(limiting scan to {sub})")
        else:
            print(f"warning: {sub} not found; scanning all of Metadata_Capture")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(RENDER_OUT, ignore_errors=True)   # start clean: only this run's display images
    RENDER_OUT.mkdir(parents=True, exist_ok=True)

    flash_map = load_flash_tags()
    if flash_map:
        print(f"Loaded Gemini Flash tags for {len(flash_map)} designs from {FLASH_CSV.name}")
    else:
        print(f"No Flash tags found ({FLASH_CSV.name} missing/empty) -- tagging tab will be empty")

    GROUND_TRUTH = load_ground_truth()
    if GROUND_TRUTH:
        print(f"Loaded Mark's ground truth for {len(GROUND_TRUTH)} designs")
    files = find_files(SCAN_ROOT)
    print(f"Found {len(files)} design files under {SCAN_ROOT}")

    rows = []
    unreadable = []
    reused = 0
    seen_uuid = set()
    for i, path in enumerate(files, 1):
        ext = path.suffix.lower().lstrip(".")
        test_set = path.parent.name
        tmp_png = RENDER_OUT / (safe_name(f"{test_set}__{path.stem}__{ext}") + ".phash.png")
        tags = flash_map.get(path.stem, [])
        row = dict(file=path.name, path=str(path), fmt=ext.upper(), test_set=test_set,
                   width=None, height=None, stitches=None, colors=None,
                   uuid=None, phash=None, render=None, error=None, tags=tags)
        try:
            pattern, used, converted, err = W.load_pattern(path)
            if not pattern:
                row["error"] = err or "could not read"
                unreadable.append((path, row["error"]))
            else:
                w, h, st, col = W.extract_metadata(pattern)
                row.update(width=round(w, 2), height=round(h, 2),
                           stitches=int(st), colors=(int(col) if col else None))
                mnx, mny, mxx, mxy = pattern.bounds()
                extent = max(mxx - mnx, mxy - mny, 1)
                rscale = 4.0 if full else min(4.0, TARGET_RENDER_PX / float(extent))  # --full = production scale
                ok_render = W.render_pattern(pattern, tmp_png, scale=rscale)
                if ok_render and tmp_png.exists():
                    duuid, phash = W.uuid_from_render_png(tmp_png)
                    disp_name = f"{duuid}.jpg"   # one display image per design group
                    if duuid not in seen_uuid:
                        _make_display(tmp_png, RENDER_OUT / disp_name, DISPLAY_PX, DISPLAY_QUALITY)
                        seen_uuid.add(duuid)
                    else:
                        reused += 1
                    row.update(uuid=duuid, phash=phash, render=f"renders/{disp_name}")
                else:
                    row["error"] = "render failed (no stitches)"
                    unreadable.append((path, row["error"]))
                tmp_png.unlink(missing_ok=True)   # keep only the small display JPEGs
        except Exception as e:
            row["error"] = str(e)
            unreadable.append((path, str(e)))
        rows.append(row)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} processed...", flush=True)

    # --- cluster --------------------------------------------------------------
    groups = cluster_by_identity(rows)
    groups.sort(key=lambda g: (-len(g), g[0]["file"]))
    vip_groups = [g for g in groups if any(r["fmt"] == "VIP" for r in g)]
    multi_groups = [g for g in groups if len(g) > 1]
    readable = [r for r in rows if r["uuid"]]
    formats_present = sorted({r["fmt"] for r in readable})
    tagged_designs = sum(1 for g in groups if any(r["tags"] for r in g))

    # --- CSV ------------------------------------------------------------------
    with open(OUT_DIR / "metadata.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["file", "format", "test_set", "width_in", "height_in",
                       "stitches", "colors", "uuid", "phash", "flash_tags", "error", "path"])
        for r in rows:
            wcsv.writerow([r["file"], r["fmt"], r["test_set"], r["width"], r["height"],
                           r["stitches"], r["colors"], r["uuid"], r["phash"],
                           "; ".join(r.get("tags") or []), r["error"], r["path"]])

    # --- Excel downloads (per-tab + combined) ---------------------------------
    token_usage = load_flash_token_usage()
    build_excel(rows, groups, OUT_DIR, token_usage)

    # --- HTML -----------------------------------------------------------------
    write_html(OUT_DIR / "index.html", rows, groups, vip_groups, multi_groups,
               readable, unreadable, formats_present, tagged_designs, time.time() - t0,
               token_usage)

    # --- web locator tab (best-effort) ---------------------------------------
    # If run_locate_and_tag.py has produced web results, add/refresh the
    # "Web locator" tab in the gallery BEFORE zipping, so the tab is durable
    # across rebuilds and travels in the shareable zip. Never breaks the build.
    try:
        import subprocess as _sp
        _r = _sp.run([sys.executable, str(PROJECT_ROOT / "add_web_tab.py")],
                     capture_output=True, text=True)
        _msg = (_r.stdout or _r.stderr or "").strip().splitlines()
        if _r.returncode == 0:
            print(_r.stdout.strip())
        else:
            print(f"(web tab skipped: {_msg[-1] if _msg else 'no web results yet'})")
    except Exception as _e:
        print(f"(web tab step skipped: {_e})")

    # --- zip for sharing ------------------------------------------------------
    zip_base = str(PROJECT_ROOT / "_diamond_demo")
    shutil.make_archive(zip_base, "zip", OUT_DIR)

    print()
    print("=" * 62)
    print(f"Files scanned:           {len(files)}")
    print(f"Read + rendered:         {len(readable)}   ({len(seen_uuid)} design images, {reused} format-copies shared)")
    print(f"Could not read:          {len(unreadable)}  (ART needs Bernina ARTlink)")
    print(f"Distinct designs:        {len(groups)}")
    print(f"Cross-format groups:     {len(multi_groups)}")
    print(f"Groups including VIP:    {len(vip_groups)}")
    print(f"Designs with Flash tags: {tagged_designs}")
    print("=" * 62)
    print(f"Open : {OUT_DIR / 'index.html'}")
    print(f"Send : {zip_base}.zip")
    print(f"Done in {time.time() - t0:0.1f}s")


# --- HTML rendering -----------------------------------------------------------
def _card(r):
    label = html.escape(f'{r["fmt"]}  -  {r["file"]}')
    sub = []
    if r["width"] is not None:
        sub.append(f'{r["width"]}"x{r["height"]}"')
    if r["stitches"] is not None:
        sub.append(f'{r["stitches"]:,} st')
    if r["colors"]:
        sub.append(f'{r["colors"]} col')
    sub = html.escape("  -  ".join(sub))
    vip = " vip" if r["fmt"] == "VIP" else ""
    if r["render"]:
        img = f'<img src="{html.escape(r["render"])}" loading="lazy">'
    else:
        img = '<div class="noimg">no render</div>'
    return (f'<figure class="card{vip}">{img}'
            f'<figcaption><b>{label}</b><span>{sub}</span></figcaption></figure>')


def _chips(tags, limit=18):
    if not tags:
        return ""
    out = "".join(f"<span class='chip'>{html.escape(t)}</span>" for t in tags[:limit])
    return f"<div class='tags'>{out}</div>"


def _group_block(g, has_vip):
    cls = "group hasvip" if has_vip else "group"
    uid = html.escape(g[0]["uuid"] or "")
    fmts = html.escape(", ".join(sorted({r["fmt"] for r in g})))
    badge = "<span class='badge'>VIP</span> " if has_vip else ""
    head = (f"<div class='ghead'><div>{badge}<b>{fmts}</b> "
            f"<span style='color:#6b7280'>&nbsp;{len(g)} files</span></div>"
            f"<div class='id'>{uid}</div></div>")
    cards = "".join(_card(r) for r in sorted(g, key=lambda r: r["fmt"]))
    return f"<div class='{cls}'>{head}<div class='row'>{cards}</div></div>"


def _design_tag_block(g):
    """One distinct design: representative render + its Gemini Flash tags."""
    rep = next((r for r in sorted(g, key=lambda r: (r["fmt"], r["file"])) if r.get("render")), g[0])
    tags = []
    for r in g:
        if r.get("tags"):
            tags = r["tags"]
            break
    has_vip = any(r["fmt"] == "VIP" for r in g)
    cls = "design hasvip" if has_vip else "design"
    raw_fmts = ", ".join(sorted({r["fmt"] for r in g}))
    fmts = html.escape(raw_fmts)
    if rep.get("render"):
        img = f'<img src="{html.escape(rep["render"])}" loading="lazy">'
    else:
        img = '<div class="noimg">no render</div>'
    name = html.escape(rep["file"])
    _sc, _ = _match(tags, GROUND_TRUTH.get(Path(rep["file"]).stem))
    mbadge = _match_badge_html(_sc)
    body = _chips(tags, limit=40) if tags else "<div class='notags'>no Flash tags yet</div>"
    blob = html.escape((" ".join(tags) + " " + rep["file"] + " " + raw_fmts).lower(), quote=True)
    duuid = html.escape(rep.get("uuid") or "", quote=True)
    dtags = html.escape(", ".join(tags), quote=True)
    dnamea = html.escape(rep["file"], quote=True)
    dfmtsa = html.escape(raw_fmts, quote=True)
    return (f"<div class='{cls}' data-search='{blob}'"
            f" data-name='{dnamea}' data-fmts='{dfmtsa}' data-uuid='{duuid}' data-tags='{dtags}'>"
            f"<div class='dimg'>{img}</div>"
            f"<div class='dbody'><div class='dname'>{name}{mbadge}</div>"
            f"<div class='dfmts'>{fmts}</div>{body}</div></div>")


def write_html(path, rows, groups, vip_groups, multi_groups, readable, unreadable,
               formats_present, tagged_designs, elapsed, token_usage=None):
    P = []
    A = P.append
    logo_uri = ""
    try:
        _lp = PROJECT_ROOT / "assets" / "ps_logo.png"
        if _lp.exists():
            logo_uri = "data:image/png;base64," + base64.b64encode(_lp.read_bytes()).decode()
    except Exception:
        logo_uri = ""
    A("<!doctype html><html><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    A("<title>Project Diamond - results</title>")
    A("""<style>
      :root{--ink:#1d2330;--mut:#6b7280;--line:#e6e8ee;--vip:#b8004a;--vipbg:#fff0f5;--bg:#fafbfc;--accent:#2b7fa8;}
      *{box-sizing:border-box}
      body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
      header{padding:24px 32px 0;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:5}
      h1{margin:0 0 4px;font-size:21px}
      header::before{content:"";position:absolute;left:0;right:0;top:0;height:3px;background:linear-gradient(90deg,#203090,#e01020)}
      .brandrow{display:flex;align-items:center;gap:14px;padding-top:4px;margin-bottom:2px}
      .brandrow .logo{height:34px;width:auto;display:block}
      .brandrow .bdiv{width:1px;height:28px;background:var(--line)}
      .brandrow .btitles{display:flex;flex-direction:column;gap:1px;line-height:1.05}
      .brandrow .eyebrow{font:600 10.5px/1 ui-monospace,Menlo,monospace;letter-spacing:.2em;text-transform:uppercase;color:#203090}
      .brandrow h1{margin:0;font-size:20px}
      .dl{position:relative;margin-left:auto}
      .dlbtn{font:600 12.5px/1 inherit;color:#fff;background:var(--accent);border:1px solid var(--accent);padding:10px 15px;border-radius:9px;cursor:pointer;display:inline-flex;align-items:center;gap:7px;transition:background .14s,border-color .14s,box-shadow .14s}
      .dlbtn:hover{background:#246b8e;border-color:#246b8e;box-shadow:0 3px 12px rgba(43,127,168,.25)}
      .dlbtn .caret{font-size:10px;opacity:.85}
      .dlmenu{position:absolute;right:0;top:calc(100% + 7px);min-width:232px;background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 12px 32px rgba(20,30,55,.16);padding:6px;display:none;z-index:30}
      .dlmenu.open{display:block}
      .dlmenu a{display:block;padding:9px 11px;border-radius:7px;font-size:13px;color:var(--ink);text-decoration:none}
      .dlmenu a:hover{background:#eaf3f7}
      .dlmenu .dlall{font-weight:650;color:var(--accent)}
      .dlsep{height:1px;background:var(--line);margin:5px 6px}
      .sub{color:var(--mut);font-size:13px}
      .stats{display:flex;flex-wrap:wrap;gap:22px;margin:16px 0 18px}
      .stat b{display:block;font-size:22px}
      .stat span{color:var(--mut);font-size:12px}
      .tabs{display:flex;gap:4px}
      .tab-btn{appearance:none;border:1px solid var(--line);border-bottom:none;background:var(--bg);color:var(--mut);font:600 13px/1 inherit;padding:11px 16px;border-radius:9px 9px 0 0;cursor:pointer}
      .tab-btn:hover{color:var(--ink)}
      .tab-btn.active{background:#fff;color:var(--ink);box-shadow:0 -2px 0 var(--accent) inset}
      .tabpane{display:none;padding:24px 32px 48px}
      .tabpane.active{display:block}
      h2{font-size:16px;margin:0 0 4px}
      .vipdot{color:var(--vip)}
      .note{color:var(--mut);margin:0 0 18px;max-width:78ch}
      .group{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:14px 0;background:#fff}
      .group.hasvip{border-color:var(--vip);background:var(--vipbg)}
      .ghead{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:10px}
      .ghead .id{font:12px ui-monospace,Menlo,monospace;color:var(--mut)}
      .badge{display:inline-block;font-size:11px;font-weight:700;color:#fff;background:var(--vip);border-radius:999px;padding:2px 9px}
      .row{display:flex;flex-wrap:wrap;gap:14px}
      .card{margin:0;width:150px}
      .card img,.card .noimg{width:150px;height:150px;object-fit:contain;background:#fff;border:1px solid var(--line);border-radius:8px}
      .card .noimg{display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:12px}
      .card.vip img{border-color:var(--vip);border-width:2px}
      figcaption{font-size:12px;margin-top:6px;line-height:1.35}
      figcaption b{display:block;font-weight:600}
      figcaption span{color:var(--mut)}
      .tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px}
      .chip{display:inline-block;font-size:11px;background:#eef4f8;border:1px solid #d6e4ec;border-radius:999px;padding:2px 9px;color:#27566b}
      table{border-collapse:collapse;width:100%;font-size:13px}
      th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
      th{color:var(--mut);font-weight:600}
      tr.vip td{background:var(--vipbg)}
      td.u{font:11px ui-monospace,Menlo,monospace;color:var(--mut)}
      .fmtpill{display:inline-block;font-size:11px;background:#eef1f6;border-radius:6px;padding:1px 7px;margin:2px}
      .designs{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}
      .design{display:flex;gap:13px;border:1px solid var(--line);border-radius:12px;padding:12px;background:#fff}
      .design.hasvip{border-color:var(--vip);background:var(--vipbg)}
      .dimg img,.dimg .noimg{width:104px;height:104px;object-fit:contain;background:#fff;border:1px solid var(--line);border-radius:8px;flex:none}
      .dimg .noimg{display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:11px}
      .dbody{min-width:0}
      .dname{font-weight:600;font-size:13px;word-break:break-word}
      .dfmts{font:11px ui-monospace,Menlo,monospace;color:var(--mut);margin:2px 0 7px}
      .notags{color:var(--mut);font-size:12px;font-style:italic}
      .searchbar{display:flex;align-items:center;gap:12px;margin:0 0 16px}
      #tagsearch{flex:1;max-width:520px;font:14px inherit;padding:9px 13px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink)}
      #tagsearch:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px rgba(43,127,168,.12)}
      .searchcount{color:var(--mut);font-size:12px;white-space:nowrap}
      .tokpanel{border:1px solid var(--line);border-radius:12px;background:#fff;padding:14px 16px;margin:4px 0 18px}
      .tokhead{font-weight:650;font-size:13px;color:var(--ink);margin-bottom:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
      .tokmodel{font:600 10.5px/1 ui-monospace,Menlo,monospace;letter-spacing:.04em;color:#246b8e;background:#eaf3f7;border:1px solid #d4e6ee;border-radius:999px;padding:3px 8px}
      .toktiles{display:flex;flex-wrap:wrap;gap:14px}
      .toktile{min-width:108px;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:9px 13px}
      .toktile b{display:block;font-size:18px;color:#203090}
      .toktile span{font-size:11px;color:var(--mut)}
      .toknote{margin-top:10px;font-size:11.5px;color:var(--mut)}
      .ddlg{position:fixed;inset:0;z-index:200;display:none;align-items:center;justify-content:center;background:rgba(17,22,35,.72);padding:24px}
      .ddlg.open{display:flex}
      .ddbox{background:#fff;border-radius:16px;max-width:880px;width:100%;max-height:88vh;overflow:auto;position:relative;display:grid;grid-template-columns:minmax(0,360px) 1fr;box-shadow:0 24px 70px rgba(0,0,0,.42)}
      .ddimgwrap{background:#f6f7f9;display:flex;align-items:center;justify-content:center;padding:22px;border-right:1px solid var(--line)}
      #ddimg{max-width:100%;max-height:62vh;object-fit:contain;background:#fff;border:1px solid var(--line);border-radius:8px}
      .ddinfo{padding:24px 26px;min-width:0}
      .ddinfo h3{margin:0 0 14px;font-size:16px;word-break:break-word;color:var(--ink)}
      #ddmeta{display:grid;grid-template-columns:auto 1fr;gap:6px 16px;margin:0;font-size:13px}
      #ddmeta dt{color:var(--mut)}
      #ddmeta dd{margin:0;color:var(--ink)}
      .ddsec{font:11px ui-monospace,Menlo,monospace;letter-spacing:.13em;text-transform:uppercase;color:var(--mut);margin:14px 0 6px}
      .dduuid{font:12px ui-monospace,Menlo,monospace;color:var(--ink);word-break:break-all;background:#f6f7f9;padding:8px 10px;border-radius:7px}
      .ddtags{display:flex;flex-wrap:wrap;gap:6px}
      .ddnone{color:var(--mut);font-size:12px;font-style:italic}
      .ddx{position:absolute;top:10px;right:12px;border:none;background:rgba(0,0,0,.06);width:30px;height:30px;border-radius:50%;font-size:21px;line-height:1;cursor:pointer;color:#333;z-index:2}
      .ddx:hover{background:rgba(0,0,0,.12)}
      #tab-meta tbody tr{cursor:pointer}
      #tab-meta tbody tr:hover{background:#f3f6f9}
      #tab-tags .dimg img{cursor:zoom-in}
      @media (max-width:680px){.ddbox{grid-template-columns:1fr}.ddimgwrap{border-right:none;border-bottom:1px solid var(--line)}}
    </style>""")
    A("</head><body>")

    A("<header>")
    DL = ("<div class='dl'><button class='dlbtn' id='dlbtn' type='button' aria-haspopup='true' aria-expanded='false'>"
          "Download Excel <span class='caret'>&#9662;</span></button>"
          "<div class='dlmenu' id='dlmenu' role='menu'>"
          "<a role='menuitem' href='downloads/Project_Diamond_Metadata.xlsx' download>Metadata</a>"
          "<a role='menuitem' href='downloads/Project_Diamond_UUID.xlsx' download>UUID &amp; cross-format</a>"
          "<a role='menuitem' href='downloads/Project_Diamond_Tags.xlsx' download>Image tagging</a>"
          "<div class='dlsep'></div>"
          "<a role='menuitem' class='dlall' href='downloads/Project_Diamond_Results_All.xlsx' download>Everything (all tabs)</a>"
          "</div></div>")
    if logo_uri:
        A(f"<div class='brandrow'><img class='logo' src='{logo_uri}' alt='Patrick Sebastian Technologies'>"
          "<span class='bdiv'></span><div class='btitles'>"
          "<span class='eyebrow'>Project Diamond</span><h1>Results</h1></div>" + DL + "</div>")
    else:
        A("<div class='brandrow'><div class='btitles'>"
          "<span class='eyebrow'>Project Diamond</span><h1>Results</h1></div>" + DL + "</div>")
    A(f"<div class='sub'>{len(rows)} embroidery files &middot; {len(groups)} distinct designs "
      f"&middot; generated in {elapsed:0.0f}s</div>")
    A("<div class='stats'>")
    for big, small in [(len(readable), "read &amp; rendered"),
                       (len(groups), "distinct designs"),
                       (len(multi_groups), "cross-format groups"),
                       (len(vip_groups), "groups with VIP"),
                       (tagged_designs, "tagged (Flash)"),
                       (len(unreadable), "couldn't read (ART)")]:
        A(f"<div class='stat'><b>{big}</b><span>{small}</span></div>")
    A("</div>")
    A("<nav class='tabs'>"
      "<button class='tab-btn active' data-tab='meta'>Metadata</button>"
      "<button class='tab-btn' data-tab='uuid'>UUID &amp; cross-format</button>"
      "<button class='tab-btn' data-tab='tags'>Image tagging</button>"
      "</nav></header>")

    # ---- TAB 1: UUID & cross-format ----
    A("<section id='tab-uuid' class='tabpane'>")
    A("<h2><span class='vipdot'>&#9679;</span> Same design, every format &mdash; one UUID</h2>")
    A("<p class='note'>Each block is one design that exists in several file formats. The content-derived "
      "UUID (perceptual hash, 7-bit tolerance) is identical across PES / DST / EXP / VP3 / JEF / HUS / XXX "
      "and the natively-read VIP, so the formats group together automatically. VIP files are outlined in magenta.</p>")
    fp = "".join(f"<span class='fmtpill'>{html.escape(f)} &middot; "
                 f"{sum(1 for r in readable if r['fmt']==f)}</span>" for f in formats_present)
    A(f"<div style='margin-bottom:14px'>{fp}</div>")
    if vip_groups:
        A("<h2 style='margin-top:8px'>VIP, alongside its siblings</h2>")
        for g in sorted(vip_groups, key=lambda g: (-len(g), g[0]["file"])):
            A(_group_block(g, True))
    A("<h2 style='margin-top:22px'>All cross-format groups</h2>")
    for g in multi_groups:
        A(_group_block(g, any(r["fmt"] == "VIP" for r in g)))
    A("</section>")

    # ---- TAB 2: Metadata ----
    A("<section id='tab-meta' class='tabpane active'>")
    A("<h2>Per-file metadata</h2>")
    A(f"<p class='note'>All {len(rows)} files. Dimensions, stitch count and thread-colour count are read "
      "straight from each embroidery file; the UUID is computed from the rendered design. ART files can't be "
      "read without Bernina ARTlink and show no UUID.</p>")
    A("<table><thead><tr><th>File</th><th>Format</th><th>Set</th><th>Size</th>"
      "<th>Stitches</th><th>Colors</th><th>UUID</th></tr></thead><tbody>")
    for r in sorted(rows, key=lambda r: (r["test_set"], r["file"])):
        cls = " class='vip'" if r["fmt"] == "VIP" else ""
        size = f'{r["width"]}"x{r["height"]}"' if r["width"] is not None else "&mdash;"
        stc = f'{r["stitches"]:,}' if r["stitches"] is not None else "&mdash;"
        col = r["colors"] if r["colors"] else "&mdash;"
        uid = html.escape(r["uuid"]) if r["uuid"] else (
            "<i>ART &mdash; needs ARTlink</i>" if r["fmt"] == "ART" else "&mdash;")
        E = lambda v: html.escape("" if v is None else str(v), quote=True)
        d_size = f'{r["width"]}" x {r["height"]}"' if r["width"] is not None else ""
        d_stc = f'{r["stitches"]:,}' if r["stitches"] is not None else ""
        d_tags = ", ".join(r.get("tags") or [])
        attrs = (f" data-file='{E(r['file'])}' data-fmt='{E(r['fmt'])}'"
                 f" data-set='{E(r['test_set'])}' data-size='{E(d_size)}'"
                 f" data-stitches='{E(d_stc)}' data-colors='{E(r['colors'])}'"
                 f" data-uuid='{E(r['uuid'])}' data-img='{E(r.get('render'))}'"
                 f" data-tags='{E(d_tags)}'")
        A(f"<tr{cls}{attrs}><td>{html.escape(r['file'])}</td><td>{html.escape(r['fmt'])}</td>"
          f"<td>{html.escape(r['test_set'])}</td><td>{size}</td><td>{stc}</td>"
          f"<td>{col}</td><td class='u'>{uid}</td></tr>")
    A("</tbody></table></section>")

    # ---- TAB 3: Image tagging ----
    A("<section id='tab-tags' class='tabpane'>")
    A("<h2>Gemini Flash tags</h2>")
    A(f"<p class='note'>Search keywords generated by the tagging service (Gemini Flash) for each distinct "
      f"design &mdash; {tagged_designs} of {len(groups)} designs tagged. Tags are produced once per design and "
      "shared across all of its file formats. These are what power keyword search in the partner product.</p>")
    if token_usage:
        tu = token_usage
        A("<div class='tokpanel'>"
          "<div class='tokhead'>Tagging token usage &mdash; latest run "
          f"<span class='tokmodel'>{html.escape(tu['model'])}</span></div>"
          "<div class='toktiles'>"
          f"<div class='toktile'><b>{tu['in_tok']:,}</b><span>input tokens</span></div>"
          f"<div class='toktile'><b>{tu['out_tok']:,}</b><span>output tokens</span></div>"
          f"<div class='toktile'><b>{tu['total_tok']:,}</b><span>total tokens</span></div>"
          f"<div class='toktile'><b>{tu['avg_tok']:,.0f}</b><span>avg / design</span></div>"
          f"<div class='toktile'><b>${tu['cost']:,.2f}</b><span>est. cost</span></div>"
          "</div>"
          f"<div class='toknote'>{tu['n']} designs tagged &middot; ${tu['cost']/tu['n']:.4f} per design "
          f"&middot; billed @ ${tu['in_rate']:.2f}/M input, ${tu['out_rate']:.2f}/M output</div>"
          "</div>")
    A("<div class='searchbar'>"
      "<input id='tagsearch' type='search' placeholder='Search tags or filename (e.g. flower, christmas)' autocomplete='off' spellcheck='false'>"
      "<span id='tagcount' class='searchcount'></span>"
      "</div>")
    A("<div class='designs'>")
    tg = [g for g in groups if any(r["tags"] for r in g)]
    ug = [g for g in groups if not any(r["tags"] for r in g)]
    for g in tg + ug:
        A(_design_tag_block(g))
    A("</div></section>")

    A("""<script>
      document.querySelectorAll('.tab-btn').forEach(function(b){
        b.addEventListener('click',function(){
          document.querySelectorAll('.tab-btn').forEach(function(x){x.classList.remove('active');});
          document.querySelectorAll('.tabpane').forEach(function(x){x.classList.remove('active');});
          b.classList.add('active');
          document.getElementById('tab-'+b.dataset.tab).classList.add('active');
          window.scrollTo(0,0);
        });
      });
      (function(){
        var inp=document.getElementById('tagsearch');
        if(!inp)return;
        var cards=Array.prototype.slice.call(document.querySelectorAll('#tab-tags .design'));
        var count=document.getElementById('tagcount');
        function apply(){
          var q=inp.value.trim().toLowerCase();
          var terms=q?q.split(' ').filter(Boolean):[];
          var shown=0;
          cards.forEach(function(c){
            var hay=c.getAttribute('data-search')||'';
            var ok=terms.every(function(t){return hay.indexOf(t)!==-1;});
            c.style.display=ok?'':'none';
            if(ok)shown++;
          });
          count.textContent=q?(shown+' of '+cards.length+' designs'):'';
        }
        inp.addEventListener('input',apply);
      })();
      (function(){
        var b=document.getElementById('dlbtn'), m=document.getElementById('dlmenu');
        if(!b||!m) return;
        function close(){ m.classList.remove('open'); b.setAttribute('aria-expanded','false'); }
        b.addEventListener('click', function(e){ e.stopPropagation(); var o=m.classList.toggle('open'); b.setAttribute('aria-expanded', o?'true':'false'); });
        document.addEventListener('click', function(e){ if(!m.contains(e.target) && e.target!==b) close(); });
        document.addEventListener('keydown', function(e){ if(e.key==='Escape') close(); });
        m.addEventListener('click', close);
      })();
    </script>""")
    A("""<div id="ddlg" class="ddlg" role="dialog" aria-modal="true" aria-label="File detail"><div class="ddbox"><button class="ddx" id="ddx" type="button" aria-label="Close">&times;</button><div class="ddimgwrap"><img id="ddimg" alt=""></div><div class="ddinfo"><h3 id="ddtitle"></h3><dl id="ddmeta"></dl><div class="ddsec">UUID</div><div id="dduuid" class="dduuid"></div><div class="ddsec">Tags</div><div id="ddtags" class="ddtags"></div></div></div></div>
<script>
(function(){
  var dlg=document.getElementById('ddlg'); if(!dlg) return;
  var img=document.getElementById('ddimg'), title=document.getElementById('ddtitle'),
      meta=document.getElementById('ddmeta'), uuid=document.getElementById('dduuid'),
      tagsEl=document.getElementById('ddtags');
  function esc(s){var d=document.createElement('div'); d.textContent=(s==null?'':s); return d.innerHTML;}
  function row(k,v){ return v ? ('<dt>'+esc(k)+'</dt><dd>'+esc(v)+'</dd>') : ''; }
  function show(d){
    if(d.img){ img.src=d.img; img.style.display=''; } else { img.src=''; img.style.display='none'; }
    title.textContent=d.title||'';
    meta.innerHTML=row('Format',d.fmt)+row('Test set',d.set)+row('Size',d.size)+row('Stitches',d.stitches)+row('Colors',d.colors);
    uuid.textContent=d.uuid||'—';
    var tags=(d.tags||'').split(',').map(function(s){return s.trim();}).filter(Boolean);
    tagsEl.innerHTML=tags.length?tags.map(function(t){return '<span class="chip">'+esc(t)+'</span>';}).join(''):'<span class="ddnone">no tags</span>';
    dlg.classList.add('open'); document.body.style.overflow='hidden';
  }
  function hide(){ dlg.classList.remove('open'); img.src=''; document.body.style.overflow=''; }
  document.getElementById('ddx').addEventListener('click',hide);
  dlg.addEventListener('click',function(e){ if(e.target===dlg) hide(); });
  document.addEventListener('keydown',function(e){ if(e.key==='Escape') hide(); });
  var mt=document.querySelector('#tab-meta tbody');
  if(mt) mt.addEventListener('click',function(e){
    var tr=e.target.closest('tr[data-file]'); if(!tr) return;
    show({img:tr.getAttribute('data-img'),title:tr.getAttribute('data-file'),fmt:tr.getAttribute('data-fmt'),set:tr.getAttribute('data-set'),size:tr.getAttribute('data-size'),stitches:tr.getAttribute('data-stitches'),colors:tr.getAttribute('data-colors'),uuid:tr.getAttribute('data-uuid'),tags:tr.getAttribute('data-tags')});
  });
  var tt=document.getElementById('tab-tags');
  if(tt) tt.addEventListener('click',function(e){
    var im=e.target.closest('.dimg img'); if(!im) return;
    var c=im.closest('.design'); if(!c) return;
    show({img:im.getAttribute('src'),title:c.getAttribute('data-name'),fmt:c.getAttribute('data-fmts'),uuid:c.getAttribute('data-uuid'),tags:c.getAttribute('data-tags')});
  });
})();
</script>""")
    A("</body></html>")
    path.write_text("".join(P), encoding="utf-8")


if __name__ == "__main__":
    main()
