import sys, shutil, time
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/amans/Desktop/Mark-VLM")
sys.path.insert(0, str(ROOT / "Image_Recognition"))
sys.path.insert(0, str(ROOT))
import run_compare_aws_azure_clarifai as R

CSV = ROOT / "Image_Recognition" / "image_recognition_results.csv"
ts = time.strftime("%Y%m%d_%H%M%S")
bak = CSV.with_name(CSV.stem + f".bak_{ts}.csv")
shutil.copy(CSV, bak)
print("backup ->", bak.name)

df = pd.read_csv(CSV).set_index("image_file", drop=True)

def get_tags(name):
    if name not in df.index: return None
    v = df.at[name, "gemini_flash_tags"]
    return "" if pd.isna(v) else str(v)

# ---------- 1) repair 12702-16 blank color render, then re-tag via Flash ----------
bw    = ROOT / "Metadata_Capture" / "rendered_images" / "12702-16.png"
color = ROOT / "Metadata_Capture" / "rendered_images_color" / "12702-16.png"
print("\n[1] 12702-16 BEFORE:", repr(get_tags("12702-16.png")))
if bw.exists():
    shutil.copy(bw, color)           # replace the blank 6KB color render with the good B&W one
    print("    repaired color render (copied B&W ->", color.name + ")")
try:
    cfg = R.load_cfg(); client = R.gemini_client(cfg)
    prepared = R.prepare_image_bytes_for_apis(bw.read_bytes())
    tags, conf, it, ot, tt = R.gemini_tags_with_usage(client, prepared, R.GEMINI_FLASH_MODEL)
    if tags and str(tags).strip():
        df.at["12702-16.png", "gemini_flash_tags"] = tags
        df.at["12702-16.png", "gemini_flash_confidences"] = conf
        df.at["12702-16.png", "gemini_flash_model"] = R.GEMINI_FLASH_MODEL
        df.at["12702-16.png", "gemini_flash_input_tokens"] = it
        df.at["12702-16.png", "gemini_flash_output_tokens"] = ot
        df.at["12702-16.png", "gemini_flash_total_tokens"] = tt
        df.at["12702-16.png", "gemini_flash_error"] = None
        print("    Flash re-tag ->", tags)
    else:
        print("    Flash returned empty again; craft tags will still be forced below.")
except Exception as e:
    print("    Flash re-tag FAILED:", e, "(craft tags will still be forced below)")

# ---------- helpers ----------
def force_front(name, canon):
    """Force the canon craft tags at the front (canonical order), keep the rest, dedup."""
    cur = get_tags(name)
    parts = [t.strip() for t in cur.split(",") if t.strip()]
    have = {p.lower() for p in parts}
    canon_low = {c.lower() for c in canon}
    rest = [p for p in parts if p.lower() not in canon_low]
    df.at[name, "gemini_flash_tags"] = ", ".join(list(canon) + rest)
    return [c for c in canon if c.lower() not in have]

def ensure_present(name, tag):
    """Add tag at the front if missing; otherwise leave as-is."""
    cur = get_tags(name)
    parts = [t.strip() for t in cur.split(",") if t.strip()]
    if tag.lower() in {p.lower() for p in parts}:
        return False
    df.at[name, "gemini_flash_tags"] = ", ".join([tag] + parts)
    return True

# ---------- 2) 12702 line: free standing lace, lace, applique on ALL 16 ----------
print("\n[2] 12702 line -> ensure {free standing lace, lace, applique} on all 16")
for n in range(1, 17):
    nm = f"12702-{n:02d}.png"
    if nm in df.index:
        added = force_front(nm, ["free standing lace", "lace", "applique"])
        if added: print(f"    {nm}: added {added}")

# ---------- 3) E-ASD: ensure 'embroidery' present on all ----------
print("\n[3] E-ASD -> ensure 'embroidery' present on all")
easd = [x for x in df.index if str(x).upper().startswith("E-ASD")]
for nm in easd:
    if ensure_present(nm, "embroidery"):
        print(f"    {nm}: added embroidery")

df.reset_index().to_csv(CSV, index=False)

# ---------- verify ----------
df2 = pd.read_csv(CSV).set_index("image_file", drop=True)
def has(nm, t): return t.lower() in str(df2.at[nm,"gemini_flash_tags"]).lower()
miss_app = [f"12702-{n:02d}.png" for n in range(1,17) if f"12702-{n:02d}.png" in df2.index and not has(f"12702-{n:02d}.png","applique")]
miss_emb = [nm for nm in easd if not has(nm,"embroidery")]
print("\n=== VERIFY ===")
print("12702-16 now:", repr(str(df2.at['12702-16.png','gemini_flash_tags'])[:140]))
print("12702 designs still missing applique:", miss_app or "NONE")
print("E-ASD still missing embroidery       :", miss_emb or "NONE")
empties = df2[df2['gemini_flash_tags'].isna() | (df2['gemini_flash_tags'].astype(str).str.strip().isin(['','nan']))]
print("rows with empty flash tags anywhere  :", len(empties))
