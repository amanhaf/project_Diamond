"""
Export CDUID validation results from diamond_cache.sqlite to THREE CSV files:

  - uuid_results_all.csv   : every processed file with its CDUID and match metadata
  - uuid_duplicates.csv    : only files involved in any kind of duplicate detection
  - uuid_cross_format.csv  : only cross-format matches (same design, different
                             extensions), with an optional sha256_render column

Match types:
  cross_format    : same design appears in multiple file extensions (e.g. .dst + .jef)
  cross_filename  : same UUID appears across files with different design names
                    (e.g. Wedding.PES + Wedding_Rings.PES, OR a rotation match,
                    OR a genuine pHash collision -- review these)
  unique          : design appears in only one file

Note on sha256_render:
  The current pipeline is pHash-only and does NOT store a SHA-256 of the render
  in the cache (the SHA gate was removed from UUID generation). For continuity
  with the earlier uuid_cross_format.csv report, this script will compute the
  SHA-256 of the render PNG on the fly when the file is present on disk; if the
  render file is missing it leaves the field blank. Set COMPUTE_SHA256 = False
  to skip this entirely (faster, and the column will be blank).
"""

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "diamond_cache.sqlite"

ALL_OUT = SCRIPT_DIR / "uuid_results_all.csv"
DUPS_OUT = SCRIPT_DIR / "uuid_duplicates.csv"
CROSS_FORMAT_OUT = SCRIPT_DIR / "uuid_cross_format.csv"

# Compute SHA-256 of each render PNG for the cross-format report. Only affects
# uuid_cross_format.csv. Set False to leave the sha256_render column blank.
COMPUTE_SHA256 = True


def classify_match(formats_per_uuid: int, files_with_same_uuid: int) -> str:
    if files_with_same_uuid == 1:
        return "unique"
    if formats_per_uuid > 1:
        return "cross_format"
    return "cross_filename"


def sha256_of_file(path_str: str) -> str:
    """SHA-256 of a file's bytes, or '' if the file is missing/unreadable."""
    try:
        p = Path(path_str)
        if not p.exists():
            return ""
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM designs", conn)
    conn.close()

    print(f"Loaded {len(df)} rows from {DB_PATH.name}")

    # Derived columns
    df["filename"] = df["original_path"].str.extract(r"/([^/]+)$")[0]
    df["extension"] = df["original_path"].str.extract(r"\.([^.]+)$")[0].str.lower()
    df["test_set"] = df["original_path"].str.extract(r"/(Test_Set_\d+|vip\+art_converted)/")[0]
    df["design_name"] = df["filename"].str.replace(r"\.[^.]+$", "", regex=True)

    # Group statistics
    df["files_with_same_uuid"] = df.groupby("uuid")["uuid"].transform("count")
    df["formats_per_uuid"] = df.groupby("uuid")["extension"].transform("nunique")

    # Classify each row
    df["match_type"] = [
        classify_match(f, n)
        for f, n in zip(df["formats_per_uuid"], df["files_with_same_uuid"])
    ]

    # Sort: cross_format first, then cross_filename, then unique
    type_order = {"cross_format": 0, "cross_filename": 1, "unique": 2}
    df["_sort"] = df["match_type"].map(type_order)
    df = df.sort_values(by=["_sort", "uuid", "extension", "filename"]).drop(columns="_sort")

    # ---- Report 1: all results -------------------------------------------
    all_columns = [
        "uuid",
        "design_name",
        "filename",
        "extension",
        "test_set",
        "match_type",
        "files_with_same_uuid",
        "formats_per_uuid",
        "phash_render",
        "original_path",
    ]
    df_all = df[all_columns]
    df_all.to_csv(ALL_OUT, index=False)
    print(f"Wrote {len(df_all):>4} rows -> {ALL_OUT.name}")

    # ---- Report 2: duplicates only ---------------------------------------
    dups = df_all[df_all["match_type"] != "unique"].copy()
    dups.to_csv(DUPS_OUT, index=False)
    print(f"Wrote {len(dups):>4} rows -> {DUPS_OUT.name}")

    # ---- Report 3: cross-format matches only -----------------------------
    cf = df[df["match_type"] == "cross_format"].copy()
    cf["cross_format_match"] = True

    if COMPUTE_SHA256 and len(cf):
        # Hash the render PNG (render_path) for each cross-format file.
        print(f"Computing SHA-256 for {len(cf)} cross-format render files...")
        cf["sha256_render"] = cf["render_path"].apply(sha256_of_file)
        missing = (cf["sha256_render"] == "").sum()
        if missing:
            print(f"  ({missing} render file(s) missing on disk -> sha256 left blank)")
    else:
        cf["sha256_render"] = ""

    cross_columns = [
        "uuid",
        "design_name",
        "filename",
        "extension",
        "test_set",
        "cross_format_match",
        "files_with_same_uuid",
        "formats_per_uuid",
        "sha256_render",
        "phash_render",
        "original_path",
    ]
    cf_out = cf[cross_columns]
    cf_out.to_csv(CROSS_FORMAT_OUT, index=False)
    print(f"Wrote {len(cf_out):>4} rows -> {CROSS_FORMAT_OUT.name}")

    # ---- Summary ----------------------------------------------------------
    total = len(df_all)
    unique_uuids = df_all["uuid"].nunique()

    cross_format_uuids = df_all[df_all["match_type"] == "cross_format"]["uuid"].nunique()
    cross_format_files = (df_all["match_type"] == "cross_format").sum()

    cross_filename_uuids = df_all[df_all["match_type"] == "cross_filename"]["uuid"].nunique()
    cross_filename_files = (df_all["match_type"] == "cross_filename").sum()

    total_dup_uuids = cross_format_uuids + cross_filename_uuids

    print()
    print("=== Summary ===")
    print(f"Total files processed:                       {total}")
    print(f"Unique CDUIDs (distinct designs):            {unique_uuids}")
    print(f"Designs detected as duplicates:              {total_dup_uuids}")
    print()
    print("  By match type:")
    print(f"    cross_format   (same design, different extensions): {cross_format_uuids:>3} designs across {cross_format_files} files")
    print(f"    cross_filename (same UUID, different design names):  {cross_filename_uuids:>3} designs across {cross_filename_files} files")
    print()

    # cross_filename deserves a closer look: after rotation invariance ships,
    # this bucket holds rotation matches AND genuine pHash collisions.
    if cross_filename_uuids:
        print("  cross_filename groups (REVIEW — rotation matches vs false collisions):")
        cfn = df_all[df_all["match_type"] == "cross_filename"]
        for u, grp in cfn.groupby("uuid"):
            names = sorted(set(grp["design_name"]))
            print(f"    {u[:8]}...  {names}")
        print()

    # Distribution: how many files per duplicate group
    print("  Group sizes (files sharing one UUID):")
    grp_dist = (
        df_all[df_all["match_type"] != "unique"]
        .drop_duplicates("uuid")["files_with_same_uuid"]
        .value_counts()
        .sort_index()
    )
    for n_files, n_designs in grp_dist.items():
        print(f"    {n_designs:>3} design(s) appear in {n_files} files")


if __name__ == "__main__":
    main()