#!/usr/bin/env python3
"""
run_scan_and_tag.py
===================
Run the Rust scan/UUID pass and Gemini Flash tagging TOGETHER as a single
process. The two steps are independent -- the scanner renders files and writes
UUIDs into the SQLite cache, while the tagger reads already-rendered images and
writes tags into image_recognition_results.csv -- so they run concurrently.

Both children's output is streamed live, each line prefixed with [scan] or
[tag] so you can tell them apart. Exit code is non-zero if either step fails.

This backs the dashboard's single "Scan / UUID / Tagging" box.

Run directly:
    venv/bin/python run_scan_and_tag.py
"""
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / "venv" / "bin" / "python"

STEPS = [
    {
        "tag": "scan",
        "cmd": [str(ROOT / "Client_Rust" / "target" / "release" / "diamond_client")],
        "cwd": str(ROOT / "Client_Rust"),
    },
    {
        "tag": "tag",
        "cmd": [str(VENV_PY), "Image_Recognition/run_gemini_flash_tags.py"],
        "cwd": str(ROOT),
    },
]

_print_lock = threading.Lock()


def _emit(prefix: str, line: str) -> None:
    with _print_lock:
        print(f"[{prefix}] {line}", flush=True)


def run_step(step: dict, results: dict) -> None:
    tag = step["tag"]
    try:
        proc = subprocess.Popen(
            step["cmd"], cwd=step["cwd"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except Exception as e:
        _emit(tag, f"failed to start: {e}")
        results[tag] = 1
        return
    assert proc.stdout is not None
    for raw in proc.stdout:
        _emit(tag, raw.rstrip("\n"))
    rc = proc.wait()
    results[tag] = rc
    _emit(tag, f"exited with code {rc}")


def main() -> None:
    results: dict = {}
    threads = []
    for step in STEPS:
        t = threading.Thread(target=run_step, args=(step, results), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    failed = [k for k, v in results.items() if v != 0]
    if failed:
        print(f"FAILED steps: {', '.join(failed)}", flush=True)
        sys.exit(1)
    print("All steps complete.", flush=True)


if __name__ == "__main__":
    main()
