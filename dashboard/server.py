#!/usr/bin/env python3
"""Project Diamond — pipeline control plane.

Launches each Diamond pipeline as a managed subprocess so several can run at
once, with live logs and progress streamed to the browser over a WebSocket.
It does NOT re-implement any orchestration; it supervises the binaries/scripts
you already have and parses the lines they already print.

Run:
    pip install fastapi "uvicorn[standard]"
    uvicorn server:app --reload --port 8077
    # open http://127.0.0.1:8077

Local UI smoke test without the real binaries:
    DEMO=1 uvicorn server:app --port 8077
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# CONFIG  — edit to match your machine.
#
# Each pipeline launches as a subprocess from `cwd`; `cmd` is a token list.
# If `arg_label` is set, the value typed in the UI is appended to `cmd`.
# `total_re` (optional) pulls a total count out of one log line; `tick_re`
# (optional) counts progress lines. Omit both for an indeterminate (stitch)
# progress bar. `long_running` marks services that stay up (no completion).
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("DIAMOND_ROOT", "/Users/amans/Desktop/Mark-VLM")
VENV_PY = os.environ.get("DIAMOND_PY", f"{PROJECT_ROOT}/venv/bin/python")

PIPELINES = [
    {
        "id": "ingest",
        "label": "Scan / UUID / Tagging",
        "desc": "Rust scan + UUID and Gemini Flash tagging, run together as one job",
        "cmd": [VENV_PY, "run_scan_and_tag.py"],
        "cwd": PROJECT_ROOT,
        "arg_label": None,
        "total_re": r"\[scan\] Found (\d+) candidate files",
        "tick_re": r"\[scan\] (NEW|MATCH|ALREADY CACHED)\b",
    },
    {
        "id": "catalog",
        "label": "Build catalog",
        "desc": "Render + UUID-group + attach Flash tags -> results gallery (/results)",
        "cmd": [VENV_PY, "run_diamond_demo.py"],
        "cwd": PROJECT_ROOT,
        "arg_label": None,
        "total_re": r"Found (\d+) design files",
        "tick_re": r"processed\.\.\.",
    },
    {
        "id": "scrape",
        "label": "Web locator",
        "desc": "Find design images on a designer site",
        "cmd": [VENV_PY, "web_locator.py"],
        "cwd": PROJECT_ROOT,
        "arg_label": "Site URL",
        "arg_kind": "url",
        "total_re": r"Found (\d+) candidate images",
        "tick_re": r"\[\d+/\d+\]",
    },
]

# Hide any pipeline whose entry .py script is not present yet (e.g. web_locator.py),
# so a missing script never shows as a red card or breaks "Start all".
def _script_present(spec) -> bool:
    for tok in spec.get("cmd", []):
        if isinstance(tok, str) and tok.endswith(".py"):
            if not (Path(spec.get("cwd", PROJECT_ROOT)) / tok).exists():
                return False
    return True

PIPELINES = [p for p in PIPELINES if _script_present(p)]

# Fake pipelines so the UI can be exercised end-to-end without the real tools.
if os.environ.get("DEMO"):
    PIPELINES = [
        {
            "id": "ingest", "label": "Scan / UUID / Tagging",
            "desc": "demo: scan + tag together",
            "cmd": ["bash", "-c",
                    "echo '[scan] Found 6 candidate files'; "
                    "for i in 1 2 3 4 5 6; do "
                    "  if [ $((i%3)) -eq 0 ]; then echo \"[scan] MATCH: /x/file$i -> uuid$i\"; "
                    "  else echo \"[scan] NEW: /x/file$i -> uuid$i\"; fi; "
                    "  echo \"[tag] [$i/6] file$i.png  flower, leaf\"; sleep 0.5; done; "
                    "echo '[scan] scan complete'; echo 'All steps complete.'"],
            "cwd": str(HERE),
            "arg_label": None,
            "total_re": r"\[scan\] Found (\d+) candidate files",
            "tick_re": r"\[scan\] (NEW|MATCH|ALREADY CACHED)\b",
        },
        {
            "id": "scrape", "label": "Web locator",
            "desc": "demo: takes a URL arg",
            "cmd": ["bash", "-c", "echo \"locating images on: $0\"; sleep 1; echo 'found 12 images'"],
            "cwd": str(HERE), "arg_label": "Site URL", "arg_kind": "url",
        },
    ]


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, spec: dict):
        self.spec = spec
        self.status = "idle"            # idle | running | done | error | stopped
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.task: Optional[asyncio.Task] = None
        self.return_code: Optional[int] = None
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.total: Optional[int] = None
        self.done = 0
        self.log = deque(maxlen=600)

    def snapshot(self, include_log: bool = True) -> dict:
        s = {
            "id": self.spec["id"],
            "label": self.spec["label"],
            "desc": self.spec.get("desc", ""),
            "arg_label": self.spec.get("arg_label"),
            "arg_kind": self.spec.get("arg_kind", "text"),
            "long_running": self.spec.get("long_running", False),
            "status": self.status,
            "total": self.total,
            "done": self.done,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "return_code": self.return_code,
        }
        if include_log:
            s["log"] = list(self.log)[-150:]
        return s


JOBS = {p["id"]: Job(p) for p in PIPELINES}
clients: "set[WebSocket]" = set()


async def broadcast(message: dict) -> None:
    data = json.dumps(message)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


def full_snapshot() -> dict:
    return {"type": "snapshot",
            "pipelines": [JOBS[p["id"]].snapshot() for p in PIPELINES]}


async def push_status(job: Job) -> None:
    await broadcast({"type": "status", "pipeline": job.snapshot(include_log=False)})


async def run_pipeline(pipe_id: str, arg: Optional[str]) -> None:
    job = JOBS[pipe_id]
    spec = job.spec
    cmd = list(spec["cmd"])
    if arg:
        cmd.append(arg)

    job.status = "running"
    job.return_code = None
    job.started_at = time.time()
    job.ended_at = None
    job.total = None
    job.done = 0
    job.log.clear()
    job.log.append(f"$ {' '.join(cmd)}   (cwd={spec['cwd']})")
    await push_status(job)

    total_re = re.compile(spec["total_re"]) if spec.get("total_re") else None
    tick_re = re.compile(spec["tick_re"]) if spec.get("tick_re") else None

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=spec["cwd"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=os.environ.copy(),
        )
    except Exception as e:
        job.status = "error"
        job.ended_at = time.time()
        msg = f"[failed to start] {e}"
        job.log.append(msg)
        await broadcast({"type": "log", "id": pipe_id, "line": msg})
        await push_status(job)
        return

    job.proc = proc
    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip("\n")
        job.log.append(line)
        if total_re:
            m = total_re.search(line)
            if m:
                job.total = int(m.group(1))
        if tick_re and tick_re.search(line):
            job.done += 1
        await broadcast({"type": "log", "id": pipe_id, "line": line,
                         "done": job.done, "total": job.total})

    rc = await proc.wait()
    job.return_code = rc
    job.ended_at = time.time()
    if job.status != "stopped":
        job.status = "done" if rc == 0 else "error"
    job.proc = None
    await push_status(job)


# ---------------------------------------------------------------------------
# HTTP / WebSocket
# ---------------------------------------------------------------------------
app = FastAPI(title="Project Diamond Dashboard")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/api/state")
async def state() -> JSONResponse:
    return JSONResponse(full_snapshot())


@app.post("/api/pipelines/{pipe_id}/start")
async def start(pipe_id: str, body: dict = Body(default={})) -> JSONResponse:
    job = JOBS.get(pipe_id)
    if job is None:
        return JSONResponse({"ok": False, "error": "unknown pipeline"}, status_code=404)
    if job.status == "running":
        return JSONResponse({"ok": False, "error": "already running"}, status_code=409)
    arg = (body or {}).get("arg") or None
    job.task = asyncio.create_task(run_pipeline(pipe_id, arg))
    return JSONResponse({"ok": True})


@app.post("/api/pipelines/{pipe_id}/stop")
async def stop(pipe_id: str) -> JSONResponse:
    job = JOBS.get(pipe_id)
    if job is None or job.proc is None:
        return JSONResponse({"ok": False, "error": "not running"}, status_code=409)
    job.status = "stopped"
    proc = job.proc
    proc.terminate()
    await push_status(job)

    async def _kill_later(p: asyncio.subprocess.Process) -> None:
        await asyncio.sleep(5)
        if p.returncode is None:
            p.kill()

    asyncio.create_task(_kill_later(proc))
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    await ws.send_text(json.dumps(full_snapshot()))
    try:
        while True:
            await ws.receive_text()   # client never sends; this just keeps it open
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


# --- results gallery (generated by run_diamond_demo.py) -------------------
from fastapi.staticfiles import StaticFiles as _StaticFiles
_RESULTS_DIR = Path(PROJECT_ROOT) / "_diamond_demo"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", _StaticFiles(directory=str(_RESULTS_DIR), html=True), name="results")
