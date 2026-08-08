from pathlib import Path
import shutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from registry import ConversionRegistry

app = FastAPI(title="Local Embroidery Converter")

CACHE_DIR = Path.home() / ".embroidery_converter_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

registry = ConversionRegistry()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/convert")
async def convert(file: UploadFile = File(...), target_format: str = Form("pes")):
    if target_format.lower() != "pes":
        return JSONResponse(
            {"ok": False, "error": "Only target_format=pes is supported."},
            status_code=400,
        )

    src_name = Path(file.filename).name
    src_path = CACHE_DIR / src_name

    with open(src_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = registry.convert_to_pes(src_path, CACHE_DIR)

    if not result.ok or not result.output_path:
        return JSONResponse(
            {
                "ok": False,
                "backend": result.backend,
                "error": result.error,
            },
            status_code=500,
        )

    output_path = Path(result.output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        return JSONResponse(
            {
                "ok": False,
                "backend": result.backend,
                "error": "Output file missing or empty.",
            },
            status_code=500,
        )

    return FileResponse(
        path=output_path,
        media_type="application/octet-stream",
        filename=output_path.name,
    )