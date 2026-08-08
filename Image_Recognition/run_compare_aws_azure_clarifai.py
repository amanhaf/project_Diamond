"""
Tag-generation benchmark for Project Diamond.

v5 changes (260528, from Mark's updated Descriptions doc + row 50-55 callout):
  - JSON output mode. Gemini now returns
    [{"tag": "...", "confidence": 0.x}, ...] instead of a comma-separated
    string. We filter by MIN_CONFIDENCE, sort by confidence descending, and
    cap at TOP_K. Per Mark's spec: "Deliver only tags with a 50% or greater
    confidence interval — we may adjust the threshold."
  - Two new CSV columns per Gemini model: gemini_confidences and
    gemini_flash_confidences. The existing gemini_tags / gemini_flash_tags
    columns stay clean comma-separated tag strings (top-K, filtered) so
    Mark's macro keeps working unchanged.
  - Prompt additions from Mark's 260528 doc:
      * FSL → ALSO output `lace`.
      * Charm = circle ONLY at center top (or center side if rotated).
      * If sockets/pegs present, IGNORE the outer geometric shape.
      * Abstract-shape fallback (e.g. `spiral`) when no subject identifiable.
      * Compound rule (from row 50-55): sockets + empty center → BOTH
        `free standing lace` AND `applique`. This was implicit in v4 but is
        now an explicit standalone rule.

Prior changes preserved:
  - Color renders at Metadata_Capture/rendered_images_color/ (B&W fallback).
  - v4 embroidery hierarchy rule (no `embroidery` when a subcategory applies).
  - Claude branch removed (Flash beat Sonnet 50% vs 36% at 1/27th the cost).
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageFile

# --------------------------------------------------
# PIL SAFETY
# --------------------------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 250_000_000
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

# --------------------------------------------------
# Ensure project root is on sys.path
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Gemini
from google import genai
from google.genai import types

# Clarifai (optional — pretrained general model for keywords)
CLARIFAI_AVAILABLE = True
try:
    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
    from clarifai_grpc.grpc.api.status import status_code_pb2
except Exception:
    CLARIFAI_AVAILABLE = False

# Local ONNX inference (for craft classification)
LOCAL_AVAILABLE = True
try:
    from Image_Recognition.local_inference import local_predict
except Exception as _local_imp_err:
    LOCAL_AVAILABLE = False
    _LOCAL_IMP_ERR = str(_local_imp_err)


# --------------------------------------------------
# PATHS
# --------------------------------------------------
ENV_PATH = PROJECT_ROOT / ".env"

# NOTE: Now reading from COLOR renders. Run rerender_color.py first to
# populate this directory; falls back to the B&W renders if not present.
RENDER_DIR = PROJECT_ROOT / "Metadata_Capture" / "rendered_images_color"
RENDER_DIR_BW_FALLBACK = PROJECT_ROOT / "Metadata_Capture" / "rendered_images"

METADATA_CSV = PROJECT_ROOT / "Metadata_Capture" / "metadata_results.csv"
OUT_CSV = PROJECT_ROOT / "Image_Recognition" / "image_recognition_results.csv"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
MAX_DIM_FOR_APIS = 2048
MAX_BYTES = 5 * 1024 * 1024

# Clarifai concept filtering
CL_MIN_CONF = 0.20
CL_MAX_CONCEPTS = 25

# Two Gemini models
GEMINI_PRO_MODEL = "gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini-2.5-flash"

# Disable Gemini thinking for keyword tagging (huge speedup, no quality loss for this task)
GEMINI_THINKING_BUDGET = 128  # 0 = no thinking; -1 = dynamic (default); >0 = capped budget

# --------------------------------------------------
# v5 OUTPUT TUNABLES — change here, nowhere else
# --------------------------------------------------
MIN_CONFIDENCE = 0.5   # Mark's spec: "50% or greater confidence interval"
TOP_K = 12             # cap on tags per design after filtering & sorting
                       # Raised from 8 to 12 (260528 scoring run): v5 with K=8
                       # regressed Flash mean recall by 1.3% vs v4. Root cause
                       # was the cap dropping useful tags (e.g. the literal
                       # letter 'p' on Embossed P, 'celtic' on Cross2) that v4
                       # had been emitting in slots 9-15. K=12 keeps all the
                       # v5 prompt wins (compound applique on 12702 series,
                       # FSL→lace, etc.) while restoring v4-like breadth.

RUN_CLARIFAI_SMOKE_TEST = True

"""
GEMINI_PROMPT v5 — Mark's Descriptions doc (260528) + row 50-55 callout.

Changes from v4:
  - JSON output (with confidence per tag) replaces comma-separated string.
    The prompt instructs Gemini to return up to 15 candidates but only
    include tags with confidence >= 0.5; downstream parser then sorts by
    confidence and caps at TOP_K.
  - FSL → ALSO output `lace` (explicit; Flash was at 89% on this already).
  - Charm rule tightened: only when the circle is at center top.
  - Sockets/pegs → IGNORE outer geometric envelope (no oval/star/flower
    for the design's shape).
  - Compound rule promoted: sockets + empty center → BOTH FSL AND applique.
    Targets the 12702 series specifically (rows 50-55 in Mark's sheet).
  - Abstract-shape fallback (e.g. `spiral`) when no subject identifiable.
  - Singular+plural meta rule alongside the explicit pair list.

Preserved from v4:
  - Embroidery hierarchy (no `embroidery` when subcategory applies).
  - All vocabulary rules (AFF, damask, christmas ornaments, etc.).
  - Pearl rule, sparkle rule, text rule.
"""

# v6 (260629, Mark's IR refinement notes): +`embroidery border` type, +required design-type
# tag, +border-vs-edge-to-edge shape test, +quilting/lace mutual exclusion. The 1-color and
# size rules are enforced deterministically in apply_metadata_rules.py (run after tagging).
GEMINI_PROMPT = """You are an expert at analyzing images of designs intended for machine embroidery, applique, quilting, and related crafts. Generate keywords describing the design.

# Output format (CRITICAL)
Return a JSON array of objects, each with a `tag` (lowercase string) and a `confidence` (number between 0 and 1 reflecting how certain you are the tag applies). Example:

[{"tag": "applique", "confidence": 0.92}, {"tag": "butterfly", "confidence": 0.88}]

Output ONLY the JSON array — no preamble, no explanation, no code fences.

# How many tags to emit
- Emit up to 15 candidates, but only INCLUDE a tag if you are at least 50% confident it applies.
- Confidence should reflect genuine certainty, not enthusiasm. A confidence of 0.95 means the tag almost certainly applies; 0.55 means it might apply but you would not stake the design on it.
- It is BETTER to return 4 confident tags than 12 padded ones. Do not invent tags to reach a quota.
- Do not repeat a tag.

# Craft-category taxonomy (MOST IMPORTANT SECTION)
Every design is a form of embroidery. The categories below are sub-types. Identify which sub-type the design is and tag it. IMPORTANT: if the design matches a sub-type below, do NOT also output the generic word `embroidery`. Only output `embroidery` when NONE of the sub-types apply.

- `applique` — a shape whose OUTER EDGE is defined by a band of heavy, narrow satin stitching, with little or no stitching filling the center (a piece of fabric is laid in the middle and the stitching secures it). Output `applique` ONLY when you see this open-center-with-stitched-border construction. Do NOT output `applique` for a design that is fully stitched/filled, or for plain outlined line art.
- `lace` — stitching that contains a regular SQUARE or DIAMOND grid pattern (this grid is the signature of lace). For lace attached to a garment or project, output just `lace`.
- `free standing lace` — lace with no fabric backing (stitched on wash-away stabilizer so only the lace remains). Cues: open lacework that stands alone; small round "pegs" on one edge and "sockets" or "loops" on the opposite edge (these are connection points for assembling components). When you output `free standing lace`, ALSO output `lace`.
- `charm` — a free-standing-lace decoration with a hanging loop. CRITICAL: a circle on the edge of a design is a charm hang-loop ONLY when it sits at the CENTER TOP of the design (or center side if the design is rotated 90 degrees). A circle elsewhere on the perimeter is NOT a charm. When you output `charm`, ALSO output `free standing lace` and `lace`.
- `font` — an alphabet, or letters/numbers/characters shown as a typeface. Output `font`.
- `monogram` — one to three letters in the center of a design, usually in a stylized or block letter style. Output `monogram`.
- `in the hoop project` — a complete standalone finished item (not a decorative element of a larger project). Often rectangular, often with an outer "echo" line marking where to cut it out. If it is a small coaster-style mat, ALSO output `mug rug` and `coaster`.
- `embroidery border` -- a SMALL, THIN rectangular strip of stitching meant to be sewn along the EDGE of fabric (clothing, tablecloth, drapery), NOT to join quilt layers. Cues: a thin, elongated rectangle; a motif that REPEATS along its length; sometimes a center line that lines up with the next repeat. Output `embroidery border` AND `border`. Do NOT output `quilting` or `edge-to-edge` for these.
- `quilting` — a single continuous line of stitching that fills a rectangular, square, or triangular area, used to join fabric layers. Then add the variant:
    * `edge-to-edge` — stitching runs off one edge and onto another (start/end points sit at the design edges).
    * `border` — a long rectangular strip meant to frame a quilt edge.
    * `cornerstone` — a roughly square block (within ~90% square) for the corner where borders meet.
    * (self-connecting square designs get just `quilting`, no variant.)
  For quilting designs, ALSO identify and tag the pictorial elements the stitching draws (e.g. `pumpkin`, `leaf`, `leaves`).

# Every design MUST carry a type tag
Output at least one design-type tag from this list: embroidery, embroidery border, applique, lace, free standing lace, charm, font, monogram, in the hoop project, quilting, edge-to-edge, border, cornerstone. If none of the specific sub-types clearly applies, output `embroidery` as the default. Never return a design with no type tag.

# Applique is easy to miss -- look for outlined subjects
An OUTLINED subject is applique: a letter, animal, flower, or motif drawn as an outline
(a bold satin or zigzag stitched border) with an EMPTY or lightly-stitched middle that a
piece of fabric fills. Whenever the subject you tag is rendered as a bordered outline with
a sparse or open interior, ALSO output `applique` -- including an applique alphabet (output
`font` AND `applique`), an applique animal, or an applique coaster / mug rug. A fully-filled
shape with dense interior stitching is NOT applique.

# Border vs edge-to-edge (shape test)
`edge-to-edge` quilting fills a LARGE, roughly square-ish area (the stitching runs off one edge onto the next). A THIN, ELONGATED strip carrying a repeating motif is a BORDER, not edge-to-edge -- output `border`, not `edge-to-edge`.

# Mutual exclusion (quilting vs lace)
A design is EITHER quilting/`edge-to-edge` OR `lace`/`free standing lace` -- never both. A stitched square/diamond grid means lace. A single continuous fill line joining fabric layers means quilting. Do not output both families on the same design.

# Compound rule: sockets + empty center (HIGH PRIORITY)
When you see BOTH of these together in one design:
  (a) small connection marks around the perimeter — pegs, sockets, loops, or notches
  (b) a large empty (un-stitched) area in the middle
output BOTH `free standing lace` AND `applique` (and `lace` per the FSL rule above). This combination is a free-standing-lace applique component — the empty middle is where fabric is laid in. Examples: the 12702 series, 12644 series.

# Ignore the outer envelope when sockets are present
When a design has pegs/sockets/loops on the perimeter, IGNORE the outer geometric shape of the design itself. Do NOT tag `oval`, `star`, `flower`, `circle`, or other shape descriptors for the envelope — those describe the component's footprint, not its content. Tag what is INSIDE the design (subjects, motifs) and the construction (FSL, applique, lace).

# Fallback for abstract designs
If you cannot identify any objects, animals, plants, or recognizable subjects, it is OK to tag pure visual elements: `spiral`, `swirl`, `radial`, `zigzag`, `wave`, `grid`, `dots`, `lines`, etc. Use these only when there is no identifiable subject — not as fillers alongside real subjects.

# Required vocabulary — use these EXACT phrases when applicable
- `american flag` (not just "flag" or "patriotic")
- `holly leaf`, `holly berries`
- `candy cane`
- `clam shell`
- `pine tree`, `christmas tree`
- `pine branches`, `pine branch`
- `christmas ornament`, `christmas ornaments` (plural if multiple shown)
- `oval ornaments`
- `christmas presents`
- `damask` — ornate symmetrical scrolling floral textile pattern; also output `flower` and the specific flower if identifiable
- `string lights`, `christmas lights`

# Pearl rule
When you output `clam shell` or `shell`, ALSO output `pearl`.

# Sparkle rule
When the design has small radiating starburst/twinkle/shimmer marks (common on Christmas ornament designs), output BOTH `sparkle` and `sparkles`.

# American flag font rule
A letter/number filled with red and white horizontal stripes plus a blue field of white stars is an American flag font. Output: `american flag`, `font`, `alphabet`, `letter`, `star`, `stars`, `stripes`, `patriotic`.

# Text rule
Output `text` whenever the design contains ANY letter, number, or word — even a single stylized letter. Do NOT transcribe the actual letters.

# Singular and plural pairs
When you tag a design element, also tag its singular/plural counterpart where the counterpart is a real English word. Specifically these pairs — emit BOTH forms:
- `leaf` and `leaves`
- `star` and `stars`
- `flower` and `flowers`
- `branch` and `branches`
- `tree` and `trees`
- `ornament` and `ornaments`
- `snowflake` and `snowflakes`
- `swirl` — also `scroll`

For other countable nouns you tag (e.g. `bell`, `dog`, `bird`), include both singular and plural where the design shows one OR many.

EXCEPTIONS — singular ONLY, do not output the plural:
- `letter` only (never `letters`)
- `stripes` only (never `stripe`)

# Subjects
- Primary subjects: objects, animals, plants, distinct shapes.
- Broader categories (butterfly → also `insect`, `animal`).
- Themes where clear: `christmas`, `nautical`, `floral`, `patriotic`, `wedding`.

# Do NOT output
- `embroidery` when any sub-type above applies (see taxonomy section).
- Stitch-technique words: `satin stitch`, `tack down stitch`, `diagonal fill`, `underlay`, `jump stitch`, `fringe`, `jagged edge`.
- Generic filler: `decorative`, `ornamental`, `pattern`, `design`, `motif`, `elegant`, `delicate` — unless the image is truly abstract with no identifiable subject.
- Geometric envelope shapes (`oval`, `star`, `flower`, `circle`) when the design has FSL pegs/sockets — see "Ignore the outer envelope" rule above.
- Both `line art` and `embroidery` on the same image.

Now analyze the image and return the JSON array.
"""


# --------------------------------------------------
# GEMINI JSON-MODE RESPONSE SCHEMA
# --------------------------------------------------
# This schema is passed to Gemini as response_schema; combined with
# response_mime_type="application/json" it forces structured output that
# the SDK guarantees will parse as JSON conforming to this shape.
GEMINI_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "tag": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["tag", "confidence"],
    },
}


def parse_gemini_json_response(response_text: str) -> Tuple[str, str]:
    """
    Parse Gemini's JSON output into the two CSV strings the writer expects.

    Pipeline:
      1. Parse JSON (robust to malformed responses → returns empty strings).
      2. Filter to confidence >= MIN_CONFIDENCE.
      3. De-duplicate by tag, keeping the highest confidence per tag.
      4. Sort by confidence descending.
      5. Cap at TOP_K.

    Returns:
        (tags_csv, conf_csv) — both comma-separated, same length, same order.
        Example: ("applique, butterfly, lace", "0.92, 0.88, 0.71")
    """
    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return "", ""

    if not isinstance(parsed, list):
        return "", ""

    cleaned: List[Tuple[str, float]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag")
        conf = item.get("confidence")
        if not isinstance(tag, str) or not isinstance(conf, (int, float)):
            continue
        tag = tag.strip().lower()
        if not tag:
            continue
        conf = max(0.0, min(1.0, float(conf)))  # clamp to [0,1]
        if conf < MIN_CONFIDENCE:
            continue
        cleaned.append((tag, conf))

    # De-duplicate by tag, keeping the highest confidence seen
    by_tag: Dict[str, float] = {}
    for tag, conf in cleaned:
        if tag not in by_tag or conf > by_tag[tag]:
            by_tag[tag] = conf
    deduped = list(by_tag.items())

    # Sort by confidence descending, cap at TOP_K
    deduped.sort(key=lambda tc: -tc[1])
    deduped = deduped[:TOP_K]

    tags_csv = ", ".join(t for t, _ in deduped)
    conf_csv = ", ".join(f"{c:.2f}" for _, c in deduped)
    return tags_csv, conf_csv


# --------------------------------------------------
# ENV
# --------------------------------------------------
def load_cfg() -> Dict[str, str]:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f".env not found at {ENV_PATH}")

    load_dotenv(ENV_PATH, override=True)

    required = ["GEMINI_API_KEY"]
    missing = [k for k in required if not (os.getenv(k) or "").strip()]
    if missing:
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

    cfg: Dict[str, str] = {
        "gemini_api_key": os.getenv("GEMINI_API_KEY").strip(),
    }

    for k in [
        "CLARIFAI_PAT",
        "CLARIFAI_GENERAL_USER_ID",
        "CLARIFAI_GENERAL_APP_ID",
        "CLARIFAI_GENERAL_MODEL_ID",
    ]:
        v = os.getenv(k)
        if v and v.strip():
            cfg[k.lower()] = v.strip()

    return cfg


def list_images(images_dir: Path) -> List[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.webp"]
    files: List[Path] = []
    for ext in exts:
        files.extend(images_dir.glob(ext))
    return sorted(files)


def resolve_render_dir() -> Path:
    """Prefer the color renders; fall back to B&W with a loud warning."""
    if RENDER_DIR.exists() and any(RENDER_DIR.glob("*.png")):
        print(f"Reading color renders from: {RENDER_DIR}")
        return RENDER_DIR
    print(
        f"WARNING: color render directory {RENDER_DIR} is empty or missing.\n"
        f"  Falling back to B&W renders at {RENDER_DIR_BW_FALLBACK}.\n"
        f"  AFF (American flag font) detection requires color renders.\n"
        f"  Run worker_python/rerender_color.py to generate them."
    )
    return RENDER_DIR_BW_FALLBACK


# --------------------------------------------------
# IMAGE PREP
# --------------------------------------------------
def prepare_image_bytes_for_apis(
    image_bytes: bytes,
    max_dim: int = MAX_DIM_FOR_APIS,
    max_bytes: int = MAX_BYTES,
) -> bytes:
    img = Image.open(BytesIO(image_bytes))
    img.load()

    w, h = img.size
    longest = max(w, h)

    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img = img.convert("RGB")

    out = BytesIO()
    img.save(out, format="PNG")
    b = out.getvalue()
    if len(b) <= max_bytes:
        return b

    for quality in [85, 75, 65, 55, 45, 35]:
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        b = out.getvalue()
        if len(b) <= max_bytes:
            return b

    return b


# --------------------------------------------------
# GEMINI
# --------------------------------------------------
def gemini_client(cfg: Dict[str, str]) -> genai.Client:
    return genai.Client(api_key=cfg["gemini_api_key"])


def gemini_tags_with_usage(
    client: genai.Client, image_bytes: bytes, model_name: str
) -> Tuple[str, str, Optional[int], Optional[int], Optional[int]]:
    """
    Returns (tags_text, conf_text, input_tokens, output_tokens, total_tokens).

    v5: requests JSON output (response_mime_type + response_schema) and
    post-processes via parse_gemini_json_response() to apply the confidence
    threshold, sort, and TOP_K cap. tags_text and conf_text are matched
    comma-separated strings; tags_text remains the clean column Mark's
    macro reads (just filtered and ranked now), conf_text is the new
    parallel column with confidence numbers.
    """
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=GEMINI_PROMPT),
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
        )
    ]

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
        response_mime_type="application/json",
        response_schema=GEMINI_RESPONSE_SCHEMA,
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config,
    )

    tags_text, conf_text = parse_gemini_json_response(resp.text or "")

    in_tok: Optional[int] = None
    out_tok: Optional[int] = None
    total_tok: Optional[int] = None

    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        in_tok = getattr(usage, "prompt_token_count", None)
        out_tok = getattr(usage, "candidates_token_count", None)
        total_tok = getattr(usage, "total_token_count", None)

    return tags_text, conf_text, in_tok, out_tok, total_tok


# --------------------------------------------------
# CLARIFAI (pretrained general model)
# --------------------------------------------------
def clarifai_general_concepts(image_bytes: bytes, cfg: Dict[str, str]) -> List[Tuple[str, float]]:
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)
    metadata = (("authorization", f"Key {cfg['clarifai_pat']}"),)

    request = service_pb2.PostModelOutputsRequest(
        user_app_id=resources_pb2.UserAppIDSet(
            user_id=cfg["clarifai_general_user_id"],
            app_id=cfg["clarifai_general_app_id"],
        ),
        model_id=cfg["clarifai_general_model_id"],
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(base64=image_bytes)
                )
            )
        ],
    )
    response = stub.PostModelOutputs(request, metadata=metadata)

    if response.status.code != status_code_pb2.SUCCESS:
        raise RuntimeError(
            f"Clarifai error: code={response.status.code}, "
            f"description={response.status.description}, "
            f"details={response.status.details}"
        )

    out: List[Tuple[str, float]] = []
    concepts = (response.outputs[0].data.concepts or []) if response.outputs else []
    for c in concepts[:CL_MAX_CONCEPTS]:
        name = (c.name or "").strip().lower()
        val = float(c.value or 0.0)
        if name and val >= CL_MIN_CONF:
            out.append((name, val))
    return out


def run_clarifai_smoke_test(images: List[Path], cfg: Dict[str, str]) -> None:
    if not images:
        return

    sample = images[0]
    print("\n--- Clarifai (pretrained general) smoke test ---")
    print(f"Sample image: {sample.name}")
    print(
        "Clarifai config:",
        {
            "user_id": cfg.get("clarifai_general_user_id"),
            "app_id": cfg.get("clarifai_general_app_id"),
            "model_id": cfg.get("clarifai_general_model_id"),
        },
    )

    try:
        prepared = prepare_image_bytes_for_apis(sample.read_bytes())
        sample_concepts = clarifai_general_concepts(prepared, cfg)
        print("Clarifai sample concepts:", sample_concepts[:10])
    except Exception as e:
        print("Clarifai smoke test FAILED:")
        print(str(e))


# --------------------------------------------------
# PER-IMAGE PIPELINE (4 calls in parallel)
# --------------------------------------------------
def process_image(
    img_path: Path,
    g_client: genai.Client,
    cfg: Dict[str, str],
    clarifai_enabled: bool,
) -> Dict[str, Any]:
    """
    Run all sources (Gemini Pro, Gemini Flash, Clarifai, Local ONNX) in
    parallel for a single image. Returns the row dict ready for CSV.
    """
    raw = img_path.read_bytes()
    prepared = prepare_image_bytes_for_apis(raw)

    def call_pro():
        return gemini_tags_with_usage(g_client, prepared, GEMINI_PRO_MODEL)

    def call_flash():
        return gemini_tags_with_usage(g_client, prepared, GEMINI_FLASH_MODEL)

    def call_clarifai():
        if not clarifai_enabled:
            return []
        return clarifai_general_concepts(prepared, cfg)

    def call_local():
        if not LOCAL_AVAILABLE:
            return None
        return local_predict(raw)

    pro_result = flash_result = cla_result = loc_result = None
    pro_err = flash_err = cla_err = loc_err = None

    with ThreadPoolExecutor(max_workers=4) as pool:
        f_pro = pool.submit(call_pro)
        f_flash = pool.submit(call_flash)
        f_cla = pool.submit(call_clarifai)
        f_loc = pool.submit(call_local)

        try:
            pro_result = f_pro.result()
        except Exception as e:
            pro_err = str(e)

        try:
            flash_result = f_flash.result()
        except Exception as e:
            flash_err = str(e)

        try:
            cla_result = f_cla.result()
        except Exception as e:
            cla_err = str(e)

        try:
            loc_result = f_loc.result()
        except Exception as e:
            loc_err = str(e)

    # Unpack Gemini Pro (v5: now also returns conf_text)
    if pro_result is not None:
        gemini_out, gemini_conf, gemini_in_tokens, gemini_out_tokens, gemini_total_tokens = pro_result
    else:
        gemini_out = gemini_conf = gemini_in_tokens = gemini_out_tokens = gemini_total_tokens = None

    # Unpack Gemini Flash (v5: now also returns conf_text)
    if flash_result is not None:
        flash_tags, flash_conf, flash_in_tok, flash_out_tok, flash_total_tok = flash_result
    else:
        flash_tags = flash_conf = flash_in_tok = flash_out_tok = flash_total_tok = None

    # Unpack Clarifai
    cla_out = cla_result if cla_result is not None else []

    # Unpack Local
    if loc_result is not None:
        local_craft_str = f"{loc_result['craft']} ({loc_result['craft_confidence']:.2f})"
        local_tags_str = ", ".join(f"{t} ({c:.2f})" for t, c in loc_result["tags"])
    else:
        local_craft_str = None
        local_tags_str = None

    row: Dict[str, Any] = {
        "image_file": img_path.name,

        # Gemini Pro
        "gemini_model": GEMINI_PRO_MODEL,
        "gemini_tags": gemini_out,
        "gemini_confidences": gemini_conf,
        "gemini_input_tokens": gemini_in_tokens,
        "gemini_output_tokens": gemini_out_tokens,
        "gemini_total_tokens": gemini_total_tokens,
        "gemini_error": pro_err,

        # Gemini Flash
        "gemini_flash_model": GEMINI_FLASH_MODEL,
        "gemini_flash_tags": flash_tags,
        "gemini_flash_confidences": flash_conf,
        "gemini_flash_input_tokens": flash_in_tok,
        "gemini_flash_output_tokens": flash_out_tok,
        "gemini_flash_total_tokens": flash_total_tok,
        "gemini_flash_error": flash_err,

        # Clarifai pretrained general
        "clarifai_concepts": ", ".join([t[0] for t in cla_out]),
        "clarifai_error": cla_err,

        # Local ONNX
        "local_craft": local_craft_str,
        "local_tags": local_tags_str,
        "local_error": loc_err,

        "prepared_bytes_len": len(prepared),
    }

    return row


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    cfg = load_cfg()
    g_client = gemini_client(cfg)

    print(f"Gemini models: Pro={GEMINI_PRO_MODEL}, Flash={GEMINI_FLASH_MODEL}")
    print(f"Gemini thinking budget: {GEMINI_THINKING_BUDGET} (0 = disabled, fastest)")

    render_dir = resolve_render_dir()
    images = list_images(render_dir)
    if not images:
        print(f"No images found in: {render_dir}")
        return

    print(f"Found {len(images)} images.")

    meta_df = None
    if METADATA_CSV.exists():
        meta_df = pd.read_csv(METADATA_CSV)
        meta_df["render_file"] = meta_df["render_file"].astype(str)

    clarifai_enabled = (
        CLARIFAI_AVAILABLE
        and all(
            k in cfg
            for k in [
                "clarifai_pat",
                "clarifai_general_user_id",
                "clarifai_general_app_id",
                "clarifai_general_model_id",
            ]
        )
    )

    if not CLARIFAI_AVAILABLE:
        print("Clarifai SDK not installed; skipping.")
    elif not clarifai_enabled:
        print("Clarifai env vars (CLARIFAI_GENERAL_*) not fully set; skipping.")
    else:
        print(
            "Clarifai pretrained general model loaded:",
            cfg.get("clarifai_general_user_id"),
            "/",
            cfg.get("clarifai_general_app_id"),
            "/",
            cfg.get("clarifai_general_model_id"),
        )
        if RUN_CLARIFAI_SMOKE_TEST:
            run_clarifai_smoke_test(images, cfg)

    if LOCAL_AVAILABLE:
        print("Local ONNX inference loaded (craft + tag models).")
    else:
        print(f"Local inference not available; skipping. ({_LOCAL_IMP_ERR})")

    rows: List[Dict[str, Any]] = []

    pro_in_total = pro_out_total = pro_total_total = 0
    flash_in_total = flash_out_total = flash_total_total = 0
    pro_n = flash_n = 0

    import time
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        t_img = time.time()
        row = process_image(img_path, g_client, cfg, clarifai_enabled)
        elapsed = time.time() - t_img

        # Disable Clarifai for the rest of the run if quota hit
        if row.get("clarifai_error") and ("quota" in row["clarifai_error"].lower()
                                          or "limit" in row["clarifai_error"].lower()):
            print(f"  Clarifai quota/limit hit; disabling for rest of run.")
            clarifai_enabled = False

        # Running totals
        if row.get("gemini_total_tokens") is not None:
            pro_in_total += row["gemini_input_tokens"] or 0
            pro_out_total += row["gemini_output_tokens"] or 0
            pro_total_total += row["gemini_total_tokens"] or 0
            pro_n += 1
        if row.get("gemini_flash_total_tokens") is not None:
            flash_in_total += row["gemini_flash_input_tokens"] or 0
            flash_out_total += row["gemini_flash_output_tokens"] or 0
            flash_total_total += row["gemini_flash_total_tokens"] or 0
            flash_n += 1

        # Metadata join
        if meta_df is not None:
            m = meta_df[meta_df["render_file"] == img_path.name]
            if not m.empty:
                m0 = m.iloc[0].to_dict()
                row.update({
                    "original_file": m0.get("original_file"),
                    "normalized_file": m0.get("normalized_file"),
                    "converted": m0.get("converted"),
                    "emb_width_in": m0.get("width_in"),
                    "emb_height_in": m0.get("height_in"),
                    "stitch_count": m0.get("stitch_count"),
                    "color_count": m0.get("color_count"),
                })

        rows.append(row)

        avg = (time.time() - t0) / i
        eta_sec = avg * (len(images) - i)
        eta_min = eta_sec / 60
        print(f"[{i}/{len(images)}] {img_path.name}  ({elapsed:.1f}s, avg {avg:.1f}s/img, ETA {eta_min:.1f}min)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    total_min = (time.time() - t0) / 60
    print(f"\nSaved results to: {OUT_CSV}")
    print(f"Total runtime: {total_min:.1f} minutes")

    if pro_n > 0:
        print()
        print(f"=== Gemini Pro token summary ({pro_n} images) ===")
        print(f"  Mean input  tokens / image: {pro_in_total / pro_n:.0f}")
        print(f"  Mean output tokens / image: {pro_out_total / pro_n:.0f}")
        print(f"  Mean total  tokens / image: {pro_total_total / pro_n:.0f}")
        cost = (pro_in_total * 1.25 + pro_out_total * 10.00) / 1_000_000
        print(f"  Est. total cost (@$1.25/M in, $10/M out): ${cost:.4f}")

    if flash_n > 0:
        print()
        print(f"=== Gemini Flash token summary ({flash_n} images) ===")
        print(f"  Mean input  tokens / image: {flash_in_total / flash_n:.0f}")
        print(f"  Mean output tokens / image: {flash_out_total / flash_n:.0f}")
        print(f"  Mean total  tokens / image: {flash_total_total / flash_n:.0f}")
        cost = (flash_in_total * 0.30 + flash_out_total * 2.50) / 1_000_000
        print(f"  Est. total cost (@$0.30/M in, $2.50/M out): ${cost:.4f}")


if __name__ == "__main__":
    main()