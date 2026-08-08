import time
from typing import Any, Dict, List, Tuple, Optional
import requests


def _clean(s: str) -> str:
    return (s or "").strip().lower()


def _normalize_endpoint(endpoint: str) -> str:
    return (endpoint or "").strip().rstrip("/")


def _normalize_key(key: str) -> str:
    # Remove whitespace + BOM if any
    return (key or "").strip().replace("\ufeff", "")


def _post_with_retries(
    url: str,
    headers: Dict[str, str],
    params: Dict[str, str],
    data: bytes,
    max_retries: int = 4,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Retry/backoff for transient failures (429/5xx).
    Raises RuntimeError for non-200 responses.
    """
    backoff = 1.0
    last_resp = None

    for attempt in range(max_retries + 1):
        resp = requests.post(url, headers=headers, params=params, data=data, timeout=timeout)
        last_resp = resp

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries:
                break
            retry_after = resp.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after else backoff
            time.sleep(sleep_s)
            backoff *= 2
            continue

        try:
            detail = resp.json()
        except Exception:
            detail = resp.text

        raise RuntimeError(f"Azure Vision error {resp.status_code}: {detail}")

    # Retries exhausted
    if last_resp is None:
        raise RuntimeError("Azure Vision error: no response")

    try:
        detail = last_resp.json()
    except Exception:
        detail = last_resp.text

    raise RuntimeError(f"Azure Vision retry exhausted. Last error {last_resp.status_code}: {detail}")


# -----------------------------
# Azure Analyze (v3.2 then v4.0 fallback)
# -----------------------------
def azure_analyze_image(
    image_bytes: bytes,
    endpoint: str,
    key: str,
    visual_features: Optional[List[str]] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Try Computer Vision v3.2 first:
      POST {endpoint}/vision/v3.2/analyze?visualFeatures=...
    If that fails with 401/404, fall back to Image Analysis v4.0:
      POST {endpoint}/computervision/imageanalysis:analyze?api-version=...&features=...
    Returns a standardized dict containing keys: tags, objects, description(captions)
    """
    endpoint = _normalize_endpoint(endpoint)
    key = _normalize_key(key)

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
    }

    # ---- Attempt 1: v3.2
    if visual_features is None:
        visual_features = ["Tags", "Categories", "Objects"]

    url_v32 = f"{endpoint}/vision/v3.2/analyze"
    params_v32 = {"visualFeatures": ",".join(visual_features), "language": language}

    try:
        return _post_with_retries(url_v32, headers, params_v32, image_bytes)
    except RuntimeError as e:
        msg = str(e)
        # Only fall back on auth/route mismatch (common when resource expects v4)
        if ("Azure Vision error 401" not in msg) and ("Azure Vision error 404" not in msg):
            raise

    # ---- Attempt 2: v4.0 Image Analysis
    url_v40 = f"{endpoint}/computervision/imageanalysis:analyze"
    params_v40 = {
        "api-version": "2023-10-01",
        "features": "tags,objects,caption",
        "language": language,
    }

    resp = _post_with_retries(url_v40, headers, params_v40, image_bytes)

    # Standardize to resemble v3.2 output
    standardized: Dict[str, Any] = {}

    # v4 tags: resp["tagsResult"]["values"] = [{"name":..., "confidence":...}, ...]
    tags = []
    for t in (resp.get("tagsResult", {}).get("values") or []):
        tags.append({"name": t.get("name"), "confidence": t.get("confidence")})
    standardized["tags"] = tags

    # v4 objects: resp["objectsResult"]["values"] = [{"confidence":..., "boundingBox":..., "tags":[{"name":...}]}]
    objects = []
    for o in (resp.get("objectsResult", {}).get("values") or []):
        obj_name = None
        tags_list = o.get("tags") or []
        if tags_list:
            obj_name = tags_list[0].get("name")
        objects.append({"object": obj_name, "confidence": o.get("confidence")})
    standardized["objects"] = objects

    # v4 caption: resp["captionResult"] = {"text":..., "confidence":...}
    cap = resp.get("captionResult")
    if cap and cap.get("text"):
        standardized["description"] = {
            "captions": [{"text": cap.get("text"), "confidence": cap.get("confidence")}]
        }
    else:
        standardized["description"] = {"captions": []}

    return standardized


# -----------------------------
# Azure Describe (v3.2 then v4.0 fallback)
# -----------------------------
def azure_describe_image(
    image_bytes: bytes,
    endpoint: str,
    key: str,
    max_candidates: int = 1,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Try v3.2 describe, fall back to v4.0 caption.
    Returns dict with description.captions list.
    """
    endpoint = _normalize_endpoint(endpoint)
    key = _normalize_key(key)

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream",
    }

    # ---- Attempt 1: v3.2 describe
    url_v32 = f"{endpoint}/vision/v3.2/describe"
    params_v32 = {"maxCandidates": str(max_candidates), "language": language}

    try:
        return _post_with_retries(url_v32, headers, params_v32, image_bytes)
    except RuntimeError as e:
        msg = str(e)
        if ("Azure Vision error 401" not in msg) and ("Azure Vision error 404" not in msg):
            raise

    # ---- Attempt 2: v4.0 caption
    url_v40 = f"{endpoint}/computervision/imageanalysis:analyze"
    params_v40 = {"api-version": "2023-10-01", "features": "caption", "language": language}
    resp = _post_with_retries(url_v40, headers, params_v40, image_bytes)

    cap = resp.get("captionResult")
    if cap and cap.get("text"):
        return {"description": {"captions": [{"text": cap.get("text"), "confidence": cap.get("confidence")}]}}

    return {"description": {"captions": []}}


# -----------------------------
# Helpers to extract keywords/caption
# -----------------------------
def azure_keywords_from_analyze(
    analyze_json: Dict[str, Any],
    min_conf: float = 0.2,
    max_tags: int = 25,
) -> List[Tuple[str, float]]:
    """
    Converts Analyze JSON (v3.2 or standardized v4.0) to [(keyword, confidence)].
    """
    out: List[Tuple[str, float]] = []

    # tags
    for t in (analyze_json.get("tags") or [])[: max_tags * 2]:
        name = _clean(t.get("name", ""))
        conf = float(t.get("confidence") or 0.0)
        if name and conf >= min_conf:
            out.append((name, conf))

    # categories (v3.2 only)
    for c in analyze_json.get("categories") or []:
        name = _clean(c.get("name", ""))
        score = float(c.get("score") or 0.0)
        if name and score >= min_conf:
            out.append((name, score))

    # objects
    for o in analyze_json.get("objects") or []:
        name = _clean(o.get("object", ""))
        conf = float(o.get("confidence") or 0.0)
        if name and conf >= min_conf:
            out.append((name, conf))

    # dedup (keep best confidence)
    best: Dict[str, float] = {}
    order: List[str] = []
    for name, conf in out:
        if name not in best:
            best[name] = conf
            order.append(name)
        else:
            if conf > best[name]:
                best[name] = conf

    final = [(n, best[n]) for n in order]
    final.sort(key=lambda x: x[1], reverse=True)
    return final[:max_tags]


def azure_caption_from_describe(describe_json: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    desc = describe_json.get("description") or {}
    caps = desc.get("captions") or []
    if not caps:
        return None
    top = caps[0]
    text = _clean(top.get("text", ""))
    conf = float(top.get("confidence") or 0.0)
    return (text, conf) if text else None