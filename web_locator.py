#!/usr/bin/env python3
"""
web_locator.py -- Project Diamond / Diamond Collector: Web Site image locator.

v2 (260730) refinements:
  - Crawl SCOPE: by default stays UNDER the path you point it at (--scope path),
    so aiming at /product-category/newsletter-designs/ no longer wanders into the
    homepage, software, etc. Use --scope domain for whole-site.
  - EMBROIDERY-PAGE filter: only image-dense pages with an embroidery signal in
    URL/text contribute images, so the manifest holds designs, not chrome.
    Disable with --any-page.
  - Tighter JUNK filter: drops sale/gift-card/banner/category-tile images, plus a
    dimension test that removes very-wide banners when WxH is encoded in the URL.

LOCATE-ONLY (no downloads kept, no Gemini cost). Progress streams to stdout.
"""
import argparse, csv, os, re, sys, time
from urllib.parse import urljoin, urlparse, urldefrag
from html.parser import HTMLParser

try:
    import requests
except Exception:
    print("web_locator: 'requests' is required (pip install requests)", flush=True)
    sys.exit(1)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

RASTER_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff")

JUNK = ("logo", "icon", "sprite", "favicon", "pixel", "spacer", "blank",
        "avatar", "badge", "button", "arrow", "loader", "loading",
        "placeholder", "social", "share", "cart", "search", "menu",
        "close", "hamburger", "chevron", "caret", "star-", "rating",
        "payment", "visa", "mastercard", "paypal", "amex", "stripe",
        "sale", "gift-card", "gift_card", "giftcard", "banner", "hero",
        "header", "footer", "category-image", "category_image",
        "-category", "sidebar", "widget", "flag-", "ribbon", "sticker")

EMB_TERMS = ("embroider", "applique", "appliqu", "quilt", "in the hoop",
             "monogram", "redwork", "cutwork", "free standing lace",
             "freestanding", "fsl", "stipple", "mug rug", "coaster",
             "cross stitch", "stitch")
DESIGN_URL_HINTS = ("product", "/design", "shop", "collection", "categor",
                    "pattern", "embroider", "/store", "gallery")

BANNER_ASPECT = 2.8


def log(msg=""):
    print(msg, flush=True)


def largest_from_srcset(ss):
    best, bestw = None, -1
    for part in ss.split(","):
        part = part.strip()
        if not part:
            continue
        toks = part.split()
        u = toks[0]
        w = -1
        if len(toks) > 1 and toks[1].endswith("w"):
            try:
                w = int(toks[1][:-1])
            except ValueError:
                w = -1
        if w >= bestw:
            bestw, best = w, u
    return best


class Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.images = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "img":
            src = (d.get("srcset") and largest_from_srcset(d["srcset"])) \
                or d.get("src") or d.get("data-src") or d.get("data-lazy-src") \
                or d.get("data-original") or d.get("data-srcset") or ""
            if src:
                self.images.append((src.strip(), (d.get("alt") or "").strip()))
        elif tag == "source" and d.get("srcset"):
            u = largest_from_srcset(d["srcset"])
            if u:
                self.images.append((u.strip(), ""))
        elif tag == "a" and d.get("href"):
            self.links.append(d["href"].strip())


def path_prefix_of(start):
    sp = urlparse(start).path or "/"
    if not sp.endswith("/"):
        sp = sp.rsplit("/", 1)[0] + "/"
    return sp if sp else "/"


def is_html_page(href, domain, scope, prefix):
    if not href or href.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
        return False
    p = urlparse(href)
    if p.netloc and p.netloc.replace("www.", "") != domain.replace("www.", ""):
        return False
    path = p.path.lower()
    if path.endswith(RASTER_EXT + (".pdf", ".zip", ".css", ".js", ".svg", ".ico",
                                   ".mp4", ".woff", ".woff2", ".xml", ".json")):
        return False
    if scope == "path" and prefix not in ("", "/"):
        if not p.path.startswith(prefix):
            return False
    return True


def dims_from_url(url):
    m = re.search(r"(\d{2,5})[xX](\d{2,5})", url)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def keep_image(url, alt):
    low = (url + " " + alt).lower()
    if low.startswith("data:") or url.startswith("data:"):
        return False
    path = urlparse(url).path.lower()
    ext = os.path.splitext(path)[1]
    if ext and ext not in RASTER_EXT:
        return False
    if any(j in low for j in JUNK):
        return False
    d = dims_from_url(url)
    if d:
        w, h = d
        lo, hi = min(w, h), max(w, h)
        if lo and hi / lo >= BANNER_ASPECT:
            return False
    return True


def dedup_key(url):
    u, _ = urldefrag(url)
    return u.split("?")[0]


def url_is_designish(url):
    return any(h in url.lower() for h in DESIGN_URL_HINTS)


def fetch(url, timeout):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    return r, r.headers.get("Content-Type", "")


def main():
    ap = argparse.ArgumentParser(description="Diamond web image locator")
    ap.add_argument("url")
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--scope", choices=("path", "domain"), default="path")
    ap.add_argument("--min-design-images", type=int, default=3)
    ap.add_argument("--any-page", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    start = args.url.strip()
    if not re.match(r"^https?://", start):
        start = "https://" + start
    domain = urlparse(start).netloc
    if not domain:
        log("web_locator: could not parse a domain from that URL.")
        sys.exit(1)
    prefix = path_prefix_of(start)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_web_locator")
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9.-]+", "_", domain)
    out_path = args.out or os.path.join(out_dir, safe + ".csv")

    log("=" * 56)
    log(f"Web locator starting: {start}")
    log(f"Domain: {domain}   |   max pages: {args.max_pages}   |   scope: {args.scope}")
    if args.scope == "path":
        log(f"Staying under path: {prefix}")
    log(f"Embroidery-page filter: {'OFF (--any-page)' if args.any_page else 'ON'}")
    log("=" * 56)

    seen_pages, queue = set(), [start]
    pages = []
    pages_done = 0

    while queue and pages_done < args.max_pages:
        page = queue.pop(0)
        page_key, _ = urldefrag(page)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        pages_done += 1
        try:
            r, ctype = fetch(page, args.timeout)
            if "html" not in ctype.lower():
                log(f"[page {pages_done}/{args.max_pages}] skip (not html): {page}")
                continue
            ex = Extractor()
            ex.feed(r.text)
            page_imgs = [(urljoin(page, src), page, alt) for src, alt in ex.images]
            kept_here = [t for t in page_imgs if keep_image(t[0], t[2])]
            pages.append({"url": page, "imgs": kept_here, "total": len(page_imgs), "html_lower": r.text.lower()})
            nd, no = 0, 0
            for href in ex.links:
                absl = urljoin(page, href)
                lk, _ = urldefrag(absl)
                if is_html_page(absl, domain, args.scope, prefix) and lk not in seen_pages and absl not in queue:
                    if url_is_designish(absl):
                        queue.insert(0, absl); nd += 1
                    else:
                        queue.append(absl); no += 1
            log(f"[page {pages_done}/{args.max_pages}] {page}  ->  "
                f"{len(kept_here)}/{len(page_imgs)} design imgs, +{nd} design links, +{no} other")
        except Exception as e:
            log(f"[page {pages_done}/{args.max_pages}] error: {e}")
        time.sleep(args.delay)

    log("-" * 56)
    kept, seen_img = [], set()
    design_pages, skipped_pages = 0, 0
    for pg in pages:
        n = len(pg["imgs"]); total = pg["total"] or 1
        ratio = n / total
        emb_hits = sum(1 for t in EMB_TERMS if t in pg["html_lower"])
        url_hit = url_is_designish(pg["url"])
        qualifies = args.any_page or (n >= args.min_design_images and ratio >= 0.5 and (url_hit or emb_hits >= 1))
        log(f"  [{'KEEP' if qualifies else 'skip'}] {pg['url']}  ({n} design imgs, {ratio:.0%} of page, emb-terms={emb_hits})")
        if not qualifies:
            skipped_pages += 1
            continue
        design_pages += 1
        for url, page, alt in pg["imgs"]:
            k = dedup_key(url)
            if k in seen_img:
                continue
            seen_img.add(k)
            kept.append((url, page, alt))

    log("-" * 56)
    log(f"Crawled {pages_done} page(s); {design_pages} embroidery-design page(s), {skipped_pages} skipped.")

    if not kept:
        log("No candidate design images found on qualifying pages.")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["image_url", "source_page", "alt", "domain"])
        log(f"Manifest: {out_path} (0 rows)")
        return

    log(f"Found {len(kept)} candidate images")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_url", "source_page", "alt", "domain"])
        for i, (url, page, alt) in enumerate(kept, 1):
            w.writerow([url, page, alt, domain])
            log(f"  [{i}/{len(kept)}] {url}")
    log("-" * 56)
    log(f"Manifest: {out_path} ({len(kept)} rows)")
    log("Done.")


if __name__ == "__main__":
    main()