# Project Diamond

**Content-derived identification and cataloguing for proprietary embroidery design files.**

Project Diamond is a proof-of-concept pipeline — developed in partnership with **Patrick Sebastian Technologies (PST)** — that makes large collections of machine-embroidery files searchable. Embroidery designs ship in proprietary, machine-specific formats (PES, DST, EXP, VP3, JEF, HUS, XXX, …) that operating systems cannot preview or index, and that are usually named with cryptic 8-character codes. Owners often hold thousands of these files with no practical way to organise or find them.

Project Diamond solves this by deriving a stable identity and searchable tags for every design directly from its **visual content** — independent of file name, format, or resolution.

## How it works

The pipeline has two halves:

**1. Content-derived UUID (Rust + Python)**
Each embroidery file is rendered to a normalised image, reduced to a perceptual hash, and mapped to a deterministic UUIDv5. Because the identifier is derived from the rendered design rather than the raw bytes, the *same* design exported to different formats — or at different resolutions — resolves to the *same* UUID. This enables cross-format de-duplication and lets the catalogue keep one canonical record per design.

**2. Search-tag generation (Python + Gemini)**
The rendered preview is sent to a vision-language model (Google Gemini Flash) to generate descriptive and thematic search keywords. Vision models reliably recognise *subjects* but are weaker at *construction technique* (appliqué, quilting, monogram lettering, etc.), so a layer of deterministic post-processing rules corrects and completes the tags using measurable file properties — stitch density, finished dimensions, colour count, and filename conventions.

## What it achieves

- **One canonical ID per design.** The same artwork exported to different formats or resolutions collapses to a single UUID, enabling reliable cross-format de-duplication across a large corpus.
- **Searchable, category-aware tags.** Vision-model keywords are augmented with deterministic rules that recover construction categories — appliqué, quilting, monogram lettering — that the model alone tends to miss, including reading monogram letters directly from filenames where the rendered script is too ornate to read reliably.

## Repository layout

| Path | Purpose |
|---|---|
| `Client_Rust/` | Rust scanner and content-derived UUID generation (invokes the Python worker) |
| `Worker_python/` | Renders embroidery files to images and extracts metadata (`pyembroidery`) |
| `converter_service/` | FastAPI service that converts legacy formats (VIP / ART) into a readable format |
| `Image_Recognition/` | Gemini tagging pipeline and the deterministic tag-correction rules |
| `UUID/` | Standalone UUID-generation utilities |
| `dashboard/` | Lightweight results/review dashboard |

## Tech stack

- **Rust** — file scanning and perceptual-hash UUID generation
- **Python** — rendering (`pyembroidery`, `Pillow`), perceptual hashing (`ImageHash`), tag rules
- **Google Gemini** — vision-language tagging
- **FastAPI** — local format-conversion service

## Getting started

```bash
# Python environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Provide a Gemini API key via a .env file or environment variable (never commit it)
export GEMINI_API_KEY="your-key-here"

# Build the Rust client
cd Client_Rust && cargo build --release
```

Point the pipeline at a folder of embroidery files; it renders each design, generates its UUID, groups cross-format duplicates, and produces search tags.

## Notes

- This repository contains **source code only**. Proprietary embroidery design files, rendered previews, ground-truth data, and trained model artifacts are intentionally excluded (see [`.gitignore`](.gitignore)).
- Built as a Master's project proof-of-concept in partnership with Patrick Sebastian Technologies (PST).

## Author

**Aman Hafez**
M.S. in Computer Science & Engineering, The Ohio State University
[LinkedIn](https://www.linkedin.com/in/amanhafez/) · [GitHub](https://github.com/amanhaf)

## Project report and evaluation

See the [portfolio case study](docs/portfolio/README.md) for the report-based architecture, my contributions, reported runtime measurements, and evaluation limitations. The case study records historical findings and distinguishes unresolved report details from verified current implementation behavior.
