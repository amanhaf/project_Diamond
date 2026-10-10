# Architecture and design decisions

The original project separates native filesystem orchestration from embroidery parsing and machine-learning processing.

| Component | Reported responsibility |
| --- | --- |
| Rust client | Scan directories, invoke per-file workers, deserialize results, maintain cache |
| Python worker | Read embroidery instructions, render stitches, normalize images, derive identifiers, extract metadata |
| SQLite cache | Track paths and index design identifiers to recognize cross-format matches |
| Tagging integrations | Combine multimodal descriptions, generic tags, and local category predictions |
| ONNX classifier | Execute the trained EfficientNetB0 craft-category model locally |
| FastAPI conversion prototype | Isolate legacy conversion backends; VIP/ART later descoped |
| Website image locator | Extract image URLs from static HTML and process image bytes in memory |

## Identity pipeline

Rendering draws stitch segments while breaking continuity at non-stitch commands. The raster is cropped to content, converted to grayscale, resized with aspect ratio preserved, and centered on a canonical canvas. A 256-bit perceptual hash supplies the name for a UUIDv5 in a fixed application namespace.

The initial hybrid proposal included SHA-256. The implementation report records removal of the cryptographic component from design identity because small render differences across formats changed exact bytes. A byte hash could still serve file-integrity or storage-deduplication purposes separately.

Exact UUID derivation is deterministic only when normalization, hash inputs, and namespace are identical. It does not establish universal format or rotation invariance. Hamming-distance matching is a separate lookup operation whose clustering, tie-breaking, and representative-selection policy needs explicit definition.

## Processing contract

The Rust client invokes a Python worker per file and receives one structured success or error result. Reported success fields include the identifier, perceptual hash, file/render paths, dimensions, stitch count, and color count. Per-file failures can be logged without terminating an entire batch.

The path-keyed cache answers whether a file has already been processed. An identifier index answers whether artwork represented by that identifier appeared under another path. Those are different checks.

## Tagging and inference

The reported tagging cycle invokes Gemini Pro, Gemini Flash, Clarifai, and local ONNX inference concurrently. Outputs are normalized into tags; filtering addresses some inaccurate generated labels. The ONNX model requires the preprocessing used during training, including OpenCV resizing and raw floating-point pixel values in the 0–255 range as documented for this export.

Cloud catalog services, customer-facing search, catalog installation, and partner product integration were outside the author's project scope. The public documentation should not imply those services were deployed.
