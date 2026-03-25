#!/usr/bin/env python3
"""Batch-test classifiers on a directory of bird crop images.

Runs one or all classifier backends against every image in test-photos/
(or a directory you choose) and produces:

  <output-dir>/
    annotated/          — copies of each image with species label overlay
    inference_log.json  — per-image JSON with top-5 candidates per backend
    summary.csv         — one row per image per backend

The overlay is colour-coded by confidence:
    green  ≥ 0.70   yellow ≥ 0.40   red < 0.40

Usage:
    # Compare all three backends on test-photos/
    python scripts/test_photos.py

    # Single backend
    python scripts/test_photos.py --backend bioclip
    python scripts/test_photos.py --backend nabirds
    python scripts/test_photos.py --backend tfhub

    # Custom dirs
    python scripts/test_photos.py --input-dir my_crops/ --output-dir results/

    # Lower confidence bar to see more candidates
    python scripts/test_photos.py --min-confidence 0.2

    # Skip writing annotated images (faster)
    python scripts/test_photos.py --no-annotate

Install deps for each backend before first use:
    tfhub   : pip install tensorflow tensorflow-hub requests
    bioclip : pip install open-clip-torch pillow
    nabirds : pip install transformers torch pillow
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap: point imports at bird-detector/ ────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIRD_DETECTOR = PROJECT_ROOT / "bird-detector"
sys.path.insert(0, str(BIRD_DETECTOR))

os.environ.setdefault(
    "SPECIES_LIST_PATH", str(BIRD_DETECTOR / "species_list.txt")
)
os.environ.setdefault("DATA_DIR", str(PROJECT_ROOT / "data"))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("test-photos")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ALL_BACKENDS = ("tfhub", "bioclip", "nabirds", "efficientnet")

_COLOUR_HIGH = (34, 197, 94)
_COLOUR_MED  = (234, 179, 8)
_COLOUR_LOW  = (239, 68, 68)
_TEXT_DARK   = (15, 15, 15)
_TEXT_LIGHT  = (245, 245, 245)


def _conf_colour(conf: float) -> tuple[int, int, int]:
    if conf >= 0.70:
        return _COLOUR_HIGH
    if conf >= 0.40:
        return _COLOUR_MED
    return _COLOUR_LOW


# ── Classifier loading ────────────────────────────────────────────────────────

def _load_classifier(backend: str, species_list: Path):
    """Instantiate and load the requested classifier. Returns None on failure."""
    try:
        if backend == "tfhub":
            from classifier import SpeciesClassifier
            clf = SpeciesClassifier(species_list_path=species_list)
        elif backend == "bioclip":
            from classifier_bioclip import BioCLIPClassifier
            clf = BioCLIPClassifier(species_list_path=species_list)
        elif backend == "nabirds":
            from classifier_nabirds import NABirdsClassifier
            clf = NABirdsClassifier(species_list_path=species_list)
        elif backend == "efficientnet":
            import os
            from pathlib import Path
            from classifier_efficientnet import EfficientNetClassifier
            model_path = Path(os.environ.get("EFFICIENTNET_MODEL_PATH", "data/models/feeder_birds.pt"))
            clf = EfficientNetClassifier(model_path=model_path)
        else:
            log.error("Unknown backend: %r", backend)
            return None
        clf.load()
        return clf
    except ImportError as e:
        log.error(
            "Backend %r missing deps — %s\n"
            "  tfhub  : pip install tensorflow tensorflow-hub requests\n"
            "  bioclip: pip install open-clip-torch pillow\n"
            "  nabirds: pip install transformers torch pillow",
            backend, e,
        )
        return None
    except Exception:
        log.exception("Failed to load backend %r", backend)
        return None


# ── Classification ────────────────────────────────────────────────────────────

def _classify_image(clf, image_path: Path, top_n: int) -> list[tuple[str, float]]:
    """Read image with OpenCV and run classifier. Returns top-N results."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        log.warning("Could not read image: %s", image_path)
        return []
    return clf.classify(img, top_n=top_n)


# ── Annotation ────────────────────────────────────────────────────────────────

_MIN_DISPLAY_WIDTH = 400   # upscale crops smaller than this before annotating
_FONT_SIZE_MAIN    = 16
_FONT_SIZE_SMALL   = 13


def _annotate(
    image_path: Path,
    results_by_backend: dict[str, list[tuple[str, float]]],
    output_path: Path,
    top_n: int,
) -> None:
    """Draw prediction panel below the image and save to output_path."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # Upscale tiny crops so the output is always readable.
    if W < _MIN_DISPLAY_WIDTH:
        scale = _MIN_DISPLAY_WIDTH / W
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        W, H = img.size

    font_main = font_small = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        if Path(fp).exists():
            try:
                font_main  = ImageFont.truetype(fp, _FONT_SIZE_MAIN)
                font_small = ImageFont.truetype(fp, _FONT_SIZE_SMALL)
            except OSError:
                pass
            break
    if font_main is None:
        font_main = font_small = ImageFont.load_default()

    # Measure label panel height before compositing.
    first_results = next(iter(results_by_backend.values()), [])
    runners = first_results[1:top_n]
    row_h_main  = _FONT_SIZE_MAIN  + 8
    row_h_small = _FONT_SIZE_SMALL + 6
    n_main_rows = sum(1 for r in results_by_backend.values() if r)
    panel_h = n_main_rows * row_h_main + len(runners) * row_h_small + 8

    # Composite: original image on top, label panel below.
    canvas = Image.new("RGB", (W, H + panel_h), (20, 20, 20))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # One row per backend.
    y = H + 4
    for backend, results in results_by_backend.items():
        if not results:
            continue
        species, conf = results[0]
        bg = _conf_colour(conf)
        lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        fg = _TEXT_DARK if lum > 140 else _TEXT_LIGHT

        draw.rectangle([(0, y - 2), (W, y + row_h_main - 2)], fill=bg)
        text = f"[{backend}]  {species}  {conf:.0%}"
        draw.text((6, y), text, fill=fg, font=font_main)
        y += row_h_main

    # Runner-up candidates below the backend rows.
    for i, (sp, cf) in enumerate(runners):
        draw.text(
            (6, y),
            f"#{i + 2}  {sp}  {cf:.0%}",
            fill=(200, 200, 200),
            font=font_small,
        )
        y += row_h_small

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


# ── Summary printing ──────────────────────────────────────────────────────────

def _print_summary(
    records: list[dict],
    backends: list[str],
    min_conf: float,
) -> None:
    width = 80
    print(f"\n{'─' * width}")
    print(f"  RESULTS   ({len(records)} images,  min_confidence={min_conf:.2f})")
    print(f"{'─' * width}")

    col_w = 28
    header = f"  {'Image':<24}" + "".join(f"  {b:<{col_w}}" for b in backends)
    print(header)
    print(f"  {'─' * (24 + (col_w + 2) * len(backends))}")

    for r in records:
        row = f"  {r['filename']:<24}"
        for b in backends:
            res = r["backends"].get(b)
            if res:
                cell = f"{res['species']} ({res['confidence']:.0%})"
            else:
                cell = "—"
            row += f"  {cell:<{col_w}}"
        print(row)

    print(f"{'─' * width}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    species_list = Path(args.species_list)
    min_conf = args.min_confidence
    top_n = args.top_n

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)
    if not species_list.exists():
        log.warning("Species list not found: %s — classifiers will use all labels", species_list)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        log.error("No images found in %s", input_dir)
        sys.exit(1)

    backends = ALL_BACKENDS if args.backend == "all" else (args.backend,)
    log.info("Backends: %s", ", ".join(backends))
    log.info("Found %d image(s) in %s", len(images), input_dir)

    # Load classifiers
    classifiers: dict[str, object] = {}
    for b in backends:
        log.info("Loading backend: %s …", b)
        clf = _load_classifier(b, species_list)
        if clf is not None:
            classifiers[b] = clf

    if not classifiers:
        log.error("No classifiers loaded successfully — exiting")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    run_ts = datetime.now(timezone.utc).astimezone().isoformat()

    log_records: list[dict] = []
    csv_rows: list[dict] = []

    for i, img_path in enumerate(images, 1):
        log.info("[%d/%d] %s", i, len(images), img_path.name)
        results_by_backend: dict[str, list[tuple[str, float]]] = {}

        for b, clf in classifiers.items():
            results = _classify_image(clf, img_path, top_n)
            results_by_backend[b] = results
            if results:
                log.info("  %-10s → %-38s %.2f", b, results[0][0], results[0][1])
            else:
                log.info("  %-10s → (no result)", b)

        record: dict = {
            "filename": img_path.name,
            "timestamp": run_ts,
            "backends": {
                b: {
                    "species": res[0][0],
                    "confidence": res[0][1],
                    "candidates": [{"species": s, "confidence": c} for s, c in res],
                }
                if res else None
                for b, res in results_by_backend.items()
            },
        }
        log_records.append(record)

        for b, res in results_by_backend.items():
            if res:
                sp, cf = res[0]
                csv_rows.append({
                    "filename": img_path.name,
                    "backend": b,
                    "species": sp,
                    "confidence": f"{cf:.4f}",
                    "above_threshold": str(cf >= min_conf),
                    "timestamp": run_ts,
                })

        if not args.no_annotate and results_by_backend:
            try:
                _annotate(img_path, results_by_backend, annotated_dir / img_path.name, top_n)
            except Exception:
                log.exception("Failed to annotate %s", img_path.name)

    # Write outputs
    json_path = output_dir / "inference_log.json"
    json_path.write_text(json.dumps(log_records, indent=2))
    log.info("Wrote %s", json_path)

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "backend", "species", "confidence", "above_threshold", "timestamp"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("Wrote %s", csv_path)

    _print_summary(log_records, list(classifiers.keys()), min_conf)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-classify bird crop images and compare backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend",
        choices=["tfhub", "bioclip", "nabirds", "efficientnet", "all"],
        default="efficientnet",
        help="Classifier backend to use (default: efficientnet)",
    )
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "test-photos"),
        metavar="DIR",
        help="Directory of input images (default: test-photos/)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "test-photos-results"),
        metavar="DIR",
        help="Where to write annotated images + logs (default: test-photos-results/)",
    )
    parser.add_argument(
        "--species-list",
        default=str(BIRD_DETECTOR / "species_list.txt"),
        metavar="FILE",
        help="Species allowlist (default: bird-detector/species_list.txt)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        metavar="N",
        help="Minimum confidence to flag as a detection (default: 0.3)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of candidate predictions per image (default: 5)",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Skip writing annotated output images",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
