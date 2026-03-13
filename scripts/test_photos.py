#!/usr/bin/env python3
"""Batch-test the classifier on a directory of still images.

Iterates every image in test-photos/ (or a directory you choose), runs the
two-stage classifier (local TFHub → iNaturalist API), and produces:

  <output-dir>/
    annotated/          — copies of each image with a species label overlay
    inference_log.json  — per-image JSON records with top-5 candidates
    summary.csv         — one row per image: filename, top species, confidence, …

The overlay shows the top prediction with a confidence-coloured box:
    green  ≥ 0.70  (high confidence)
    yellow ≥ 0.40  (medium)
    red    <  0.40  (low / uncertain)

NOTE: the TFHub model is a whole-image classifier, not an object detector, so
there are no true bounding boxes — the label strip is drawn across the full
image width.  For real crop-level boxes you need a detector such as Frigate/
YOLO upstream.

Usage:
    # Default: test-photos/ → test-photos-results/
    python scripts/test_photos.py

    # Custom directories
    python scripts/test_photos.py --input-dir my_photos/ --output-dir results/

    # Skip iNat (fast, no network)
    python scripts/test_photos.py --skip-inat

    # Lower confidence threshold to see more candidates
    python scripts/test_photos.py --min-confidence 0.2

    # Top-N candidates to include in the log
    python scripts/test_photos.py --top-n 5

    # Don't write annotated images
    python scripts/test_photos.py --no-annotate
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("DATA_DIR", str(PROJECT_ROOT / "data"))
os.environ.setdefault(
    "SPECIES_LIST_PATH",
    str(PROJECT_ROOT / "bird-classifier" / "species_list.txt"),
)
sys.path.insert(0, str(PROJECT_ROOT / "bird-classifier"))

from config import (  # noqa: E402
    LOCAL_MODEL_CONFIDENCE_THRESHOLD,
    MIN_CONFIDENCE_LOG,
)
from inat_client import INatClient  # noqa: E402
from local_model import LocalBirdModel  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("test-photos")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Confidence colour thresholds (R, G, B)
_COLOUR_HIGH = (34, 197, 94)    # green
_COLOUR_MED  = (234, 179, 8)    # yellow
_COLOUR_LOW  = (239, 68, 68)    # red
_TEXT_DARK   = (15, 15, 15)
_TEXT_LIGHT  = (245, 245, 245)


def _conf_colour(conf: float) -> tuple[int, int, int]:
    if conf >= 0.70:
        return _COLOUR_HIGH
    if conf >= 0.40:
        return _COLOUR_MED
    return _COLOUR_LOW


# ── Classification ─────────────────────────────────────────────────────────────


def classify_image(
    image_path: Path,
    local_model: LocalBirdModel | None,
    inat_client: INatClient | None,
    local_threshold: float,
    top_n: int,
) -> dict:
    """Return a dict with top predictions for one image.

    Keys: top_species, top_scientific, top_confidence, classifier, candidates.
    """
    local_results: list[tuple[str, float]] = []
    inat_results = []

    # Stage 1 — local TFHub model
    if local_model is not None:
        local_results = local_model.classify(image_path)
        if local_results:
            top_name, top_conf = local_results[0]
            if top_conf >= local_threshold:
                return {
                    "top_species": top_name,
                    "top_scientific": "",
                    "top_confidence": top_conf,
                    "classifier": "tfhub_birds",
                    "candidates": [
                        {"species": n, "scientific": "", "confidence": c, "classifier": "tfhub_birds"}
                        for n, c in local_results[:top_n]
                    ],
                }

    # Stage 2 — iNaturalist API
    if inat_client is not None:
        inat_results = inat_client.classify(image_path)
        if inat_results:
            top = inat_results[0]
            return {
                "top_species": top.species_common,
                "top_scientific": top.species_scientific,
                "top_confidence": top.confidence,
                "classifier": "inaturalist",
                "candidates": [
                    {
                        "species": r.species_common,
                        "scientific": r.species_scientific,
                        "confidence": r.confidence,
                        "classifier": "inaturalist",
                    }
                    for r in inat_results[:top_n]
                ],
            }

    # Fallback: best local result even if below threshold
    if local_results:
        top_name, top_conf = local_results[0]
        return {
            "top_species": top_name,
            "top_scientific": "",
            "top_confidence": top_conf,
            "classifier": "tfhub_birds",
            "candidates": [
                {"species": n, "scientific": "", "confidence": c, "classifier": "tfhub_birds"}
                for n, c in local_results[:top_n]
            ],
        }

    return {}


# ── Annotation ─────────────────────────────────────────────────────────────────


def annotate_image(
    image_path: Path,
    prediction: dict,
    output_path: Path,
    top_n: int,
) -> None:
    """Draw a species label strip on the image and save to output_path."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    species  = prediction.get("top_species", "Unknown")
    sci      = prediction.get("top_scientific", "")
    conf     = prediction.get("top_confidence", 0.0)
    clf      = prediction.get("classifier", "")
    bg_color = _conf_colour(conf)

    # ── Pick fonts ─────────────────────────────────────────────────────────────
    # Attempt to load a system TrueType font; fall back to PIL's built-in bitmap.
    font_main = font_small = None
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for fp in candidate_paths:
        if Path(fp).exists():
            try:
                font_main  = ImageFont.truetype(fp, max(14, H // 24))
                font_small = ImageFont.truetype(fp, max(11, H // 36))
            except OSError:
                pass
            break

    if font_main is None:
        font_main  = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # ── Main label strip (top of image) ────────────────────────────────────────
    main_text = f"{species}  {conf:.0%}"
    if sci:
        main_text += f"  |  {sci}"

    bbox = draw.textbbox((0, 0), main_text, font=font_main)
    text_h = bbox[3] - bbox[1]
    strip_h = text_h + 12
    draw.rectangle([(0, 0), (W, strip_h)], fill=bg_color)

    # Choose text colour for legibility
    lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    text_color = _TEXT_DARK if lum > 140 else _TEXT_LIGHT

    draw.text((6, 6), main_text, fill=text_color, font=font_main)

    # ── Confidence badge (top-right corner) ────────────────────────────────────
    badge = f"[{clf}]"
    bb = draw.textbbox((0, 0), badge, font=font_small)
    bw = bb[2] - bb[0]
    draw.rectangle([(W - bw - 10, 0), (W, strip_h)], fill=bg_color)
    draw.text((W - bw - 5, 6), badge, fill=text_color, font=font_small)

    # ── Full-image border (same colour as label strip) ─────────────────────────
    border = max(3, H // 160)
    draw.rectangle([(0, 0), (W - 1, H - 1)], outline=bg_color, width=border)

    # ── Candidate list (bottom strip) ─────────────────────────────────────────
    candidates = prediction.get("candidates", [])[1:top_n]  # skip the top (already shown)
    if candidates:
        row_h = max(16, H // 32)
        panel_h = row_h * len(candidates) + 6
        draw.rectangle(
            [(0, H - panel_h), (W, H)],
            fill=(20, 20, 20, 200),
        )
        for i, cand in enumerate(candidates):
            y = H - panel_h + 4 + i * row_h
            cand_text = (
                f"  #{i + 2}  {cand['species']}  {cand['confidence']:.0%}"
            )
            draw.text((4, y), cand_text, fill=(220, 220, 220), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=92)


# ── Main ───────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    annotated_dir = output_dir / "annotated"
    min_conf = args.min_confidence

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        log.error("No images found in %s", input_dir)
        sys.exit(1)

    log.info("Found %d image(s) in %s", len(images), input_dir)

    # ── Load models ────────────────────────────────────────────────────────────
    local_model: LocalBirdModel | None = None
    if not args.skip_local:
        local_model = LocalBirdModel()
        try:
            local_model.load()
        except Exception:
            log.exception("Failed to load local TFHub model — falling back to iNat only")
            local_model = None

    inat_client: INatClient | None = None if args.skip_inat else INatClient()

    if local_model is None and inat_client is None:
        log.error("Both classifiers disabled — nothing to do")
        sys.exit(1)

    # ── Process images ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    log_records: list[dict] = []
    csv_rows: list[dict] = []
    run_ts = datetime.now(timezone.utc).astimezone().isoformat()

    for i, img_path in enumerate(images, 1):
        log.info("[%d/%d] %s", i, len(images), img_path.name)

        prediction = classify_image(
            img_path, local_model, inat_client,
            LOCAL_MODEL_CONFIDENCE_THRESHOLD, args.top_n,
        )

        if not prediction:
            log.warning("  → no result")
            prediction = {
                "top_species": "Unknown",
                "top_scientific": "",
                "top_confidence": 0.0,
                "classifier": "none",
                "candidates": [],
            }

        top_conf = prediction.get("top_confidence", 0.0)
        top_species = prediction.get("top_species", "Unknown")
        log.info(
            "  → %-40s conf=%.2f  [%s]",
            top_species, top_conf, prediction.get("classifier", ""),
        )

        record = {
            "filename": img_path.name,
            "timestamp": run_ts,
            **prediction,
        }
        log_records.append(record)

        above_threshold = top_conf >= min_conf
        if not above_threshold:
            log.info("  → below --min-confidence %.2f", min_conf)

        csv_rows.append({
            "filename":       img_path.name,
            "top_species":    top_species,
            "top_scientific": prediction.get("top_scientific", ""),
            "confidence":     f"{top_conf:.4f}",
            "classifier":     prediction.get("classifier", ""),
            "above_threshold": str(above_threshold),
            "timestamp":      run_ts,
        })

        if not args.no_annotate:
            out_img = annotated_dir / img_path.name
            try:
                annotate_image(img_path, prediction, out_img, args.top_n)
                log.info("  → annotated → %s", out_img.relative_to(PROJECT_ROOT))
            except Exception:
                log.exception("  Failed to annotate %s", img_path.name)

    # ── Write outputs ──────────────────────────────────────────────────────────
    json_path = output_dir / "inference_log.json"
    json_path.write_text(json.dumps(log_records, indent=2))
    log.info("Wrote %s", json_path.relative_to(PROJECT_ROOT))

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "filename", "top_species", "top_scientific",
            "confidence", "classifier", "above_threshold", "timestamp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("Wrote %s", csv_path.relative_to(PROJECT_ROOT))

    _print_summary(log_records, min_conf)


def _print_summary(records: list[dict], min_conf: float) -> None:
    from collections import Counter

    width = 72
    above = [r for r in records if r.get("top_confidence", 0.0) >= min_conf]

    print(f"\n{'─' * width}")
    print(f"  PHOTO TEST SUMMARY   ({len(records)} images,  min_confidence={min_conf:.2f})")
    print(f"{'─' * width}")

    if not above:
        print(f"  No images above confidence threshold ({min_conf:.2f}).\n")
        return

    counts: Counter = Counter(r["top_species"] for r in above)
    avg_conf = {
        s: sum(r["top_confidence"] for r in above if r["top_species"] == s) / c
        for s, c in counts.items()
    }

    print(f"  {'Filename':<28} {'Species':<30} {'Conf':>6}  Classifier")
    print(f"  {'─' * 68}")
    for r in records:
        conf = r.get("top_confidence", 0.0)
        flag = "" if conf >= min_conf else "  ← below threshold"
        print(
            f"  {r['filename']:<28} {r.get('top_species','Unknown'):<30} "
            f"{conf:>5.0%}  {r.get('classifier','')}{flag}"
        )

    print()
    print(f"  {'Species':<40} {'Count':>5}  {'Avg Conf':>8}")
    print(f"  {'─' * 58}")
    for species, count in counts.most_common():
        print(f"  {species:<40} {count:>5}  {avg_conf[species]:>8.2%}")
    print(f"{'─' * width}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-classify images in a directory and produce annotated output + logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE_LOG,
        metavar="N",
        help=f"Minimum confidence to flag as a detection (default: {MIN_CONFIDENCE_LOG})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of candidate predictions to include in the log (default: 5)",
    )
    parser.add_argument(
        "--skip-inat",
        action="store_true",
        help="Skip iNaturalist API — local TFHub model only (faster, no network)",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip local TFHub model — iNaturalist API only",
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
