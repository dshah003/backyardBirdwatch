#!/usr/bin/env python3
"""Test the bird detection pipeline on local video files or image snapshots.

Extracts frames from video, runs them through the classifier pipeline
(local TFHub model → iNaturalist API), and produces a detection report.
No MQTT, Frigate, or Docker required.

Usage:
    # Test a video file (1 frame/sec by default)
    python scripts/test_pipeline.py path/to/video.mp4

    # Multiple videos, faster frame rate
    python scripts/test_pipeline.py *.mp4 --fps 2

    # Only run the local model (fast, no API calls)
    python scripts/test_pipeline.py video.mp4 --skip-inat

    # Only use iNaturalist API
    python scripts/test_pipeline.py video.mp4 --skip-local

    # A directory of JPEG snapshots (e.g. existing Frigate clips)
    python scripts/test_pipeline.py frigate/storage/clips/

    # Lower the confidence bar to see more candidates
    python scripts/test_pipeline.py video.mp4 --min-confidence 0.2

    # Don't write results to the database
    python scripts/test_pipeline.py video.mp4 --no-log
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap: load .env and fix sys.path before importing project modules ─────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(env_path: Path) -> None:
    """Parse .env and populate os.environ (existing vars are not overwritten)."""
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

# Point the classifier modules at local paths so they work outside Docker.
os.environ.setdefault("DATA_DIR", str(PROJECT_ROOT / "data"))
os.environ.setdefault(
    "SPECIES_LIST_PATH",
    str(PROJECT_ROOT / "bird-classifier" / "species_list.txt"),
)

sys.path.insert(0, str(PROJECT_ROOT / "bird-classifier"))

# Import after env setup so config.py reads the correct values.
from config import (  # noqa: E402
    LOCAL_MODEL_CONFIDENCE_THRESHOLD,
    MIN_CONFIDENCE_LOG,
)
from inat_client import INatClient  # noqa: E402
from local_model import LocalBirdModel  # noqa: E402
from logger import DetectionLogger, DetectionRecord  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("test-pipeline")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".ts"}


# ── Frame extraction ───────────────────────────────────────────────────────────


def extract_frames(video_path: Path, fps: float, output_dir: Path) -> list[Path]:
    """Use ffmpeg to extract frames from a video at the given fps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high-quality JPEG (scale 1–31, lower = better)
        str(output_dir / "frame_%06d.jpg"),
        "-hide_banner", "-loglevel", "error",
    ]
    log.info("Extracting frames at %.2f fps from %s …", fps, video_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg failed:\n%s", result.stderr)
        return []
    frames = sorted(output_dir.glob("frame_*.jpg"))
    log.info("  → %d frames extracted", len(frames))
    return frames


def collect_inputs(
    paths: list[Path], fps: float, tmp_dir: Path
) -> list[tuple[Path, str]]:
    """Return a flat (image_path, label) list from videos, images, or dirs."""
    items: list[tuple[Path, str]] = []
    for p in paths:
        if not p.exists():
            log.warning("Path not found, skipping: %s", p)
            continue
        if p.is_dir():
            images = sorted(
                f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
            )
            if not images:
                log.warning("No images found in directory: %s", p)
            for img in images:
                items.append((img, f"{p.name}/{img.name}"))
        elif p.suffix.lower() in IMAGE_SUFFIXES:
            items.append((p, p.name))
        elif p.suffix.lower() in VIDEO_SUFFIXES:
            frame_dir = tmp_dir / p.stem
            frames = extract_frames(p, fps, frame_dir)
            for frame in frames:
                items.append((frame, f"{p.name}:{frame.stem}"))
        else:
            log.warning("Unsupported file type, skipping: %s", p)
    return items


# ── Classification ─────────────────────────────────────────────────────────────


def classify_image(
    image_path: Path,
    local_model: LocalBirdModel | None,
    inat_client: INatClient | None,
    local_threshold: float,
) -> tuple[str, str, float, str] | None:
    """Run the two-stage classifier on one image.

    Returns (species_common, species_scientific, confidence, classifier_name)
    or None if classification produced no result.
    Mirrors the logic in classifier.py: local first, iNat as fallback.
    """
    local_fallback: tuple[str, float] | None = None

    # Stage 1 — local model
    if local_model is not None:
        local_results = local_model.classify(image_path)
        if local_results:
            top_name, top_conf = local_results[0]
            if top_conf >= local_threshold:
                return top_name, "", top_conf, "tfhub_birds"
            local_fallback = (top_name, top_conf)

    # Stage 2 — iNaturalist
    if inat_client is not None:
        inat_results = inat_client.classify(image_path)
        if inat_results:
            top = inat_results[0]
            return top.species_common, top.species_scientific, top.confidence, "inaturalist"

    # Use local result if iNat was skipped or failed
    if local_fallback is not None:
        return local_fallback[0], "", local_fallback[1], "tfhub_birds"

    return None


# ── Main ───────────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    min_confidence: float = args.min_confidence

    # Load local TFHub model
    local_model: LocalBirdModel | None = None
    if not args.skip_local:
        local_model = LocalBirdModel()
        try:
            local_model.load()
        except Exception:
            log.exception("Failed to load local TFHub model — falling back to iNat only")
            local_model = None

    # iNaturalist client
    inat_client: INatClient | None = None if args.skip_inat else INatClient()

    if local_model is None and inat_client is None:
        log.error("Both classifiers are disabled — nothing to run")
        sys.exit(1)

    detection_logger = DetectionLogger() if not args.no_log else None

    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="birdtest_") as tmp:
        items = collect_inputs(
            [Path(p) for p in args.inputs], args.fps, Path(tmp)
        )

        if not items:
            log.error("No frames or images to process — exiting")
            sys.exit(1)

        log.info("Processing %d frames/images …", len(items))

        for i, (image_path, label) in enumerate(items, 1):
            log.info("[%d/%d] %s", i, len(items), label)

            result = classify_image(
                image_path, local_model, inat_client, LOCAL_MODEL_CONFIDENCE_THRESHOLD
            )

            if result is None:
                log.info("  → no classification result")
                continue

            species, scientific, conf, clf = result

            if conf < min_confidence:
                log.info(
                    "  → %s (%.2f) — below --min-confidence %.2f, skipping",
                    species, conf, min_confidence,
                )
                continue

            now = datetime.now(timezone.utc).astimezone()
            log.info("  → %-40s conf=%.2f  [%s]", species, conf, clf)

            results.append(
                {
                    "source": label,
                    "species": species,
                    "scientific": scientific,
                    "confidence": conf,
                    "classifier": clf,
                    "timestamp": now.isoformat(),
                }
            )

            if detection_logger is not None:
                record = DetectionRecord(
                    timestamp=now.isoformat(),
                    species_common=species,
                    species_scientific=scientific,
                    confidence=conf,
                    source="visual",
                    classifier=clf,
                    frigate_event_id=f"test_{label}",
                )
                detection_logger.log(record)

    _print_summary(results, min_confidence)


def _print_summary(results: list[dict], min_confidence: float) -> None:
    width = 70
    print(f"\n{'─' * width}")
    print(f"  DETECTION SUMMARY   (min_confidence={min_confidence:.2f})")
    print(f"{'─' * width}")

    if not results:
        print("  No detections above the confidence threshold.\n")
        return

    print(f"  Total detections: {len(results)}")
    print()

    species_counts: Counter = Counter(r["species"] for r in results)
    avg_conf: dict[str, float] = {
        s: sum(r["confidence"] for r in results if r["species"] == s) / c
        for s, c in species_counts.items()
    }

    print(f"  {'Species':<38} {'Count':>5}  {'Avg Conf':>8}  Classifier")
    print(f"  {'─' * 62}")
    for species, count in species_counts.most_common():
        clfs = [r["classifier"] for r in results if r["species"] == species]
        clf = max(set(clfs), key=clfs.count)
        scientific = next(
            (r["scientific"] for r in results if r["species"] == species and r["scientific"]),
            "",
        )
        label = f"{species}" + (f"  ({scientific})" if scientific else "")
        print(f"  {label:<38} {count:>5}  {avg_conf[species]:>8.2f}  {clf}")

    print(f"{'─' * width}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the bird classifier pipeline on local videos or images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="PATH",
        help="Video file(s), image file(s), or directory of snapshots",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        metavar="N",
        help="Frames per second to extract from video files (default: 1.0)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_CONFIDENCE_LOG,
        metavar="N",
        help=f"Minimum confidence to include in results (default: {MIN_CONFIDENCE_LOG} from .env)",
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
        "--no-log",
        action="store_true",
        help="Do not write results to data/detections.csv and data/detections.db",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
