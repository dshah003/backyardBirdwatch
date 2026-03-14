#!/usr/bin/env python3
"""Test the full detection pipeline on local video files or image directories.

Runs the same Motion → YOLO → Species Classifier stack used in the live
pipeline against a local video file or a folder of still images.  No Docker,
MQTT, or camera required.

Output: detection summary printed to terminal + optional CSV/SQLite log.

Usage:
    # Run on a video file (samples 1 frame/sec by default)
    python scripts/test_pipeline.py path/to/video.mp4

    # Multiple videos
    python scripts/test_pipeline.py clip1.mp4 clip2.mp4 --fps 2

    # Test on a folder of pre-cropped images (skips YOLO — classifies directly)
    python scripts/test_pipeline.py data/corrections/

    # Choose classifier backend
    python scripts/test_pipeline.py video.mp4 --backend bioclip
    python scripts/test_pipeline.py video.mp4 --backend nabirds
    python scripts/test_pipeline.py video.mp4 --backend tfhub   # default

    # Disable motion gating (run YOLO on every sampled frame)
    python scripts/test_pipeline.py video.mp4 --no-motion

    # Lower confidence bar to see more candidates
    python scripts/test_pipeline.py video.mp4 --min-confidence 0.2

    # Don't write to the database
    python scripts/test_pipeline.py video.mp4 --no-log

Install deps for each backend:
    tfhub   : pip install tensorflow tensorflow-hub requests
    bioclip : pip install open-clip-torch pillow
    nabirds : pip install transformers torch pillow
"""

import argparse
import logging
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIRD_DETECTOR = PROJECT_ROOT / "bird-detector"
sys.path.insert(0, str(BIRD_DETECTOR))

os.environ.setdefault("VIDEO_SOURCE", "")
os.environ.setdefault("DATA_DIR", str(PROJECT_ROOT / "data"))
os.environ.setdefault("SPECIES_LIST_PATH", str(BIRD_DETECTOR / "species_list.txt"))

import config
from detector import BirdDetector
from logger import DetectionLogger, DetectionRecord
from motion import MotionDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("test-pipeline")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".ts"}


# ── Classifier factory (mirrors pipeline.py) ──────────────────────────────────

def _load_classifier(backend: str):
    species_list = Path(os.environ["SPECIES_LIST_PATH"])
    try:
        if backend == "bioclip":
            from classifier_bioclip import BioCLIPClassifier
            clf = BioCLIPClassifier(species_list_path=species_list)
        elif backend == "nabirds":
            from classifier_nabirds import NABirdsClassifier
            clf = NABirdsClassifier(species_list_path=species_list)
        else:
            from classifier import SpeciesClassifier
            clf = SpeciesClassifier(species_list_path=species_list)
        clf.load()
        return clf
    except ImportError as e:
        log.error("Backend %r missing deps: %s", backend, e)
        sys.exit(1)
    except Exception:
        log.exception("Failed to load backend %r", backend)
        sys.exit(1)


# ── Input collection ──────────────────────────────────────────────────────────

def _extract_frames(video_path: Path, fps: float, output_dir: Path) -> list[Path]:
    """Sample frames from a video at target fps using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(1, round(video_fps / fps))
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % sample_every == 0:
            out = output_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append(out)
    cap.release()
    log.info("  → %d frames extracted from %s", len(frames), video_path.name)
    return frames


def _collect_inputs(paths: list[Path], fps: float, tmp_dir: Path) -> list[tuple[Path, str, bool]]:
    """Return list of (image_path, label, is_precrop).

    is_precrop=True means the image is already a bird crop — skip YOLO and
    classify directly (useful for testing data/corrections/).
    """
    items = []
    for p in paths:
        if not p.exists():
            log.warning("Path not found, skipping: %s", p)
            continue
        if p.is_dir():
            images = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
            for img in images:
                items.append((img, f"{p.name}/{img.name}", True))
        elif p.suffix.lower() in IMAGE_SUFFIXES:
            items.append((p, p.name, True))
        elif p.suffix.lower() in VIDEO_SUFFIXES:
            frame_dir = tmp_dir / p.stem
            frames = _extract_frames(p, fps, frame_dir)
            for f in frames:
                items.append((f, f"{p.name}:{f.stem}", False))
        else:
            log.warning("Unsupported file type, skipping: %s", p)
    return items


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    log.info("Loading classifier backend: %s", args.backend)
    classifier = _load_classifier(args.backend)

    bird_detector: BirdDetector | None = None
    motion_detector: MotionDetector | None = None

    if not args.skip_yolo:
        bird_detector = BirdDetector(
            model_name=config.YOLO_MODEL,
            confidence=config.YOLO_CONFIDENCE,
            min_area=config.DETECTION_MIN_AREA,
            max_area=config.DETECTION_MAX_AREA,
            predator_min_area=config.PREDATOR_MIN_AREA,
        )
        bird_detector.load()

        if not args.no_motion:
            motion_detector = MotionDetector(
                history=config.MOTION_HISTORY,
                var_threshold=config.MOTION_VAR_THRESHOLD,
                min_area=config.MOTION_MIN_AREA,
                max_area=config.MOTION_MAX_AREA,
                dilate_iterations=config.MOTION_DILATE_ITER,
            )

    det_logger = DetectionLogger() if not args.no_log else None
    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="birdtest_") as tmp:
        items = _collect_inputs(
            [Path(p) for p in args.inputs], args.fps, Path(tmp)
        )
        if not items:
            log.error("No frames or images to process")
            sys.exit(1)

        log.info("Processing %d frames/images …", len(items))

        for i, (image_path, label, is_precrop) in enumerate(items, 1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                log.warning("[%d/%d] Cannot read: %s", i, len(items), image_path)
                continue

            log.info("[%d/%d] %s", i, len(items), label)

            crops_to_classify: list[tuple[np.ndarray, str]] = []

            if is_precrop or bird_detector is None:
                # Image is already a crop — classify the whole thing directly
                crops_to_classify.append((frame, "bird"))
            else:
                # Motion gate
                if motion_detector is not None:
                    regions = motion_detector.detect(frame)
                    if not regions:
                        log.debug("  no motion")
                        continue

                detections = bird_detector.detect(frame)
                if not detections:
                    log.debug("  no YOLO detections")
                    continue

                for det in detections:
                    crop = frame[det.y1:det.y2, det.x1:det.x2]
                    if crop.size > 0:
                        crops_to_classify.append((crop, det.label))

            for crop, det_label in crops_to_classify:
                top = classifier.classify(crop, top_n=5)
                if not top:
                    log.info("  → no classifier result")
                    continue

                species, conf = top[0]
                if conf < args.min_confidence:
                    log.info("  → %s (%.2f) — below threshold", species, conf)
                    continue

                log.info("  → %-40s conf=%.2f  [%s]", species, conf, args.backend)

                now = datetime.now(timezone.utc).astimezone()
                results.append({
                    "source": label,
                    "det_label": det_label,
                    "species": species,
                    "confidence": conf,
                    "backend": args.backend,
                    "timestamp": now.isoformat(),
                    "candidates": top,
                })

                if det_logger is not None:
                    det_logger.log(DetectionRecord(
                        timestamp=now.isoformat(),
                        species_common=species,
                        confidence=conf,
                        classifier=args.backend,
                    ))

    _print_summary(results, args.min_confidence, args.backend)


def _print_summary(results: list[dict], min_confidence: float, backend: str) -> None:
    width = 72
    print(f"\n{'─' * width}")
    print(f"  DETECTION SUMMARY   backend={backend}  min_confidence={min_confidence:.2f}")
    print(f"{'─' * width}")

    if not results:
        print("  No detections above the confidence threshold.\n")
        return

    print(f"  Total detections: {len(results)}\n")
    counts: Counter = Counter(r["species"] for r in results)
    avg_conf = {
        s: sum(r["confidence"] for r in results if r["species"] == s) / c
        for s, c in counts.items()
    }

    print(f"  {'Species':<40} {'Count':>5}  {'Avg Conf':>8}")
    print(f"  {'─' * 58}")
    for species, count in counts.most_common():
        print(f"  {species:<40} {count:>5}  {avg_conf[species]:>8.2f}")
    print(f"{'─' * width}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the bird detection pipeline on local video files or images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="PATH",
        help="Video file(s), image file(s), or directory of crops",
    )
    parser.add_argument(
        "--backend",
        choices=["tfhub", "bioclip", "nabirds"],
        default="tfhub",
        help="Classifier backend (default: tfhub)",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0, metavar="N",
        help="Frames per second to sample from video files (default: 1.0)",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=config.MIN_CONFIDENCE_LOG, metavar="N",
        help=f"Minimum confidence to include (default: {config.MIN_CONFIDENCE_LOG})",
    )
    parser.add_argument(
        "--no-motion", action="store_true",
        help="Disable motion gating — run YOLO on every sampled frame",
    )
    parser.add_argument(
        "--skip-yolo", action="store_true",
        help="Classify the full frame directly without YOLO (same as passing a crop directory)",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Do not write results to data/detections.csv and data/detections.db",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
