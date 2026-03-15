#!/usr/bin/env python3
"""Extract YOLO-detected object crops from a feeder camera video.

Runs the same BirdDetector (YOLOv8) and MotionDetector used in the live
pipeline against a local video file.  Every detection above the confidence
threshold is saved as a cropped JPEG, organised by label, ready to drop
straight into test-photos/ for classifier comparison.

Output structure:
  <output-dir>/
    bird/           — cropped bird detections
    cat/            — cropped predator detections (cat/dog/bear)
    frames/         — full frames with bounding boxes drawn (optional)
    detections.csv  — one row per crop: frame, label, confidence, bbox, area

Usage:
    # Basic — same settings as the live pipeline
    python scripts/extract_yolo_crops.py path/to/video.mp4

    # Sample 2 frames/sec instead of the default 1
    python scripts/extract_yolo_crops.py video.mp4 --fps 2

    # Also save annotated full frames (useful for checking false positives)
    python scripts/extract_yolo_crops.py video.mp4 --save-frames

    # Skip motion gating — run YOLO on every sampled frame
    # (catches more detections but slower; default: motion gate ON to match pipeline)
    python scripts/extract_yolo_crops.py video.mp4 --no-motion

    # Override YOLO settings without editing config
    python scripts/extract_yolo_crops.py video.mp4 --yolo-model yolov8s.pt --confidence 0.30

    # Custom output directory
    python scripts/extract_yolo_crops.py video.mp4 --output-dir my_crops/
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIRD_DETECTOR = PROJECT_ROOT / "bird-detector"
sys.path.insert(0, str(BIRD_DETECTOR))

# Set dummy env defaults so config.py doesn't crash on missing VIDEO_SOURCE etc.
os.environ.setdefault("VIDEO_SOURCE", "")
os.environ.setdefault("DATA_DIR", str(PROJECT_ROOT / "data"))

import config
from detector import BirdDetector
from motion import MotionDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("extract-yolo-crops")

# Bounding-box colours per label (BGR)
_BOX_COLOURS = {
    "bird":  (34, 197, 94),    # green
    "cat":   (239, 68, 68),    # red
    "dog":   (239, 68, 68),
    "bear":  (239, 68, 68),
}
_DEFAULT_COLOUR = (234, 179, 8)  # yellow


def _draw_box(frame, label: str, conf: float, x1: int, y1: int, x2: int, y2: int) -> None:
    colour = _BOX_COLOURS.get(label, _DEFAULT_COLOUR)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), colour, -1)
    lum = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
    text_colour = (15, 15, 15) if lum > 140 else (245, 245, 245)
    cv2.putText(frame, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colour, 2)


def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    if not video_path.exists():
        log.error("Video not found: %s", video_path)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in ("bird", "cat"):
        (output_dir / label).mkdir(exist_ok=True)

    frames_dir = output_dir / "frames"
    if args.save_frames:
        frames_dir.mkdir(exist_ok=True)

    # ── Instantiate detector — same class and defaults as pipeline.py ─────────
    detector = BirdDetector(
        model_name=args.yolo_model,
        confidence=args.confidence,
        min_area=args.min_area,
        max_area=args.max_area,
        predator_min_area=args.predator_min_area,
    )
    detector.load()

    motion_detector: MotionDetector | None = None
    if not args.no_motion:
        motion_detector = MotionDetector(
            history=config.MOTION_HISTORY,
            var_threshold=config.MOTION_VAR_THRESHOLD,
            min_area=config.MOTION_MIN_AREA,
            max_area=config.MOTION_MAX_AREA,
            dilate_iterations=config.MOTION_DILATE_ITER,
        )
        log.info("Motion gating: ON (matches live pipeline)")
    else:
        log.info("Motion gating: OFF (YOLO runs on every sampled frame)")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        sys.exit(1)

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps
    sample_every = max(1, round(video_fps / args.fps))

    log.info(
        "Video: %s  (%.1f fps, %d frames, %.1fs)",
        video_path.name, video_fps, total_frames, duration_sec,
    )
    log.info(
        "Sampling every %d frames (≈%.1f fps)  YOLO: %s  conf=%.2f  area=[%d,%d]  predator_min_area=%d",
        sample_every, video_fps / sample_every,
        args.yolo_model, args.confidence, args.min_area, args.max_area, args.predator_min_area,
    )

    # ── Warm up motion background model ──────────────────────────────────────
    if motion_detector is not None:
        log.info("Warming up motion background model …")
        ret, first_frame = cap.read()
        if ret:
            motion_detector.warmup(first_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Main extraction loop ──────────────────────────────────────────────────
    csv_rows: list[dict] = []
    counts: dict[str, int] = {}
    frame_num = 0
    sampled = 0
    motion_skipped = 0
    t_start = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % sample_every != 0:
            continue

        sampled += 1

        # Motion gate
        if motion_detector is not None:
            regions = motion_detector.detect(frame)
            if not regions:
                motion_skipped += 1
                continue

        detections = detector.detect(frame)
        if not detections:
            continue

        annotated = frame.copy() if args.save_frames else None

        for det in detections:
            label = det.label
            counts[label] = counts.get(label, 0) + 1
            crop_idx = counts[label]

            # Crop — same slice used by pipeline.py
            crop = frame[det.y1:det.y2, det.x1:det.x2]
            if crop.size == 0:
                continue

            # Determine output subfolder: predators go under "cat" regardless
            # of exact label (cat/dog/bear) so test-photos stays organised.
            subfolder = "bird" if label == "bird" else "cat"
            ts_ms = int((frame_num / video_fps) * 1000)
            filename = (
                f"f{frame_num:06d}_{ts_ms:07d}ms"
                f"_{label}_{det.confidence:.2f}"
                f"_{det.area}px.jpg"
            )
            crop_path = output_dir / subfolder / filename
            cv2.imwrite(
                str(crop_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            csv_rows.append({
                "frame":      frame_num,
                "time_ms":    ts_ms,
                "label":      label,
                "confidence": f"{det.confidence:.4f}",
                "area":       det.area,
                "x1": det.x1, "y1": det.y1,
                "x2": det.x2, "y2": det.y2,
                "crop_file":  f"{subfolder}/{filename}",
            })

            if annotated is not None:
                _draw_box(annotated, label, det.confidence, det.x1, det.y1, det.x2, det.y2)

            log.info(
                "  frame %6d  %-6s  conf=%.2f  area=%-6d  → %s",
                frame_num, label, det.confidence, det.area, crop_path.name,
            )

        if annotated is not None and detections:
            frame_out = frames_dir / f"f{frame_num:06d}.jpg"
            cv2.imwrite(str(frame_out), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

        # Progress every 500 sampled frames
        if sampled % 500 == 0:
            elapsed = time.monotonic() - t_start
            pct = frame_num / max(total_frames, 1) * 100
            log.info("Progress: %.0f%%  sampled=%d  detections=%d  elapsed=%.0fs",
                     pct, sampled, sum(counts.values()), elapsed)

    cap.release()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = output_dir / "detections.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["frame", "time_ms", "label", "confidence", "area",
                      "x1", "y1", "x2", "y2", "crop_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    total_det = sum(counts.values())
    print(f"\n{'─' * 60}")
    print(f"  VIDEO EXTRACTION SUMMARY")
    print(f"{'─' * 60}")
    print(f"  Video          : {video_path.name}")
    print(f"  Duration       : {duration_sec:.1f}s  ({total_frames} frames @ {video_fps:.1f} fps)")
    print(f"  Sampled        : {sampled} frames  ({motion_skipped} skipped by motion gate)")
    print(f"  Total crops    : {total_det}")
    for label, n in sorted(counts.items()):
        subfolder = "bird" if label == "bird" else "cat"
        print(f"    {label:<12}: {n:>4}  →  {output_dir}/{subfolder}/")
    print(f"  Detections CSV : {csv_path}")
    if args.save_frames:
        print(f"  Annotated frames: {frames_dir}/")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print(f"{'─' * 60}")
    print(f"\n  To test classifiers on the bird crops:")
    print(f"    python scripts/test_photos.py --input-dir {output_dir}/bird/\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract YOLO-detected crops from a feeder camera video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--output-dir", "-o",
        default=str(PROJECT_ROOT / "yolo-crops"),
        metavar="DIR",
        help="Where to save crops (default: yolo-crops/)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        metavar="N",
        help="Frames per second to sample from the video (default: 1.0)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save annotated full frames for every frame with a detection",
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Disable motion gating — run YOLO on every sampled frame",
    )

    # YOLO overrides — defaults mirror config.py so no .env needed
    parser.add_argument(
        "--yolo-model",
        default=config.YOLO_MODEL,
        metavar="MODEL",
        help=f"YOLO model file (default: {config.YOLO_MODEL})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.YOLO_CONFIDENCE,
        metavar="N",
        help=f"YOLO confidence threshold (default: {config.YOLO_CONFIDENCE})",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=config.DETECTION_MIN_AREA,
        metavar="PX2",
        help=f"Minimum detection area in px² (default: {config.DETECTION_MIN_AREA})",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=config.DETECTION_MAX_AREA,
        metavar="PX2",
        help=f"Maximum detection area in px² (default: {config.DETECTION_MAX_AREA})",
    )
    parser.add_argument(
        "--predator-min-area",
        type=int,
        default=config.PREDATOR_MIN_AREA,
        metavar="PX2",
        help=(
            f"Predator detections below this area are demoted to 'bird' "
            f"(default: {config.PREDATOR_MIN_AREA})"
        ),
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
