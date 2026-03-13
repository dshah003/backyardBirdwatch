#!/usr/bin/env python3
"""Convert a directory of still images into a loopable test video for the Frigate pipeline.

Each image is held for --duration seconds so Frigate has enough time to detect
objects in it.  The output is an H.264/yuv420p MP4 compatible with mediamtx
and Frigate.

Usage:
    # Default: test-photos/ → test-videos/sample.mp4, 4 s/image, 1280×720
    python scripts/make_test_video.py

    # Custom directory and duration
    python scripts/make_test_video.py --input-dir my_photos/ --duration 6

    # Different output resolution (must be even dimensions)
    python scripts/make_test_video.py --resolution 1920x1080

    # Preview the ffmpeg command without running it
    python scripts/make_test_video.py --dry-run

After the video is created run the full stack:
    docker compose -f docker-compose.yml -f docker-compose.test.yml up -d

Then open Frigate at http://localhost:5000 to watch detections accumulate.
"""

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("make-test-video")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def build_concat_file(images: list[Path], duration: float, concat_path: Path) -> None:
    """Write an ffmpeg concat demuxer file — one entry per image."""
    with open(concat_path, "w") as f:
        for img in images:
            # ffmpeg concat demuxer requires absolute paths or paths relative to
            # the concat file; use absolute to be safe.
            f.write(f"file '{img.resolve()}'\n")
            f.write(f"duration {duration}\n")
        # Repeat the last frame to avoid a truncated final image
        if images:
            f.write(f"file '{images[-1].resolve()}'\n")


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output    = Path(args.output)
    duration  = args.duration

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
    log.info("Duration per image: %.1f s  →  total ~%.0f s", duration, len(images) * duration)

    # Parse resolution
    try:
        w_str, h_str = args.resolution.lower().split("x")
        width, height = int(w_str), int(h_str)
    except ValueError:
        log.error("Invalid --resolution %r — expected WxH e.g. 1280x720", args.resolution)
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="birdtest_concat_", delete=False
    ) as tmp:
        concat_path = Path(tmp.name)

    build_concat_file(images, duration, concat_path)
    log.info("Concat list → %s", concat_path)

    # Output frame rate.  25 fps gives zoompan enough frames per image to
    # produce smooth motion; Frigate detect runs at 5 fps so it samples every
    # 5th output frame — still plenty for reliable detection.
    output_fps = 25
    frames_per_image = max(10, int(duration * output_fps))

    # ── Filter chain ──────────────────────────────────────────────────────────
    # Step 1 — fit inside target resolution, letterbox/pillarbox with black.
    #   in_range/out_range: JPEG source is full-range (YUVJ); output is
    #   limited/TV range.  Without this swscaler emits a noisy warning.
    prep = (
        f"scale={width}:{height}"
        f":force_original_aspect_ratio=decrease"
        f":in_range=full:out_range=limited,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )

    # Step 2 — Ken Burns slow zoom.
    # Frigate only runs object detection when it first detects motion.  A
    # static slideshow only triggers motion at frame transitions, producing
    # very few events.  A slow zoom keeps pixels changing every frame so
    # Frigate's motion detector fires continuously throughout each image.
    #
    # z expression: zoom in from 1.0 to ~1.2 over the image's display window.
    # x/y: keep the zoom centred on the image centre.
    # d: total output frames for this zoom cycle (= duration × fps).
    # s: output size (must match the padded canvas from step 1).
    # fps: output frame rate for the zoompan filter.
    zoom_max = 1.2
    zoom_step = (zoom_max - 1.0) / frames_per_image
    zoompan = (
        f"zoompan="
        f"z='min(zoom+{zoom_step:.6f},{zoom_max})':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={frames_per_image}:"
        f"s={width}x{height}:"
        f"fps={output_fps}"
    )

    # Step 3 — force even pixel dimensions (required by yuv420p encoder).
    even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    vf = f"{prep},{zoompan},{even}"

    cmd = [
        "ffmpeg",
        "-y",                          # overwrite output
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_path),
        "-vf", vf,
        "-r", str(output_fps),         # explicit output frame rate
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output),
        "-hide_banner", "-loglevel", "info",
    ]

    log.info("Output: %s", output)
    if args.dry_run:
        print("\nDry run — would execute:\n  " + " ".join(cmd) + "\n")
        concat_path.unlink(missing_ok=True)
        return

    log.info("Running ffmpeg (this may take a moment for large photo sets)…")
    result = subprocess.run(cmd)
    concat_path.unlink(missing_ok=True)

    if result.returncode != 0:
        log.error("ffmpeg failed (exit %d)", result.returncode)
        sys.exit(result.returncode)

    size_mb = output.stat().st_size / 1_048_576
    log.info("Done — %s  (%.1f MB)", output, size_mb)
    log.info("")
    log.info("Next step — start the full test stack:")
    log.info("  docker compose -f docker-compose.yml -f docker-compose.test.yml up -d")
    log.info("Then open Frigate at http://localhost:5000")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a loopable test video from still images for the Frigate pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "test-photos"),
        metavar="DIR",
        help="Directory of source images (default: test-photos/)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "test-videos" / "sample.mp4"),
        metavar="FILE",
        help="Output video path (default: test-videos/sample.mp4)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        metavar="SECONDS",
        help="How long each image is shown in the video (default: 4.0 s). "
             "Longer = more time for Frigate to detect each image.",
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        metavar="WxH",
        help="Output resolution, e.g. 1920x1080 (default: 1280x720)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ffmpeg command without executing it",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
