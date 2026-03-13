"""Configuration loaded from environment variables."""

import os
from pathlib import Path


# ── Video source ────────────────────────────────────────────────────────────
# RTSP URL (live camera or mediamtx test stream) or a local video file path.
VIDEO_SOURCE: str = os.environ.get("VIDEO_SOURCE", "")

# Frames per second to sample from the stream.  Lower = less CPU.
CAPTURE_FPS: int = int(os.environ.get("CAPTURE_FPS", "5"))

# For local video files: loop instead of stopping at end-of-file.
VIDEO_LOOP: bool = os.environ.get("VIDEO_LOOP", "true").lower() == "true"

# ── Motion detection (MOG2 background subtractor) ───────────────────────────
# Number of frames used to build the background model.
MOTION_HISTORY: int = int(os.environ.get("MOTION_HISTORY", "300"))

# How much a pixel must change to be considered foreground.
# Lower → more sensitive. Typical range: 16–64.
MOTION_VAR_THRESHOLD: float = float(os.environ.get("MOTION_VAR_THRESHOLD", "32"))

# Minimum area (px²) of a motion blob to bother running the detector.
# Filters out noise like swaying leaves.
MOTION_MIN_AREA: int = int(os.environ.get("MOTION_MIN_AREA", "800"))

# Maximum area (px²) of a motion blob — anything larger is likely lighting
# change, not a bird.
MOTION_MAX_AREA: int = int(os.environ.get("MOTION_MAX_AREA", "120000"))

# Morphological dilation passes applied to the motion mask before contouring.
# More passes merge nearby blobs but cost a little CPU.
MOTION_DILATE_ITER: int = int(os.environ.get("MOTION_DILATE_ITER", "2"))

# ── YOLO bird detector ───────────────────────────────────────────────────────
# YOLOv8 model variant.  'yolov8n.pt' (nano, ~6 MB) is fast on CPU.
# Downloaded automatically by ultralytics on first run.
YOLO_MODEL: str = os.environ.get("YOLO_MODEL", "yolov8n.pt")

# Minimum YOLO confidence to consider a detection.
YOLO_CONFIDENCE: float = float(os.environ.get("YOLO_CONFIDENCE", "0.25"))

# Bounding-box area bounds (px²) for valid bird detections.
# A titmouse at 1280×720 is typically 1 000–30 000 px².
# These caps prevent a sky-filling false positive from being classified.
DETECTION_MIN_AREA: int = int(os.environ.get("DETECTION_MIN_AREA", "500"))
DETECTION_MAX_AREA: int = int(os.environ.get("DETECTION_MAX_AREA", "80000"))

# ── Species classifier (TFHub AIY Birds V1) ──────────────────────────────────
# If the local model scores at or above this, skip iNat and log immediately.
LOCAL_MODEL_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("LOCAL_MODEL_CONFIDENCE_THRESHOLD", "0.85")
)

# Path to a plain-text file of allowed common names (one per line).
# Predictions are restricted to this list, preventing the global 965-class
# model from returning seabirds or shorebirds at a backyard feeder.
# Defaults to species_list.txt baked into the image at /app/species_list.txt.
SPECIES_LIST_PATH: Path = Path(os.environ.get("SPECIES_LIST_PATH", "/app/species_list.txt"))

# ── Logging / notification thresholds ───────────────────────────────────────
MIN_CONFIDENCE_LOG: float = float(os.environ.get("MIN_CONFIDENCE_LOG", "0.3"))
MIN_CONFIDENCE_NOTIFY: float = float(os.environ.get("MIN_CONFIDENCE_NOTIFY", "0.7"))

# Minimum YOLO confidence to trigger a predator alert.  Kept higher than
# YOLO_CONFIDENCE so that squirrels/raccoons mis-detected as "cat" at low
# confidence don't fire alerts.  A real cat in frame typically scores 0.7+.
PREDATOR_MIN_CONFIDENCE: float = float(os.environ.get("PREDATOR_MIN_CONFIDENCE", "0.6"))

# Seconds to wait before logging the same detection region again.
# Prevents flooding the DB with the same bird sitting at the feeder.
DETECTION_COOLDOWN_SEC: float = float(os.environ.get("DETECTION_COOLDOWN_SEC", "10.0"))

# ── MQTT ────────────────────────────────────────────────────────────────────
MQTT_HOST: str = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT: int = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER: str = os.environ.get("MQTT_USER", "")
MQTT_PASSWORD: str = os.environ.get("MQTT_PASSWORD", "")

TOPIC_DETECTION: str = "birdfeeder/detection"
TOPIC_NEW_SPECIES: str = "birdfeeder/new_species"
TOPIC_PREDATOR_ALERT: str = "birdfeeder/predator_alert"
TOPIC_UNKNOWN: str = "birdfeeder/unknown"

# ── Location (used by iNat API geo-filter when re-enabled) ──────────────────
LATITUDE: float | None = (
    float(os.environ["LATITUDE"]) if os.environ.get("LATITUDE") else None
)
LONGITUDE: float | None = (
    float(os.environ["LONGITUDE"]) if os.environ.get("LONGITUDE") else None
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "/data"))
SNAPSHOT_DIR: Path = DATA_DIR / "snapshots"
CORRECTIONS_DIR: Path = DATA_DIR / "corrections"
CSV_PATH: Path = DATA_DIR / "detections.csv"
DB_PATH: Path = DATA_DIR / "detections.db"

SNAPSHOT_JPEG_QUALITY: int = int(os.environ.get("SNAPSHOT_JPEG_QUALITY", "88"))
