"""Configuration loaded from environment variables."""

import os
from pathlib import Path


# ── Video source ────────────────────────────────────────────────────────────
VIDEO_SOURCE: str = os.environ.get("VIDEO_SOURCE", "")
CAPTURE_FPS: int = int(os.environ.get("CAPTURE_FPS", "5"))
VIDEO_LOOP: bool = os.environ.get("VIDEO_LOOP", "true").lower() == "true"

# ── Motion detection (MOG2 background subtractor) ───────────────────────────
MOTION_HISTORY: int = int(os.environ.get("MOTION_HISTORY", "300"))
MOTION_VAR_THRESHOLD: float = float(os.environ.get("MOTION_VAR_THRESHOLD", "32"))
MOTION_MIN_AREA: int = int(os.environ.get("MOTION_MIN_AREA", "800"))
MOTION_MAX_AREA: int = int(os.environ.get("MOTION_MAX_AREA", "120000"))
MOTION_DILATE_ITER: int = int(os.environ.get("MOTION_DILATE_ITER", "2"))

# ── YOLO bird detector ───────────────────────────────────────────────────────
YOLO_MODEL: str = os.environ.get("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE: float = float(os.environ.get("YOLO_CONFIDENCE", "0.25"))

# Bounding-box area bounds (px²) for valid detections.
DETECTION_MIN_AREA: int = int(os.environ.get("DETECTION_MIN_AREA", "500"))
DETECTION_MAX_AREA: int = int(os.environ.get("DETECTION_MAX_AREA", "800000"))

# Minimum YOLO confidence to trigger a predator alert.
# A real cat in frame typically scores 0.7+; raised from 0.6 to reduce
# small-bird false positives on the predator path.
PREDATOR_MIN_CONFIDENCE: float = float(os.environ.get("PREDATOR_MIN_CONFIDENCE", "0.70"))

# If a predator-class detection (cat/dog/bear) has a bounding-box area
# smaller than this, it is almost certainly a small bird misclassified by
# YOLO — demote it to "bird" and send it to the species classifier instead
# of firing a predator alert.  A titmouse at 1280x720 is ~2 000–8 000 px²;
# a real cat fills 20 000 px² or more.
PREDATOR_MIN_AREA: int = int(os.environ.get("PREDATOR_MIN_AREA", "15000"))

# ── Species classifier ───────────────────────────────────────────────────────
# Which backend to use: "tfhub" | "bioclip" | "nabirds"
CLASSIFIER_BACKEND: str = os.environ.get("CLASSIFIER_BACKEND", "tfhub")

# BioCLIP (open_clip) model string — only used when CLASSIFIER_BACKEND=bioclip
BIOCLIP_MODEL: str = os.environ.get("BIOCLIP_MODEL", "hf-hub:imageomics/bioclip")

# HuggingFace model ID — only used when CLASSIFIER_BACKEND=nabirds
NABIRDS_MODEL: str = os.environ.get("NABIRDS_MODEL", "chriamue/bird-species-classifier")

# Path to fine-tuned EfficientNet weights — only used when CLASSIFIER_BACKEND=efficientnet
# Train with: python scripts/train_efficientnet.py --data-dir training-data/
EFFICIENTNET_MODEL_PATH: Path = Path(
    os.environ.get("EFFICIENTNET_MODEL_PATH", "/data/models/feeder_birds.pt")
)

# Path to allowlist of common names (one per line).
# Restricts predictions to expected backyard species.
SPECIES_LIST_PATH: Path = Path(os.environ.get("SPECIES_LIST_PATH", "/app/species_list.txt"))

# ── Logging / notification thresholds ───────────────────────────────────────
MIN_CONFIDENCE_LOG: float = float(os.environ.get("MIN_CONFIDENCE_LOG", "0.3"))
MIN_CONFIDENCE_NOTIFY: float = float(os.environ.get("MIN_CONFIDENCE_NOTIFY", "0.7"))
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

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "/data"))
SNAPSHOT_DIR: Path = DATA_DIR / "snapshots"
CORRECTIONS_DIR: Path = DATA_DIR / "corrections"
CSV_PATH: Path = DATA_DIR / "detections.csv"
DB_PATH: Path = DATA_DIR / "detections.db"
SNAPSHOT_JPEG_QUALITY: int = int(os.environ.get("SNAPSHOT_JPEG_QUALITY", "88"))
