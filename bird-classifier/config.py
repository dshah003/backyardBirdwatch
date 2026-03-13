"""Configuration loaded from environment variables."""

import os
from pathlib import Path


# MQTT
MQTT_HOST: str = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT: int = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER: str = os.environ.get("MQTT_USER", "")
MQTT_PASSWORD: str = os.environ.get("MQTT_PASSWORD", "")

# Frigate
FRIGATE_URL: str = os.environ.get("FRIGATE_URL", "http://frigate:5000")
FRIGATE_EVENTS_TOPIC: str = "frigate/events"

# MQTT publish topics
TOPIC_DETECTION: str = "birdfeeder/detection"
TOPIC_DAILY_SUMMARY: str = "birdfeeder/daily_summary"
TOPIC_NEW_SPECIES: str = "birdfeeder/new_species"
TOPIC_PREDATOR_ALERT: str = "birdfeeder/predator_alert"
TOPIC_UNKNOWN: str = "birdfeeder/unknown"

# Location
LATITUDE: float | None = (
    float(os.environ["LATITUDE"]) if os.environ.get("LATITUDE") else None
)
LONGITUDE: float | None = (
    float(os.environ["LONGITUDE"]) if os.environ.get("LONGITUDE") else None
)

# iNaturalist
INAT_RATE_LIMIT_PER_MIN: int = int(os.environ.get("INAT_RATE_LIMIT_PER_MIN", "30"))
# JWT from https://www.inaturalist.org/users/api_token (requires approved CV API access)
INAT_API_TOKEN: str | None = os.environ.get("INAT_API_TOKEN") or None

# Classification thresholds
MIN_CONFIDENCE_LOG: float = float(os.environ.get("MIN_CONFIDENCE_LOG", "0.3"))
MIN_CONFIDENCE_NOTIFY: float = float(os.environ.get("MIN_CONFIDENCE_NOTIFY", "0.7"))
LOCAL_MODEL_CONFIDENCE_THRESHOLD: float = 0.85

# Paths
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "/data"))
SNAPSHOT_DIR: Path = DATA_DIR / "snapshots"
CORRECTIONS_DIR: Path = DATA_DIR / "corrections"
CSV_PATH: Path = DATA_DIR / "detections.csv"
DB_PATH: Path = DATA_DIR / "detections.db"
SPECIES_LIST_PATH: Path = Path(
    os.environ.get("SPECIES_LIST_PATH", "/app/species_list.txt")
)

# Frigate media
FRIGATE_MEDIA_DIR: Path = Path(
    os.environ.get("FRIGATE_MEDIA_DIR", "/media/frigate")
)
