"""Main bird classifier service.

Subscribes to Frigate MQTT events, classifies bird detections using a
two-stage pipeline (local TFHub model -> iNaturalist API), and publishes
enriched results to birdfeeder/* MQTT topics.
"""

import json
import logging
import re
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt
import requests

from config import (
    CORRECTIONS_DIR,
    FRIGATE_EVENTS_TOPIC,
    FRIGATE_URL,
    LOCAL_MODEL_CONFIDENCE_THRESHOLD,
    MIN_CONFIDENCE_LOG,
    MIN_CONFIDENCE_NOTIFY,
    MQTT_HOST,
    MQTT_PASSWORD,
    MQTT_PORT,
    MQTT_USER,
    SNAPSHOT_DIR,
    TOPIC_DETECTION,
    TOPIC_NEW_SPECIES,
    TOPIC_PREDATOR_ALERT,
    TOPIC_UNKNOWN,
)
from inat_client import INatClient
from local_model import LocalBirdModel
from logger import DetectionLogger, DetectionRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("bird-classifier")


class BirdClassifierService:
    """Listens for Frigate events and classifies bird detections."""

    def __init__(self) -> None:
        self._mqtt_client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="bird-classifier",
        )
        self._inat = INatClient()
        self._local_model = LocalBirdModel()
        self._logger = DetectionLogger()
        self._seen_species: set[str] = set()
        self._running = True

    def start(self) -> None:
        """Initialize models and connect to MQTT."""
        log.info("Starting bird classifier service")

        # Load local model
        try:
            self._local_model.load()
        except Exception:
            log.exception("Failed to load local model — will rely on iNaturalist only")

        # Load previously seen species from database
        self._load_seen_species()

        # Set up MQTT
        self._mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        self._mqtt_client.on_connect = self._on_connect
        self._mqtt_client.on_message = self._on_message

        self._mqtt_client.connect(MQTT_HOST, MQTT_PORT)

        # Handle shutdown gracefully
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

        log.info("Connected to MQTT at %s:%d", MQTT_HOST, MQTT_PORT)
        self._mqtt_client.loop_forever()

    def _shutdown(self, signum: int, frame: object) -> None:
        log.info("Shutting down (signal %d)", signum)
        self._running = False
        self._mqtt_client.disconnect()
        sys.exit(0)

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: object,
        flags: mqtt.ConnectFlags,
        rc: mqtt.ReasonCode,
        properties: mqtt.Properties | None = None,
    ) -> None:
        if rc == 0:
            log.info("MQTT connected, subscribing to %s", FRIGATE_EVENTS_TOPIC)
            client.subscribe(FRIGATE_EVENTS_TOPIC)
        else:
            log.error("MQTT connection failed: %s", rc)

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: object,
        msg: mqtt.MQTTMessage,
    ) -> None:
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            log.warning("Invalid JSON in MQTT message")
            return

        self._handle_frigate_event(payload)

    def _handle_frigate_event(self, event: dict) -> None:
        """Process a Frigate event payload."""
        after = event.get("after", {})
        event_type = event.get("type", "")

        label = after.get("label", "")
        event_id = after.get("id", "")

        # Only process events that have ended or are updates with snapshots
        if event_type not in ("new", "update", "end"):
            return

        # Handle predator alerts immediately
        if label == "cat":
            self._publish_predator_alert(after)
            return

        # Only classify birds
        if label != "bird":
            return

        # Only classify if there's a snapshot available
        if not after.get("has_snapshot", False):
            return

        # Only classify on "end" events to avoid duplicate processing
        if event_type != "end":
            return

        log.info("Processing bird detection event: %s", event_id)

        # Download the snapshot from Frigate
        snapshot_path = self._download_snapshot(event_id)
        if snapshot_path is None:
            log.warning("Could not download snapshot for event %s", event_id)
            return

        # Stage 1: Try local model first
        local_results = self._local_model.classify(snapshot_path)
        scientific_name = ""

        if local_results and local_results[0][1] >= LOCAL_MODEL_CONFIDENCE_THRESHOLD:
            species_name = local_results[0][0]
            confidence = local_results[0][1]
            classifier = "tfhub_birds"
            log.info(
                "Local model high confidence: %s (%.2f)", species_name, confidence
            )
        else:
            # Stage 2: Fall back to iNaturalist
            inat_results = self._inat.classify(snapshot_path)
            if inat_results:
                top = inat_results[0]
                species_name = top.species_common
                scientific_name = top.species_scientific
                confidence = top.confidence
                classifier = "inaturalist"
                log.info(
                    "iNaturalist result: %s (%.2f)", species_name, confidence
                )
            elif local_results:
                # iNat failed but we have local results — use them
                species_name = local_results[0][0]
                confidence = local_results[0][1]
                classifier = "tfhub_birds"
                log.info(
                    "Using local model fallback: %s (%.2f)",
                    species_name,
                    confidence,
                )
            else:
                species_name = "Unknown"
                confidence = 0.0
                classifier = "none"

        # Skip low-confidence detections
        if confidence < MIN_CONFIDENCE_LOG:
            log.info(
                "Skipping low-confidence detection: %s (%.2f)",
                species_name,
                confidence,
            )
            return

        # Archive the snapshot
        now = datetime.now(timezone.utc).astimezone()
        archived_path = self._archive_snapshot(
            snapshot_path, species_name, confidence, now
        )

        # Handle unknown/low-confidence detections
        if confidence < MIN_CONFIDENCE_NOTIFY or species_name == "Unknown":
            self._save_for_review(snapshot_path, event_id)
            self._publish_unknown(species_name, confidence, archived_path, event_id, now)
        else:
            # Publish detection
            self._publish_detection(
                species_name,
                scientific_name,
                confidence,
                classifier,
                archived_path,
                event_id,
                now,
            )

            # Check for new species
            if species_name not in self._seen_species:
                self._seen_species.add(species_name)
                self._publish_new_species(
                    species_name, scientific_name, confidence, archived_path, now
                )

        # Log to CSV + SQLite
        record = DetectionRecord(
            timestamp=now.isoformat(),
            species_common=species_name,
            species_scientific=scientific_name,
            confidence=confidence,
            source="visual",
            classifier=classifier,
            snapshot_path=str(archived_path) if archived_path else None,
            frigate_event_id=event_id,
        )
        self._logger.log(record)

    def _download_snapshot(self, event_id: str) -> Path | None:
        """Download a cropped snapshot from Frigate's API."""
        url = f"{FRIGATE_URL}/api/events/{event_id}/snapshot.jpg"
        try:
            response = requests.get(url, params={"crop": 1}, timeout=10)
            response.raise_for_status()

            tmp_path = Path(f"/tmp/snapshot_{event_id}.jpg")
            tmp_path.write_bytes(response.content)
            return tmp_path
        except requests.RequestException:
            log.exception("Failed to download snapshot from Frigate")
            return None

    def _archive_snapshot(
        self,
        snapshot_path: Path,
        species: str,
        confidence: float,
        timestamp: datetime,
    ) -> Path | None:
        """Copy snapshot to the date-organized archive."""
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
        species_slug = _slugify(species)
        filename = f"{time_str}_{species_slug}_{confidence:.2f}.jpg"

        dest_dir = SNAPSHOT_DIR / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        try:
            shutil.copy2(snapshot_path, dest_path)
            return dest_path
        except OSError:
            log.exception("Failed to archive snapshot")
            return None

    def _save_for_review(self, snapshot_path: Path, event_id: str) -> None:
        """Save low-confidence snapshots for human review."""
        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        dest = CORRECTIONS_DIR / f"{event_id}.jpg"
        try:
            shutil.copy2(snapshot_path, dest)
        except OSError:
            log.exception("Failed to save snapshot for review")

    def _publish_detection(
        self,
        species: str,
        scientific: str,
        confidence: float,
        classifier: str,
        snapshot_path: Path | None,
        event_id: str,
        timestamp: datetime,
    ) -> None:
        payload = {
            "timestamp": timestamp.isoformat(),
            "species_common": species,
            "species_scientific": scientific,
            "confidence": confidence,
            "source": "visual",
            "classifier": classifier,
            "snapshot_path": str(snapshot_path) if snapshot_path else None,
            "frigate_event_id": event_id,
            "count": 1,
            "duration_sec": None,
        }
        payload_json = json.dumps(payload)
        species_slug = _slugify(species)

        self._mqtt_client.publish(TOPIC_DETECTION, payload_json, retain=False)
        self._mqtt_client.publish(
            f"{TOPIC_DETECTION}/{species_slug}", payload_json, retain=False
        )

    def _publish_new_species(
        self,
        species: str,
        scientific: str,
        confidence: float,
        snapshot_path: Path | None,
        timestamp: datetime,
    ) -> None:
        payload = json.dumps(
            {
                "timestamp": timestamp.isoformat(),
                "species_common": species,
                "species_scientific": scientific,
                "confidence": confidence,
                "snapshot_path": str(snapshot_path) if snapshot_path else None,
            }
        )
        self._mqtt_client.publish(TOPIC_NEW_SPECIES, payload, retain=False)
        log.info("NEW SPECIES: %s", species)

    def _publish_predator_alert(self, event_data: dict) -> None:
        payload = json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
                "label": event_data.get("label", "cat"),
                "frigate_event_id": event_data.get("id", ""),
            }
        )
        self._mqtt_client.publish(TOPIC_PREDATOR_ALERT, payload, retain=False)
        log.warning("PREDATOR ALERT: %s detected", event_data.get("label"))

    def _publish_unknown(
        self,
        species: str,
        confidence: float,
        snapshot_path: Path | None,
        event_id: str,
        timestamp: datetime,
    ) -> None:
        payload = json.dumps(
            {
                "timestamp": timestamp.isoformat(),
                "species_common": species,
                "confidence": confidence,
                "snapshot_path": str(snapshot_path) if snapshot_path else None,
                "frigate_event_id": event_id,
            }
        )
        self._mqtt_client.publish(TOPIC_UNKNOWN, payload, retain=False)

    def _load_seen_species(self) -> None:
        """Load previously seen species from the database."""
        import sqlite3

        from config import DB_PATH

        if not DB_PATH.exists():
            return
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.execute(
                "SELECT DISTINCT species_common FROM detections "
                "WHERE species_common IS NOT NULL"
            )
            self._seen_species = {row[0] for row in cursor.fetchall()}
            conn.close()
            log.info("Loaded %d previously seen species", len(self._seen_species))
        except sqlite3.Error:
            log.exception("Failed to load seen species from database")


def _slugify(name: str) -> str:
    """Convert a species name to a URL/MQTT-friendly slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def main() -> None:
    service = BirdClassifierService()
    service.start()


if __name__ == "__main__":
    main()
