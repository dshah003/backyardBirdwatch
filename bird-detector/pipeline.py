"""Main detection pipeline.

Loop:
  1. Read frame from RTSP / video file at CAPTURE_FPS.
  2. MotionDetector flags frames with significant foreground movement.
  3. On motion → BirdDetector (YOLOv8n) scans the full frame for bird/cat.
  4. For each 'bird' detection → SpeciesClassifier (TFHub AIY Birds V1).
  5. Results above MIN_CONFIDENCE_LOG are written to CSV + SQLite and
     published to MQTT.  A per-region cooldown prevents duplicate entries
     for the same bird sitting at the feeder.

Run standalone (no Docker):
    VIDEO_SOURCE=path/to/video.mp4 python pipeline.py

Or via Docker Compose (see docker-compose.opencv.yml).
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config
from classifier import SpeciesClassifier
from detector import BirdDetector, Detection
from logger import DetectionLogger, DetectionRecord
from motion import MotionDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── MQTT (optional — gracefully degraded if broker unavailable) ────────────

def _make_mqtt_client():
    """Return a connected paho MQTT client, or None on failure."""
    try:
        import paho.mqtt.client as mqtt  # type: ignore

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if config.MQTT_USER:
            client.username_pw_set(config.MQTT_USER, config.MQTT_PASSWORD)
        client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=60)
        client.loop_start()
        logger.info("MQTT connected to %s:%d", config.MQTT_HOST, config.MQTT_PORT)
        return client
    except Exception:
        logger.warning("MQTT unavailable — running without broker notifications")
        return None


def _mqtt_publish(client, topic: str, payload: dict) -> None:
    if client is None:
        return
    try:
        client.publish(topic, json.dumps(payload), qos=1)
    except Exception:
        logger.debug("MQTT publish failed", exc_info=True)


# ── Snapshot saving ────────────────────────────────────────────────────────

def _save_snapshot(frame: np.ndarray, det: Detection, ts: str) -> str | None:
    """Crop the detection region and save as JPEG.  Returns relative path."""
    try:
        config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        date_str = ts[:10]  # YYYY-MM-DD
        day_dir = config.SNAPSHOT_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        slug = det.label.replace(" ", "-")
        safe_ts = ts.replace(":", "-").replace("+", "")
        filename = f"{safe_ts}_{slug}_{det.confidence:.2f}.jpg"
        path = day_dir / filename

        crop = frame[det.y1 : det.y2, det.x1 : det.x2]
        cv2.imwrite(
            str(path),
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_JPEG_QUALITY],
        )
        return str(path.relative_to(config.DATA_DIR))
    except Exception:
        logger.exception("Failed to save snapshot")
        return None


# ── Cooldown tracker ───────────────────────────────────────────────────────

class _CooldownTracker:
    """Per-region detection cooldown to suppress duplicate log entries."""

    def __init__(self, cooldown_sec: float) -> None:
        self._cooldown = cooldown_sec
        # key: (x1//50, y1//50) bucket → last-logged timestamp
        self._last: dict[tuple[int, int], float] = {}

    def is_ready(self, det: Detection) -> bool:
        key = (det.x1 // 50, det.y1 // 50)
        now = time.monotonic()
        if now - self._last.get(key, 0.0) >= self._cooldown:
            self._last[key] = now
            return True
        return False


# ── Video capture with frame-rate throttle ────────────────────────────────

class _ThrottledCapture:
    """Wrap cv2.VideoCapture and yield frames at approximately target_fps."""

    def __init__(self, source: str, target_fps: float) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")
        self._interval = 1.0 / max(target_fps, 1)
        self._next_t = time.monotonic()
        logger.info("Opened video source: %s  target_fps=%s", source, target_fps)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        while True:
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
            now = time.monotonic()
            if now >= self._next_t:
                self._next_t = now + self._interval
                return frame
            # drop the frame and grab the next one immediately
            # (VideoCapture buffers; just keep reading until in sync)

    def release(self) -> None:
        self._cap.release()


# ── Main loop ─────────────────────────────────────────────────────────────

def run() -> None:
    if not config.VIDEO_SOURCE:
        raise SystemExit("VIDEO_SOURCE is not set. Pass an RTSP URL or video file path.")

    motion_detector = MotionDetector(
        history=config.MOTION_HISTORY,
        var_threshold=config.MOTION_VAR_THRESHOLD,
        min_area=config.MOTION_MIN_AREA,
        max_area=config.MOTION_MAX_AREA,
        dilate_iterations=config.MOTION_DILATE_ITER,
    )
    bird_detector = BirdDetector(
        model_name=config.YOLO_MODEL,
        confidence=config.YOLO_CONFIDENCE,
        min_area=config.DETECTION_MIN_AREA,
        max_area=config.DETECTION_MAX_AREA,
    )
    classifier = SpeciesClassifier(species_list_path=config.SPECIES_LIST_PATH)
    det_logger = DetectionLogger()
    cooldown = _CooldownTracker(config.DETECTION_COOLDOWN_SEC)
    mqtt = _make_mqtt_client()

    bird_detector.load()
    classifier.load()

    loop_count = 0
    while True:
        try:
            cap = _ThrottledCapture(config.VIDEO_SOURCE, config.CAPTURE_FPS)
        except RuntimeError as exc:
            logger.error("%s — retrying in 5 s", exc)
            time.sleep(5)
            continue

        try:
            # Seed the background model before entering the detection loop.
            logger.info("Warming up motion background model …")
            first_frame = None
            for frame in cap:
                first_frame = frame
                break
            if first_frame is not None:
                motion_detector.warmup(first_frame)

            for frame in cap:
                loop_count += 1
                motion_regions = motion_detector.detect(frame)
                if not motion_regions:
                    continue

                logger.debug(
                    "Frame %d: %d motion region(s)", loop_count, len(motion_regions)
                )

                detections = bird_detector.detect(frame)
                for det in detections:
                    ts = datetime.now(timezone.utc).isoformat()

                    if det.label != "bird":
                        if det.confidence >= config.PREDATOR_MIN_CONFIDENCE:
                            _handle_predator(det, frame, ts, det_logger, mqtt)
                        else:
                            logger.debug(
                                "Non-bird detection below predator threshold: "
                                "%s conf=%.2f (threshold=%.2f) — skipping",
                                det.label, det.confidence, config.PREDATOR_MIN_CONFIDENCE,
                            )
                        continue

                    # Bird — run species classifier
                    crop = frame[det.y1 : det.y2, det.x1 : det.x2]
                    top = classifier.classify(crop)
                    if not top:
                        logger.debug("Classifier returned no results for detection")
                        continue

                    species, conf = top[0]
                    logger.info(
                        "Bird detected: %s  yolo_conf=%.2f  species_conf=%.2f",
                        species,
                        det.confidence,
                        conf,
                    )

                    if conf < config.MIN_CONFIDENCE_LOG:
                        _handle_unknown(det, frame, ts, conf, det_logger, mqtt)
                        continue

                    if not cooldown.is_ready(det):
                        logger.debug("Cooldown active for region (%d,%d)", det.x1, det.y1)
                        continue

                    snap = _save_snapshot(frame, det, ts)
                    record = DetectionRecord(
                        timestamp=ts,
                        species_common=species,
                        confidence=conf,
                        classifier="tfhub_birds",
                        snapshot_path=snap,
                    )
                    det_logger.log(record)

                    payload = {
                        "timestamp": ts,
                        "species_common": species,
                        "confidence": conf,
                        "classifier": "tfhub_birds",
                        "snapshot_path": snap,
                        "yolo_confidence": det.confidence,
                        "bbox": [det.x1, det.y1, det.x2, det.y2],
                    }
                    _mqtt_publish(mqtt, config.TOPIC_DETECTION, payload)

                    if conf >= config.MIN_CONFIDENCE_NOTIFY:
                        slug = species.lower().replace(" ", "-")
                        _mqtt_publish(mqtt, f"{config.TOPIC_DETECTION}/{slug}", payload)

        except StopIteration:
            logger.info("End of video stream — restarting …")
            if not config.VIDEO_LOOP:
                logger.info("VIDEO_LOOP=false — exiting")
                break
        except Exception:
            logger.exception("Unexpected error in capture loop — restarting in 5 s")
            time.sleep(5)
        finally:
            cap.release()


def _handle_predator(det: Detection, frame: np.ndarray, ts: str, det_logger: DetectionLogger, mqtt) -> None:
    logger.warning("PREDATOR detected: %s  conf=%.2f", det.label, det.confidence)
    snap = _save_snapshot(frame, det, ts)
    record = DetectionRecord(
        timestamp=ts,
        species_common=det.label,
        confidence=det.confidence,
        classifier="yolov8",
        snapshot_path=snap,
    )
    det_logger.log(record)
    payload = {
        "timestamp": ts,
        "label": det.label,
        "confidence": det.confidence,
        "snapshot_path": snap,
        "bbox": [det.x1, det.y1, det.x2, det.y2],
    }
    _mqtt_publish(mqtt, config.TOPIC_PREDATOR_ALERT, payload)


def _handle_unknown(det: Detection, frame: np.ndarray, ts: str, conf: float, det_logger: DetectionLogger, mqtt) -> None:
    logger.info("Low-confidence detection (%.2f) — saving to corrections/", conf)
    try:
        config.CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        safe_ts = ts.replace(":", "-").replace("+", "")
        path = config.CORRECTIONS_DIR / f"{safe_ts}_unknown_{conf:.2f}.jpg"
        crop = frame[det.y1 : det.y2, det.x1 : det.x2]
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_JPEG_QUALITY])
    except Exception:
        logger.exception("Failed to save unknown detection image")
    _mqtt_publish(mqtt, config.TOPIC_UNKNOWN, {"timestamp": ts, "confidence": conf})


if __name__ == "__main__":
    run()
