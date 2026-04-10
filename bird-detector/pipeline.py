"""Main detection pipeline.

Loop:
  1. Read frame from RTSP / video file at CAPTURE_FPS.
  2. MotionDetector flags frames with significant foreground movement.
  3. On motion → BirdDetector (YOLOv8n) scans the full frame for bird/cat.
  4. For each 'bird' detection → species classifier (backend set by
     CLASSIFIER_BACKEND: 'tfhub' | 'bioclip' | 'nabirds' | 'efficientnet').
  5. Detections are fed into BirdTracker, which associates them across frames
     using IoU overlap and accumulates per-species classification votes.
  6. When a track closes (bird leaves frame), the winner is logged to
     CSV + SQLite and published to MQTT — one entry per visit, not per frame.

Run standalone (no Docker):
    python pipeline.py
    python pipeline.py --debug          # MJPEG viewer at http://localhost:8090
    python pipeline.py --debug-port 8091

Or via Docker Compose (see docker-compose.opencv.yml).
"""

import argparse
import json
import logging
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import config
from detector import BirdDetector, Detection
from logger import DetectionLogger, DetectionRecord
from motion import MotionDetector
from tracker import BirdTracker, ClosedTrack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── MJPEG debug server (optional) ─────────────────────────────────────────

_FONT           = cv2.FONT_HERSHEY_SIMPLEX
_COLOR_BIRD     = (255, 120,   0)
_COLOR_PREDATOR = (  0,   0, 220)
_COLOR_MOTION   = (  0, 210,   0)

_INDEX_HTML = b"""<!DOCTYPE html><html>
<head><title>Bird Detector</title>
<style>body{margin:0;background:#111;display:flex;flex-direction:column;
align-items:center;justify-content:center;min-height:100vh}
img{max-width:100%}p{color:#888;font-family:monospace;margin:8px 0 0;font-size:13px}
</style></head>
<body><img src="/stream">
<p>&#9646;&nbsp;green&nbsp;=&nbsp;motion&nbsp;&nbsp;
&#9646;&nbsp;orange&nbsp;=&nbsp;bird&nbsp;&nbsp;
&#9646;&nbsp;red&nbsp;=&nbsp;predator</p></body></html>"""


class _MJPEGServer:
    """Serves annotated frames as an MJPEG stream on a background thread."""

    def __init__(self, port: int) -> None:
        self._port = port
        self._jpeg: bytes = b""
        self._seq: int = 0
        self._cond = threading.Condition()

    def start(self) -> None:
        server = ThreadingHTTPServer(("0.0.0.0", self._port), self._make_handler())
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info("Debug viewer → http://localhost:%d/", self._port)

    def push(self, frame: np.ndarray) -> None:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def _make_handler(self):
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_INDEX_HTML)))
                    self.end_headers()
                    self.wfile.write(_INDEX_HTML)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    last = -1
                    try:
                        while True:
                            with srv._cond:
                                srv._cond.wait_for(
                                    lambda: srv._seq != last, timeout=2.0
                                )
                                jpeg, last = srv._jpeg, srv._seq
                            if not jpeg:
                                continue
                            part = (
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.write(part)
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

            def log_message(self, *_):
                pass

        return _Handler


def _draw_debug(
    frame: np.ndarray,
    motion_regions: list,
    detections: list[Detection],
    labels: dict,  # Detection → display string
) -> np.ndarray:
    display = frame.copy()
    for region in motion_regions:
        cv2.rectangle(display, (region.x, region.y), (region.x2, region.y2), _COLOR_MOTION, 1)
    for det in detections:
        color = _COLOR_PREDATOR if det.label != "bird" else _COLOR_BIRD
        cv2.rectangle(display, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        text = labels.get(det, f"{det.label} {det.confidence:.2f}")
        (tw, th), bl = cv2.getTextSize(text, _FONT, 0.52, 1)
        bg_top = max(det.y1 - th - bl - 4, 0)
        cv2.rectangle(display, (det.x1, bg_top), (det.x1 + tw + 4, det.y1), color, -1)
        cv2.putText(
            display, text, (det.x1 + 2, det.y1 - bl - 2),
            _FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return display


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

def _save_track_snapshot(track: ClosedTrack) -> str | None:
    """Save the best crop from a closed track.  Returns relative path."""
    if track.crop is None or track.species_common is None:
        return None
    try:
        date_str = track.first_seen[:10]  # YYYY-MM-DD
        day_dir = config.SNAPSHOT_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        slug = track.species_common.lower().replace(" ", "-")
        safe_ts = track.first_seen.replace(":", "-").replace("+", "")
        filename = f"{safe_ts}_{slug}_{track.confidence:.2f}.jpg"
        path = day_dir / filename

        cv2.imwrite(
            str(path),
            track.crop,
            [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_JPEG_QUALITY],
        )
        return str(path.relative_to(config.DATA_DIR))
    except Exception:
        logger.exception("Failed to save track snapshot")
        return None


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

    def release(self) -> None:
        self._cap.release()


# ── Classifier factory ─────────────────────────────────────────────────────

def _make_classifier():
    """Instantiate the species classifier selected by CLASSIFIER_BACKEND."""
    backend = config.CLASSIFIER_BACKEND.lower()
    if backend == "bioclip":
        from classifier_bioclip import BioCLIPClassifier
        logger.info("Using classifier backend: BioCLIP (%s)", config.BIOCLIP_MODEL)
        return BioCLIPClassifier(
            species_list_path=config.SPECIES_LIST_PATH,
            model_name=config.BIOCLIP_MODEL,
        )
    elif backend == "nabirds":
        from classifier_nabirds import NABirdsClassifier
        logger.info("Using classifier backend: NABirds/HF (%s)", config.NABIRDS_MODEL)
        return NABirdsClassifier(
            model_name=config.NABIRDS_MODEL,
            species_list_path=config.SPECIES_LIST_PATH,
        )
    elif backend == "efficientnet":
        from classifier_efficientnet import EfficientNetClassifier
        logger.info("Using classifier backend: EfficientNet (%s)", config.EFFICIENTNET_MODEL_PATH)
        return EfficientNetClassifier(model_path=config.EFFICIENTNET_MODEL_PATH)
    else:
        if backend != "tfhub":
            logger.warning(
                "Unknown CLASSIFIER_BACKEND %r — falling back to tfhub", backend
            )
        from classifier import SpeciesClassifier
        logger.info("Using classifier backend: TFHub AIY Birds V1")
        return SpeciesClassifier(species_list_path=config.SPECIES_LIST_PATH)


# ── Closed-track handlers ─────────────────────────────────────────────────

def _handle_closed_track(
    track: ClosedTrack,
    det_logger: DetectionLogger,
    mqtt,
) -> None:
    """Log and publish a finalized track."""
    if track.is_unknown or track.species_common is None:
        _handle_unknown_track(track, mqtt)
        return

    snap = _save_track_snapshot(track)
    record = DetectionRecord(
        timestamp=track.first_seen,
        species_common=track.species_common,
        confidence=track.confidence,
        classifier=config.CLASSIFIER_BACKEND,
        snapshot_path=snap,
        duration_sec=track.duration_sec,
    )
    det_logger.log(record)

    payload = {
        "timestamp": track.first_seen,
        "species_common": track.species_common,
        "confidence": track.confidence,
        "classifier": config.CLASSIFIER_BACKEND,
        "snapshot_path": snap,
        "yolo_confidence": track.yolo_confidence,
        "duration_sec": track.duration_sec,
        "frame_count": track.frame_count,
        "vote_summary": track.vote_summary,
        "track_id": track.track_id,
    }
    _mqtt_publish(mqtt, config.TOPIC_DETECTION, payload)

    if track.confidence >= config.MIN_CONFIDENCE_NOTIFY:
        slug = track.species_common.lower().replace(" ", "-")
        _mqtt_publish(mqtt, f"{config.TOPIC_DETECTION}/{slug}", payload)


def _handle_unknown_track(track: ClosedTrack, mqtt) -> None:
    logger.info(
        "Low-confidence track #%d (conf=%.2f, %d frames) — saving to corrections/",
        track.track_id, track.confidence, track.frame_count,
    )
    if track.crop is not None:
        try:
            config.CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
            safe_ts = track.first_seen.replace(":", "-").replace("+", "")
            best = (track.species_common or "unknown").lower().replace(" ", "-")
            path = config.CORRECTIONS_DIR / f"{safe_ts}_{best}_{track.confidence:.2f}.jpg"
            cv2.imwrite(str(path), track.crop, [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_JPEG_QUALITY])
        except Exception:
            logger.exception("Failed to save unknown track image")
    _mqtt_publish(mqtt, config.TOPIC_UNKNOWN, {
        "timestamp": track.first_seen,
        "confidence": track.confidence,
        "track_id": track.track_id,
        "vote_summary": track.vote_summary,
    })


def _handle_predator(
    det: Detection,
    frame: np.ndarray,
    ts: str,
    det_logger: DetectionLogger,
    mqtt,
) -> None:
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


def _save_snapshot(frame: np.ndarray, det: Detection, ts: str) -> str | None:
    """Save a full-frame crop for a non-bird detection (predators)."""
    try:
        date_str = ts[:10]
        day_dir = config.SNAPSHOT_DIR / date_str
        day_dir.mkdir(parents=True, exist_ok=True)
        slug = det.label.replace(" ", "-")
        safe_ts = ts.replace(":", "-").replace("+", "")
        filename = f"{safe_ts}_{slug}_{det.confidence:.2f}.jpg"
        path = day_dir / filename
        crop = frame[det.y1 : det.y2, det.x1 : det.x2]
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, config.SNAPSHOT_JPEG_QUALITY])
        return str(path.relative_to(config.DATA_DIR))
    except Exception:
        logger.exception("Failed to save snapshot")
        return None


# ── Main loop ─────────────────────────────────────────────────────────────

def run(debug_port: int | None = None) -> None:
    if not config.VIDEO_SOURCE:
        raise SystemExit("VIDEO_SOURCE is not set. Pass an RTSP URL or video file path.")

    mjpeg: _MJPEGServer | None = None
    if debug_port is not None:
        mjpeg = _MJPEGServer(debug_port)
        mjpeg.start()

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
        predator_min_area=config.PREDATOR_MIN_AREA,
    )
    classifier = _make_classifier()
    det_logger = DetectionLogger()
    tracker = BirdTracker(
        iou_threshold=config.TRACKER_IOU_THRESHOLD,
        max_gap_sec=config.TRACKER_MAX_GAP_SEC,
        min_frames_confirm=config.TRACKER_MIN_FRAMES_CONFIRM,
        min_vote_confidence=config.TRACKER_MIN_VOTE_CONFIDENCE,
    )
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

        stream_ended = False
        try:
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

                if mjpeg is not None and not motion_regions:
                    mjpeg.push(frame)

                if not motion_regions:
                    continue

                logger.debug(
                    "Frame %d: %d motion region(s)", loop_count, len(motion_regions)
                )

                ts = datetime.now(timezone.utc).isoformat()
                detections = bird_detector.detect(frame)
                debug_labels: dict = {}

                # Collect bird detections with their classifications for the tracker.
                # Predators are handled immediately (alert-worthy, no need to debounce).
                bird_dets: list[tuple[Detection, str, float, np.ndarray]] = []

                for det in detections:
                    if det.label != "bird":
                        debug_labels[det] = f"{det.label} {det.confidence:.2f}"
                        if det.confidence >= config.PREDATOR_MIN_CONFIDENCE:
                            _handle_predator(det, frame, ts, det_logger, mqtt)
                        else:
                            logger.debug(
                                "Non-bird below predator threshold: "
                                "%s conf=%.2f — skipping",
                                det.label, det.confidence,
                            )
                        continue

                    # Bird — run species classifier on the crop
                    crop = frame[det.y1 : det.y2, det.x1 : det.x2]
                    if crop.size == 0:
                        continue
                    top = classifier.classify(crop)
                    if not top:
                        debug_labels[det] = f"bird {det.confidence:.2f}"
                        continue

                    species, conf = top[0]
                    debug_labels[det] = f"{species} {conf:.0%}"
                    logger.debug(
                        "Bird frame: %s  yolo=%.2f  species_conf=%.2f",
                        species, det.confidence, conf,
                    )
                    bird_dets.append((det, species, conf, crop))

                # Feed detections into the tracker
                closed_tracks, det_to_tid = tracker.update(bird_dets, ts)

                # Annotate debug labels with track IDs
                for d_idx, (det, species, conf, _) in enumerate(bird_dets):
                    tid = det_to_tid.get(d_idx)
                    if tid is not None:
                        debug_labels[det] = f"#{tid} {species} {conf:.0%}"

                # Log any tracks that just closed
                for track in closed_tracks:
                    _handle_closed_track(track, det_logger, mqtt)

                if mjpeg is not None:
                    mjpeg.push(_draw_debug(frame, motion_regions, detections, debug_labels))

            # for-loop exited normally → stream ended (StopIteration is swallowed
            # by the for-statement and never reaches except StopIteration below)
            stream_ended = True

        except Exception:
            logger.exception("Unexpected error in capture loop — restarting in 5 s")
            stream_ended = False
            time.sleep(5)
        finally:
            cap.release()

        if stream_ended:
            logger.info("End of video stream — flushing active tracks …")
            for track in tracker.flush():
                _handle_closed_track(track, det_logger, mqtt)
            if not config.VIDEO_LOOP:
                logger.info("VIDEO_LOOP=false — exiting")
                break
            logger.info("Restarting stream …")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backyard bird detection pipeline.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable MJPEG debug viewer (default port 8090)",
    )
    parser.add_argument(
        "--debug-port",
        type=int,
        default=int(__import__("os").environ.get("DEBUG_PORT", "8090")),
        metavar="PORT",
        help="Port for the MJPEG debug viewer (default: 8090)",
    )
    args = parser.parse_args()
    run(debug_port=args.debug_port if args.debug else None)


if __name__ == "__main__":
    main()
