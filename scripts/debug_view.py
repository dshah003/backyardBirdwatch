"""MJPEG debug viewer — draws YOLO bounding boxes and motion regions.

Serves annotated frames as an MJPEG stream at http://localhost:8090
so it can run inside Docker without a display.

Threading model:
  - Capture thread:   reads VideoCapture as fast as possible, keeps only the
                      latest raw frame.  Prevents OpenCV buffer from going stale.
  - Processing loop:  grabs the latest raw frame, runs motion+YOLO+encode, then
                      signals the HTTP handlers via a Condition.
  - HTTP handlers:    block on the Condition and send each new JPEG exactly once.

Configuration (env vars):
  VIDEO_SOURCE          RTSP URL or video file path (required)
  DEBUG_PORT            HTTP port (default 8090)
  DEBUG_CLASSIFY        Set to "true" to run TFHub species classifier (slower)
  YOLO_MODEL            YOLOv8 model variant (default yolov8n.pt)
  YOLO_CONFIDENCE       Detection threshold (default 0.25)
  CAPTURE_FPS           Frames to process per second (default 10)
  MOTION_VAR_THRESHOLD  MOG2 sensitivity (default 32)
  MOTION_MIN_AREA       Min motion blob size in px² (default 800)
  DEBUG_MAX_WIDTH       Downscale stream to this width before YOLO + encode
                        (default 1280; use 640 on slow hardware)
"""

import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Bootstrap: load .env and build VIDEO_SOURCE via config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bird-detector"))
import config  # noqa: E402 — populates os.environ from .env

# ── Config ────────────────────────────────────────────────────────────────────

VIDEO_SOURCE: str = config.VIDEO_SOURCE or os.environ.get("VIDEO_SOURCE", "")
DEBUG_PORT: int = int(os.environ.get("DEBUG_PORT", "8090"))
DEBUG_CLASSIFY: bool = os.environ.get("DEBUG_CLASSIFY", "").lower() == "true"
YOLO_MODEL: str = os.environ.get("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE: float = float(os.environ.get("YOLO_CONFIDENCE", "0.25"))
CAPTURE_FPS: float = float(os.environ.get("CAPTURE_FPS", "10"))
MOTION_VAR_THRESHOLD: float = float(os.environ.get("MOTION_VAR_THRESHOLD", "32"))
MOTION_MIN_AREA: int = int(os.environ.get("MOTION_MIN_AREA", "800"))
DEBUG_MAX_WIDTH: int = int(os.environ.get("DEBUG_MAX_WIDTH", "1280"))

COCO_CLASSES = {14: "bird", 15: "cat"}
COLOR_BIRD = (255, 120, 0)
COLOR_CAT  = (0, 0, 220)
COLOR_MOTION = (0, 210, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Shared state ──────────────────────────────────────────────────────────────

# Latest raw frame from the capture thread.
_raw_frame: Optional[np.ndarray] = None
_raw_lock = threading.Lock()

# Latest encoded JPEG for the HTTP handlers.
# Handlers wait on _frame_ready and compare sequence numbers so each new frame
# is sent exactly once (no duplicates, no skips).
_latest_jpeg: bytes = b""
_frame_seq: int = 0
_frame_ready = threading.Condition()


# ── Capture thread ────────────────────────────────────────────────────────────

class _FrameGrabber(threading.Thread):
    """Continuously drains VideoCapture so _raw_frame is always fresh."""

    def __init__(self, source: str) -> None:
        super().__init__(daemon=True)
        self._source = source

    def run(self) -> None:
        global _raw_frame
        while True:
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                print(f"[capture] Cannot open {self._source!r} — retrying in 5 s")
                time.sleep(5)
                continue

            print(f"[capture] Opened {self._source!r}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    # End of file — rewind; for live RTSP, reconnect.
                    if self._source.lower().startswith("rtsp"):
                        print("[capture] RTSP dropped — reconnecting …")
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                with _raw_lock:
                    _raw_frame = frame

            cap.release()
            time.sleep(2)


# ── MJPEG HTTP server ─────────────────────────────────────────────────────────

_INDEX_HTML = b"""<!DOCTYPE html>
<html>
<head>
  <title>Bird Detector Debug</title>
  <style>
    body { margin: 0; background: #111; display: flex; flex-direction: column;
           justify-content: center; align-items: center; min-height: 100vh; }
    img  { max-width: 100%; }
    p    { color: #888; font-family: monospace; margin: 8px 0 0; font-size: 13px; }
  </style>
</head>
<body>
  <img src="/stream">
  <p>&#9646; green = motion &nbsp; &#9646; orange = bird &nbsp; &#9646; red = cat</p>
</body>
</html>"""


class _MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/stream":
            self._stream()
        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
        else:
            self.send_error(404)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_seq = -1
        try:
            while True:
                # Block until the processing loop produces a new frame.
                with _frame_ready:
                    _frame_ready.wait_for(lambda: _frame_seq != last_seq, timeout=2.0)
                    jpeg = _latest_jpeg
                    last_seq = _frame_seq

                if not jpeg:
                    continue

                part = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.write(part)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_args) -> None:
        pass


def _start_server(port: int) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), _MJPEGHandler)
    server.serve_forever()


# ── Optional species classifier ───────────────────────────────────────────────

class _Classifier:
    MODEL_URL  = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
    LABELS_URL = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"

    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []

    def load(self) -> None:
        import requests
        import tensorflow_hub as hub

        print("Loading TFHub species classifier …")
        self._model = hub.KerasLayer(self.MODEL_URL)
        resp = requests.get(self.LABELS_URL, timeout=30)
        resp.raise_for_status()
        for line in resp.text.strip().split("\n")[1:]:
            parts = line.split(",", 1)
            self._labels.append(parts[1].strip() if len(parts) == 2 else "Unknown")
        print(f"Classifier ready — {len(self._labels)} labels")

    def top1(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if self._model is None or crop_bgr.size == 0:
            return "Unknown", 0.0
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            t = np.expand_dims(cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0, 0)
            probs = self._model(t)[0].numpy()
            idx = int(np.argmax(probs))
            return (self._labels[idx] if idx < len(self._labels) else "Unknown"), float(probs[idx])
        except Exception as exc:
            print(f"Classifier error: {exc}")
            return "Unknown", 0.0


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_label(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    text: str,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), bl = cv2.getTextSize(text, FONT, 0.52, 1)
    bg_top = max(y1 - th - bl - 4, 0)
    cv2.rectangle(frame, (x1, bg_top), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - bl - 2), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def _motion_contours(fg_mask: np.ndarray) -> list:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(fg_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MOTION_MIN_AREA]


def _maybe_scale(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


# ── Processing loop ───────────────────────────────────────────────────────────

def run() -> None:
    global _latest_jpeg, _frame_seq

    if not VIDEO_SOURCE:
        sys.exit("VIDEO_SOURCE is not set.")

    from ultralytics import YOLO

    yolo = YOLO(YOLO_MODEL)
    yolo(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)

    classifier = _Classifier()
    if DEBUG_CLASSIFY:
        classifier.load()

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=MOTION_VAR_THRESHOLD)

    # Start capture thread and HTTP server.
    _FrameGrabber(VIDEO_SOURCE).start()
    threading.Thread(target=_start_server, args=(DEBUG_PORT,), daemon=True).start()
    print(f"Debug stream → http://localhost:{DEBUG_PORT}/")

    interval = 1.0 / CAPTURE_FPS
    next_t = time.monotonic()

    while True:
        # Throttle: sleep until it's time for the next frame.
        now = time.monotonic()
        if now < next_t:
            time.sleep(next_t - now)
        next_t = time.monotonic() + interval

        # Grab the freshest raw frame.
        with _raw_lock:
            frame = _raw_frame
        if frame is None:
            continue

        frame = _maybe_scale(frame, DEBUG_MAX_WIDTH)
        display = frame.copy()

        # Motion
        fg_mask = bg_sub.apply(frame)
        contours = _motion_contours(fg_mask)
        cv2.drawContours(display, contours, -1, COLOR_MOTION, 1)

        # YOLO
        results = yolo(frame, conf=YOLO_CONFIDENCE, classes=list(COCO_CLASSES.keys()), verbose=False)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = COCO_CLASSES.get(cls_id, "?")
                yolo_conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = COLOR_CAT if label == "cat" else COLOR_BIRD

                species_text = ""
                if DEBUG_CLASSIFY and label == "bird":
                    species, sp_conf = classifier.top1(frame[y1:y2, x1:x2])
                    species_text = f" {species} ({sp_conf:.2f})"

                _draw_label(display, x1, y1, x2, y2,
                            f"{label} {yolo_conf:.2f}{species_text}", color)

        # Status chip
        cv2.putText(display,
                    "MOTION" if contours else "still",
                    (8, display.shape[0] - 8),
                    FONT, 0.5,
                    COLOR_MOTION if contours else (100, 100, 100),
                    1, cv2.LINE_AA)

        # Encode and notify all waiting HTTP handlers.
        _, jpeg_buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _frame_ready:
            _latest_jpeg = jpeg_buf.tobytes()
            _frame_seq += 1
            _frame_ready.notify_all()


if __name__ == "__main__":
    run()
