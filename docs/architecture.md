# System Architecture

## Overview

```
Tapo C200 ──RTSP──▶ bird-detector ──MQTT──▶ birdfeeder/* topics
                    (OpenCV pipeline)         │
                                              ▼
                                        CSV / SQLite
```

All services run in Docker containers on a single Ubuntu laptop.  No cloud services are required — everything runs locally.

---

## Services

### Mosquitto (MQTT Broker)
- Eclipse Mosquitto 2
- Port 1883
- Authenticated via password file at `mosquitto/config/password_file`
- Persistent storage in `mosquitto/data/`

### bird-detector
- Custom Python 3.11 service (`bird-detector/`)
- Reads directly from the camera RTSP stream via OpenCV
- Runs the full detection pipeline (motion → YOLO → species classifier)
- Publishes detections to `birdfeeder/*` MQTT topics
- Writes every detection to CSV and SQLite

---

## Detection Pipeline

```
RTSP stream (sampled at CAPTURE_FPS)
        │
        ▼
MotionDetector (OpenCV MOG2 background subtractor)
        │
        │  no motion blob in [MOTION_MIN_AREA, MOTION_MAX_AREA] → skip frame
        ▼
BirdDetector (YOLOv8n — COCO classes 14=bird 15=cat 16=dog 21=bear)
        │
        ├── label = "bird"
        │       │
        │       ▼
        │   SpeciesClassifier (backend: tfhub | bioclip | nabirds)
        │       │
        │       ├── conf ≥ MIN_CONFIDENCE_LOG  → log + MQTT publish
        │       └── conf < MIN_CONFIDENCE_LOG  → save crop to corrections/
        │
        └── label = "cat" / "dog" / "bear"
                │
                ├── area < PREDATOR_MIN_AREA   → demote to "bird" (small-bird misclassification)
                └── conf ≥ PREDATOR_MIN_CONFIDENCE → predator alert + MQTT
```

**Why motion gating?**  Running YOLO on every frame at 5 fps is expensive.  The MOG2 background subtractor is cheap and eliminates 90%+ of frames at a static feeder (wind, lighting changes are filtered by area bounds).

**Why demote small predator detections?**  YOLOv8-nano frequently mislabels small songbirds (titmouse, chickadee) as "cat" at low confidence.  A real cat fills 20 000+ px²; a titmouse is 2 000–8 000 px².  Detections below `PREDATOR_MIN_AREA` are reclassified as birds.

---

## Classifier Backends

The species classifier is selected by `CLASSIFIER_BACKEND`:

| Backend | Model | Approach | Notes |
|---------|-------|----------|-------|
| `tfhub` | Google AIY Birds V1 | 965-class softmax | Fast, no GPU needed |
| `bioclip` | imageomics/bioclip | Zero-shot CLIP | Precomputes text embeddings for species list |
| `nabirds` | Any HF AutoModelForImageClassification | Fine-tuned CNN/ViT | Model ID set by `NABIRDS_MODEL` |

All backends accept a `species_list.txt` allowlist to restrict predictions to expected backyard species.

---

## MQTT Topics

| Topic | Description |
|-------|-------------|
| `birdfeeder/detection` | Every confirmed species detection |
| `birdfeeder/detection/{species-slug}` | Per-species topic (e.g. `northern-cardinal`) |
| `birdfeeder/new_species` | First time a species is seen this session |
| `birdfeeder/predator_alert` | Cat/dog/bear detected above confidence threshold |
| `birdfeeder/unknown` | Low-confidence detections saved for review |

---

## Data Storage

```
data/
├── snapshots/
│   └── YYYY-MM-DD/
│       └── {timestamp}_{species-slug}_{confidence}.jpg
├── corrections/        # Low-confidence crops for manual review
├── detections.csv      # Append-only flat log
└── detections.db       # SQLite (indexed by timestamp, species, date)
```

---

## Source Layout

```
bird-detector/
├── pipeline.py           # Main loop: motion → YOLO → classifier → log/MQTT
├── detector.py           # BirdDetector (YOLOv8)
├── motion.py             # MotionDetector (MOG2)
├── classifier.py         # TFHub AIY Birds V1 backend
├── classifier_bioclip.py # BioCLIP zero-shot backend
├── classifier_nabirds.py # HuggingFace backend
├── logger.py             # CSV + SQLite dual-write logger
├── config.py             # All settings from environment variables
└── species_list.txt      # Allowlist of expected backyard species

scripts/
├── extract_yolo_crops.py # Extract YOLO crops from a video for testing
├── test_photos.py        # Compare classifier backends on still images
├── test_pipeline.py      # Run the full pipeline on a video file
├── debug_view.py         # Live debug overlay (MJPEG stream on :8090)
└── export_csv.py         # Export detections to CSV

docker-compose.opencv.yml       # Main stack (mosquitto + bird-detector)
docker-compose.opencv.test.yml  # Test overlay (adds mediamtx RTSP loop)
```

---

## Network Ports

| Port | Service | Purpose |
|------|---------|---------|
| 1883 | Mosquitto | MQTT |
| 8090 | debug-viewer | Live MJPEG debug stream (debug profile only) |
| 8556 | mediamtx | RTSP test stream (test overlay only) |
