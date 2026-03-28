# Backyard Bird Feeder Detection System

A local, privacy-first bird detection and species identification system for a backyard feeder. The system reads an RTSP stream from an IP camera, detects birds and predators using YOLOv8, identifies species using a swappable classifier backend, logs all activity to CSV/SQLite, and publishes events to MQTT for downstream automations. Everything runs on a single Ubuntu laptop — no cloud services required.

```
Tapo C200 ──RTSP──▶ bird-detector ──MQTT──▶ birdfeeder/* topics
                   (OpenCV pipeline)          │
                    motion → YOLO             ▼
                    → classifier         CSV / SQLite
```

---

## Quick Start

### Prerequisites

- **Camera:** TP-Link Tapo C200 (1080p, RTSP-capable) or any RTSP camera
- **Compute:** Ubuntu 24.04 laptop on the same Wi-Fi network
- **Software:** Docker, Docker Compose
  ```bash
  sudo apt install docker.io docker-compose-v2
  sudo usermod -aG docker $USER
  ```

### 1. Clone and Configure

```bash
git clone <your-repo-url> backyardBirdwatch
cd backyardBirdwatch
cp .env.example .env
nano .env
```

Required fields in `.env`:

| Variable | Example | Notes |
|----------|---------|-------|
| `CAMERA_IP` | `192.168.1.171` | IP address of your camera |
| `RTSP_USER` | `admin` | Camera RTSP username |
| `RTSP_PASSWORD` | `yourpassword` | Camera RTSP password |
| `MQTT_USER` | `birdfeeder` | Mosquitto username |
| `MQTT_PASSWORD` | a strong password | Mosquitto password |

`VIDEO_SOURCE` is built automatically from the camera credentials as `rtsp://<RTSP_USER>:<RTSP_PASSWORD>@<CAMERA_IP>:554/stream1`. Set it explicitly in `.env` only if you need a different URL.

### 2. Start the Stack

```bash
docker compose -f docker-compose.opencv.yml up -d
```

Mosquitto generates its password file automatically from `MQTT_USER` and `MQTT_PASSWORD` on first start — no manual setup required.

### 3. Verify

```bash
# All services healthy
docker compose -f docker-compose.opencv.yml ps

# Watch detections live
docker compose -f docker-compose.opencv.yml logs -f bird-detector

# Check the database
sqlite3 data/detections.db \
  "SELECT timestamp, species_common, confidence FROM detections ORDER BY id DESC LIMIT 10;"
```

---

## Running Without Docker

Useful for development and testing.

```bash
# Set up a virtual environment
python -m venv venv && source venv/bin/activate
pip install -r bird-detector/requirements.txt

# Run the pipeline
python bird-detector/pipeline.py

# Run with the MJPEG debug viewer at http://localhost:8090
python bird-detector/pipeline.py --debug

# Custom debug port
python bird-detector/pipeline.py --debug --debug-port 8091
```

The pipeline reads `.env` automatically — no need to export variables manually.

---

## Debug Viewer

Pass `--debug` to `pipeline.py` to start an MJPEG stream at `http://localhost:8090`. It reuses the frames already processed by the pipeline (no duplicate YOLO/motion work).

The overlay shows:
- **Green boxes** — motion regions
- **Orange boxes** — bird detections with species + confidence
- **Red boxes** — predator (cat/dog/bear) detections

To expose the viewer when running via Docker, add to `docker-compose.opencv.yml`:

```yaml
bird-detector:
  command: python pipeline.py --debug
  ports:
    - "8090:8090"
```

---

## Classifier Backends

The species classifier is swappable via `CLASSIFIER_BACKEND` in `.env`. The default is `efficientnet` — a model fine-tuned on crops from your own feeder.

| Backend | Setting | Notes |
|---------|---------|-------|
| Fine-tuned EfficientNet-B0 | `efficientnet` | **Recommended.** Train on your own feeder data. |
| Google AIY Birds V1 (TFHub) | `tfhub` | 965-class MobileNet. Confidence too low on feeder crops. |
| BioCLIP (zero-shot) | `bioclip` | Uniform predictions across similar species. Not suitable. |
| HuggingFace NABirds | `nabirds` | Missing common backyard species (Blue Jay, Tufted Titmouse). |

### Training the EfficientNet classifier

```bash
# 1. Collect labeled crops (run YOLO on your footage)
python scripts/extract_yolo_crops.py path/to/feeder_video.mp4

# 2. Sort crops into species folders
mkdir -p training-data/"Blue Jay" training-data/"Tufted Titmouse"
# move crops from yolo-crops/bird/ into the correct folder

# 3. Train (requires ~20+ images per species)
python scripts/train_efficientnet.py --data-dir training-data/

# 4. Model saved to data/models/feeder_birds.pt
```

Set `EFFICIENTNET_MODEL_PATH` in `.env` if you save the model elsewhere.

---

## Testing Scripts

All scripts live in `scripts/` and run without Docker. Activate your venv first.

### Extract YOLO crops from a video

```bash
# Run YOLO on a video — save a cropped JPEG per detection
python scripts/extract_yolo_crops.py path/to/video.mp4

# Sample faster, save annotated full frames too
python scripts/extract_yolo_crops.py video.mp4 --fps 2 --save-frames

# Disable motion gate — run YOLO on every frame
python scripts/extract_yolo_crops.py video.mp4 --no-motion

# Output goes to yolo-crops/bird/ and yolo-crops/cat/ by default
```

### Compare classifier backends on still images

```bash
# Drop bird crop JPEGs into test-photos/ then:
python scripts/test_photos.py                      # compare all backends
python scripts/test_photos.py --backend efficientnet
python scripts/test_photos.py --input-dir yolo-crops/bird/
```

Produces annotated images + `summary.csv` + `inference_log.json` in `test-photos-results/`.

### Test the full pipeline on a video

```bash
# Run motion → YOLO → classifier on a video file
python scripts/test_pipeline.py path/to/video.mp4

# With a specific backend
python scripts/test_pipeline.py video.mp4 --backend efficientnet

# On a folder of pre-cropped images (skips YOLO)
python scripts/test_pipeline.py data/corrections/

# Don't write to the database
python scripts/test_pipeline.py video.mp4 --no-log
```

### Typical workflow

```bash
# 1. Record a clip or use existing footage, extract YOLO crops
python scripts/extract_yolo_crops.py feeder_clip.mp4 --save-frames

# 2. Review yolo-crops/ — sort bird crops into training-data/<Species>/

# 3. Train EfficientNet on your labeled crops
python scripts/train_efficientnet.py --data-dir training-data/

# 4. Evaluate on test images
python scripts/test_photos.py --backend efficientnet

# 5. Set in .env and run the live pipeline
echo "CLASSIFIER_BACKEND=efficientnet" >> .env
python bird-detector/pipeline.py --debug
```

---

## Detection Pipeline

```
Camera RTSP stream
    │
    ▼  (sampled at CAPTURE_FPS, default 5 fps)
MotionDetector (MOG2 background subtractor)
    │
    │  no motion → skip frame
    ▼
BirdDetector (YOLOv8n)
    │
    ├── "bird"     → SpeciesClassifier (efficientnet | tfhub | bioclip | nabirds)
    │                    │
    │                    ├── conf ≥ MIN_CONFIDENCE_LOG (0.3)  → log + MQTT
    │                    └── conf < MIN_CONFIDENCE_LOG        → save to corrections/
    │
    └── "cat/dog/bear"
             │
             ├── area < PREDATOR_MIN_AREA (15000 px²) → demote to "bird" (small-bird false positive)
             └── conf ≥ PREDATOR_MIN_CONFIDENCE (0.70) → predator alert + MQTT
```

---

## Key Configuration

All settings are environment variables (`.env` or Docker Compose):

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_IP` | — | Camera IP address |
| `RTSP_USER` | — | Camera RTSP username |
| `RTSP_PASSWORD` | — | Camera RTSP password |
| `VIDEO_SOURCE` | *(built from above)* | Override RTSP URL or use a video file path |
| `CAPTURE_FPS` | `5` | Frames per second to sample |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model (`yolov8n/s/m.pt`) |
| `YOLO_CONFIDENCE` | `0.25` | YOLO detection threshold |
| `PREDATOR_MIN_AREA` | `15000` | Below this px², relabel predator → bird |
| `PREDATOR_MIN_CONFIDENCE` | `0.70` | Min confidence to fire a predator alert |
| `CLASSIFIER_BACKEND` | `efficientnet` | `efficientnet` \| `tfhub` \| `bioclip` \| `nabirds` |
| `EFFICIENTNET_MODEL_PATH` | `data/models/feeder_birds.pt` | Path to trained EfficientNet checkpoint |
| `MIN_CONFIDENCE_LOG` | `0.3` | Minimum confidence to log a detection |
| `MIN_CONFIDENCE_NOTIFY` | `0.7` | Minimum confidence to publish MQTT notification |
| `DETECTION_COOLDOWN_SEC` | `10` | Seconds before re-logging the same region |
| `DATA_DIR` | `/data` (Docker) | Where to write detections, snapshots, corrections |

See `bird-detector/config.py` for the full list.

---

## Key Commands

```bash
# Live logs
docker compose -f docker-compose.opencv.yml logs -f bird-detector

# Rebuild after code changes
docker compose -f docker-compose.opencv.yml up -d --build

# Restart the detector
docker compose -f docker-compose.opencv.yml restart bird-detector

# Stop everything
docker compose -f docker-compose.opencv.yml down

# Today's detections
sqlite3 data/detections.db \
  "SELECT species_common, COUNT(*) as visits FROM detections \
   WHERE date(timestamp) = date('now') \
   GROUP BY species_common ORDER BY visits DESC;"

# Watch MQTT traffic
docker compose -f docker-compose.opencv.yml exec mosquitto \
  mosquitto_sub -u $MQTT_USER -P $MQTT_PASSWORD -t "birdfeeder/#" -v
```

---

## Data

```
data/
├── snapshots/        # Cropped detections: YYYY-MM-DD/timestamp_species_conf.jpg
├── corrections/      # Low-confidence detections for manual review / retraining
├── models/           # Trained classifier checkpoints (feeder_birds.pt)
├── detections.csv    # Append-only flat log
└── detections.db     # SQLite (indexed, queryable)
```

CSV columns: `timestamp, date, time, species_common, species_scientific, confidence, source, classifier, duration_sec, count, snapshot_path, reviewed, corrected_species`

---

## Documentation

| Document | Contents |
|----------|----------|
| [Architecture](docs/architecture.md) | Pipeline design and data flow |
| [Configuration](docs/configuration.md) | All environment variables |
| [Troubleshooting](docs/troubleshooting.md) | Common issues |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object detection | YOLOv8n (ultralytics) |
| Motion gating | OpenCV MOG2 |
| Species classification | EfficientNet-B0 (fine-tuned) / TFHub / BioCLIP / HuggingFace |
| Video capture | OpenCV |
| MQTT broker | Eclipse Mosquitto |
| Logging | CSV + SQLite |
| Language | Python 3.11+ |
| Containers | Docker Compose |
