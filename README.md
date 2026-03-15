# Backyard Bird Feeder Detection System

A local, privacy-first bird detection and species identification system for a backyard feeder. The system reads an RTSP stream from an IP camera, detects birds and predators using YOLOv8, identifies species using a swappable classifier backend, logs all activity to CSV/SQLite, and publishes events to MQTT for downstream automations.  Everything runs on a single Ubuntu laptop — no cloud services required.

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

| Variable | Example |
|----------|---------|
| `VIDEO_SOURCE` | `rtsp://user:pass@192.168.1.100:554/stream2` |
| `MQTT_USER` | `birdfeeder` |
| `MQTT_PASSWORD` | a strong password |

### 2. Start the Stack

```bash
# Create the Mosquitto password file before first start
touch mosquitto/config/password_file
chmod 644 mosquitto/config/password_file

docker compose -f docker-compose.opencv.yml up -d

# Add the MQTT user
docker exec -it backyardbirdwatch-mosquitto-1 \
  mosquitto_passwd /mosquitto/config/password_file birdfeeder

docker exec -u root backyardbirdwatch-mosquitto-1 \
  chown mosquitto:mosquitto /mosquitto/config/password_file
docker exec -u root backyardbirdwatch-mosquitto-1 \
  chmod 0700 /mosquitto/config/password_file

docker compose -f docker-compose.opencv.yml restart mosquitto
```

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

## Testing with a Sample Video

```bash
# Loop a local video through mediamtx as a fake RTSP stream
cp /path/to/footage.mp4 test-videos/sample.mp4
docker compose -f docker-compose.opencv.yml -f docker-compose.opencv.test.yml up -d

# Tail results
tail -f data/detections.csv
```

---

## Classifier Backends

The species classifier is swappable via `CLASSIFIER_BACKEND` in `.env` (or the Docker Compose env):

| Backend | Setting | Accuracy | Extra deps |
|---------|---------|----------|------------|
| Google AIY Birds V1 (TFHub) | `tfhub` | Good | `tensorflow tensorflow-hub` |
| BioCLIP (zero-shot) | `bioclip` | Better | `open-clip-torch` |
| HuggingFace ViT/EfficientNet | `nabirds` | Best | `transformers torch` |

```bash
# In .env
CLASSIFIER_BACKEND=bioclip

# Or per-run without Docker
CLASSIFIER_BACKEND=bioclip python bird-detector/pipeline.py
```

The `NABIRDS_MODEL` variable selects which HuggingFace checkpoint to use (default: `chriamue/bird-species-classifier`).

---

## Testing Scripts

All scripts live in `scripts/` and run without Docker.

### Extract YOLO crops from a video

```bash
# Run YOLO on a video — save a cropped JPEG per detection
python scripts/extract_yolo_crops.py path/to/video.mp4

# Sample faster, save annotated full frames too
python scripts/extract_yolo_crops.py video.mp4 --fps 2 --save-frames

# Run YOLO on every frame (disable motion gate)
python scripts/extract_yolo_crops.py video.mp4 --no-motion

# Output goes to yolo-crops/bird/ and yolo-crops/cat/ by default
```

Use this to audit YOLO's accuracy and build a test image set.

### Compare classifier backends on still images

```bash
# Drop bird crop JPEGs into test-photos/ then:
python scripts/test_photos.py                   # compare all three backends
python scripts/test_photos.py --backend bioclip # single backend
python scripts/test_photos.py --input-dir yolo-crops/bird/
```

Produces annotated images + `summary.csv` + `inference_log.json` in `test-photos-results/`.

### Test the full pipeline on a video

```bash
# Run motion → YOLO → classifier on a video file
python scripts/test_pipeline.py path/to/video.mp4

# With a specific backend
python scripts/test_pipeline.py video.mp4 --backend bioclip

# On a folder of pre-cropped images (skips YOLO)
python scripts/test_pipeline.py data/corrections/

# Don't write to the database
python scripts/test_pipeline.py video.mp4 --no-log
```

### Typical test workflow

```bash
# 1. Extract and review YOLO detections
python scripts/extract_yolo_crops.py feeder_clip.mp4 --save-frames

# 2. Check yolo-crops/ — move any YOLO false positives to a separate folder

# 3. Compare classifiers on the clean bird crops
python scripts/test_photos.py --input-dir yolo-crops/bird/

# 4. Pick the best backend and set it in .env
echo "CLASSIFIER_BACKEND=bioclip" >> .env
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
    ├── "bird"     → SpeciesClassifier (tfhub | bioclip | nabirds)
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
| `VIDEO_SOURCE` | — | RTSP URL or video file path |
| `CAPTURE_FPS` | `5` | Frames per second to sample |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model (`yolov8n/s/m.pt`) |
| `YOLO_CONFIDENCE` | `0.25` | YOLO detection threshold |
| `PREDATOR_MIN_AREA` | `15000` | Below this px², relabel predator → bird |
| `PREDATOR_MIN_CONFIDENCE` | `0.70` | Min conf to fire a predator alert |
| `CLASSIFIER_BACKEND` | `tfhub` | `tfhub` \| `bioclip` \| `nabirds` |
| `BIOCLIP_MODEL` | `hf-hub:imageomics/bioclip` | BioCLIP model string |
| `NABIRDS_MODEL` | `chriamue/bird-species-classifier` | HuggingFace model ID |
| `MIN_CONFIDENCE_LOG` | `0.3` | Minimum confidence to log a detection |
| `MIN_CONFIDENCE_NOTIFY` | `0.7` | Minimum confidence to publish MQTT |
| `DETECTION_COOLDOWN_SEC` | `10` | Seconds before re-logging the same region |

See `bird-detector/config.py` for the full list.

---

## Key Commands

```bash
# Live logs
docker compose -f docker-compose.opencv.yml logs -f bird-detector

# Today's detections
sqlite3 data/detections.db \
  "SELECT species_common, COUNT(*) as visits FROM detections \
   WHERE date(timestamp) = date('now') \
   GROUP BY species_common ORDER BY visits DESC;"

# Watch MQTT traffic
docker compose -f docker-compose.opencv.yml exec mosquitto \
  mosquitto_sub -u birdfeeder -P <password> -t "birdfeeder/#" -v

# Export last 7 days to CSV
python3 scripts/export_csv.py --days 7 --output weekly.csv

# Restart the detector
docker compose -f docker-compose.opencv.yml restart bird-detector
```

---

## Data

```
data/
├── snapshots/        # Archived crops: YYYY-MM-DD/timestamp_species_conf.jpg
├── corrections/      # Low-confidence detections for manual review
├── detections.csv    # Append-only flat log
└── detections.db     # SQLite (indexed, queryable)
```

CSV columns: `timestamp, date, time, species_common, species_scientific, confidence, source, classifier, snapshot_path, reviewed, corrected_species`

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
| Species classification | TFHub AIY Birds V1 / BioCLIP / HuggingFace |
| Video capture | OpenCV |
| MQTT broker | Eclipse Mosquitto |
| Logging | CSV + SQLite |
| Language | Python 3.11+ |
| Containers | Docker Compose |
