# Backyard Bird Feeder Detection System

A local, privacy-first bird and wildlife detection/identification system for a backyard bird feeder. The system captures video from an IP camera (Tapo C200), detects visitors (birds, squirrels, raccoons), identifies species, logs all activity to CSV/SQLite, and integrates with Home Assistant for dashboards and automations. Everything runs on a single Ubuntu laptop on the home network.

```
Tapo C200 ──RTSP──▶ Frigate NVR ──MQTT──▶ Bird Classifier ──▶ CSV / SQLite Logger
                        │
                        │ MQTT
                        ▼
                   Home Assistant (future)
```

---

## Quick Start

### Prerequisites

- **Camera:** TP-Link Tapo C200 (1080p, RTSP-capable) with a camera account created
- **Compute:** Ubuntu 24.04 laptop on the same Wi-Fi network
- **Software:** Docker, Docker Compose, ffmpeg
  ```bash
  sudo apt install docker.io docker-compose-v2 ffmpeg
  sudo usermod -aG docker $USER  # run docker without sudo
  ```

### 1. Clone and Configure

```bash
git clone <your-repo-url> backyardBirdwatch
cd backyardBirdwatch

# Copy the environment template
cp .env.example .env

# Edit with your camera credentials and location
nano .env
```

Required fields in `.env`:

| Variable | Value |
|----------|-------|
| `CAMERA_IP` | Static IP of Tapo C200 on your LAN |
| `RTSP_USER` | Camera account username (from Tapo app) |
| `RTSP_PASSWORD` | Camera account password |
| `LATITUDE` / `LONGITUDE` | Your feeder's GPS coordinates |
| `MQTT_USER` | Username (e.g., `frigate`) |
| `MQTT_PASSWORD` | Strong password |

See [Configuration](docs/configuration.md) for detailed setup of camera credentials.

### 2. Test RTSP Connectivity

Verify both camera streams work before deploying:

```bash
./scripts/test_rtsp.sh
```

You should see stream details for both `stream1` and `stream2`. If it fails, see [Troubleshooting](docs/troubleshooting.md#rtsp-connection-fails).

### 3. Initialize Mosquitto

Mosquitto (the MQTT broker) requires a password file to exist **before** startup:

```bash
# Create empty password file
touch mosquitto/config/password_file
chmod 644 mosquitto/config/password_file

# Start the stack
docker compose up -d

# Add the MQTT user (will prompt for password)
docker exec -it backyardbirdwatch-mosquitto-1 \
  mosquitto_passwd /mosquitto/config/password_file frigate

# Fix ownership and permissions
docker exec -u root backyardbirdwatch-mosquitto-1 \
  chown mosquitto:mosquitto /mosquitto/config/password_file
docker exec -u root backyardbirdwatch-mosquitto-1 \
  chmod 0700 /mosquitto/config/password_file

# Restart to apply
docker compose restart mosquitto
```

Verify health:
```bash
docker compose ps
# All services should show "healthy"
```

### 4. Verify End-to-End

1. **Frigate UI** — Open http://localhost:5000
   - You should see a live feed from `bird_feeder`
   - Point the camera at your feeder and wait for a bird
   - Frigate should draw a bounding box labeled `bird`

2. **Classifier logs** — Watch detections as they happen
   ```bash
   docker compose logs -f bird-classifier
   ```
   You should see output like:
   ```
   INFO: Processing bird detection event: 1707732923.456-abcdef
   INFO: iNaturalist result: Northern Cardinal (0.91)
   INFO: Logged detection: Northern Cardinal (0.91) via inaturalist
   ```

3. **Data logged** — Check the detection database
   ```bash
   sqlite3 data/detections.db \
     "SELECT timestamp, species_common, confidence FROM detections LIMIT 5;"
   ```

---

## Testing with Sample Videos

You don't need to wait for birds to show up at the feeder to develop and tune the system. Two modes are available depending on what you want to test.

### Mode 1 — Classifier only (no Docker required)

`scripts/test_pipeline.py` extracts frames from a local video file and runs them through the same two-stage classification pipeline (local TFHub model → iNaturalist API) that the live service uses. Results are printed to the terminal and optionally written to `data/detections.csv` / `data/detections.db`.

```bash
# Basic: 1 frame/sec, both classifiers
python scripts/test_pipeline.py path/to/video.mp4

# Multiple videos, 2 frames/sec
python scripts/test_pipeline.py clip1.mp4 clip2.mp4 --fps 2

# Fast iteration — local TFHub model only, no API calls
python scripts/test_pipeline.py video.mp4 --skip-inat

# Lower the confidence bar to see more candidates
python scripts/test_pipeline.py video.mp4 --min-confidence 0.2

# Test against existing Frigate snapshot crops
python scripts/test_pipeline.py frigate/storage/clips/

# Don't write to the database while testing
python scripts/test_pipeline.py video.mp4 --no-log
```

Use this mode when tuning `MIN_CONFIDENCE_LOG`, `MIN_CONFIDENCE_NOTIFY`, or the iNaturalist/TFHub settings.

### Mode 2 — Full Frigate pipeline (Docker)

For tuning Frigate settings (motion threshold, `min_score`, `min_area`, detection zones), you need Frigate to actually process the footage. A [mediamtx](https://github.com/bluenviron/mediamtx) container loops your video file as a real RTSP stream; Frigate reads it exactly as it would the live camera.

```bash
# 1. Place your sample video here (gitignored):
cp /path/to/your/footage.mp4 test-videos/sample.mp4

# 2. Start the full stack in test mode
docker compose -f docker-compose.yml -f docker-compose.test.yml up -d

# 3. Open Frigate at http://localhost:5000 — events accumulate as the video loops

# 4. Verify the RTSP feed directly in VLC (optional):
#    rtsp://localhost:8554/sample

# 5. Tear down when done
docker compose -f docker-compose.yml -f docker-compose.test.yml down
```

The overlay (`docker-compose.test.yml`) adds the `restreamer` service and swaps Frigate's config to point at it — no changes to `docker-compose.yml` or `frigate/config.yml` needed.

---

## What Happens When a Bird Visits

1. **Frigate** detects the bird in the stream (motion + object detection)
2. **Classifier** processes the detection on the final frame when the bird leaves
3. **Local model** (TFHub) runs first for fast classification (~50ms)
   - If confidence ≥ 0.85 → use that result
   - If confidence < 0.85 → query iNaturalist API
4. **Result logged** to `data/detections.csv` and `data/detections.db`
5. **Snapshot archived** to `data/snapshots/YYYY-MM-DD/`
6. **MQTT published** to `birdfeeder/detection` and species-specific topics

Low-confidence detections (< 0.7) are saved to `data/corrections/` for your review.

---

## Documentation

| Document | Contents |
|----------|----------|
| [Getting Started](docs/getting-started.md) | Step-by-step setup and first run |
| [Architecture](docs/architecture.md) | System design, services, pipeline flow |
| [Configuration](docs/configuration.md) | Every environment variable explained, tuning tips |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |

---

## Key Commands

```bash
# View live logs
docker compose logs -f

# Check a single service
docker compose logs -f frigate
docker compose logs -f bird-classifier
docker compose logs -f mosquitto

# View today's detections
sqlite3 data/detections.db \
  "SELECT species_common, COUNT(*) as count FROM detections \
   WHERE date(timestamp) = date('now') \
   GROUP BY species_common ORDER BY count DESC;"

# Export last 7 days to CSV
python3 scripts/export_csv.py --days 7 --output weekly.csv

# Watch MQTT traffic (change password)
docker compose exec mosquitto \
  mosquitto_sub -u frigate -P <your-password> -t "birdfeeder/#" -v

# Restart a service
docker compose restart bird-classifier
```

---

## Data Capture & Logging

### How Bird Data is Captured

**Stage 1: Object Detection (Frigate)**
- Watches the RTSP stream from the Tapo C200 continuously at 5 fps (low-resolution detect stream)
- When motion is detected, runs the SSD MobileNet object detector
- Detects `bird` and `cat` classes from the COCO model
- Saves high-resolution (1080p) cropped snapshots of detections
- Publishes events to MQTT topic `frigate/events`

**Stage 2: Species Classification (Bird Classifier)**
- Subscribes to `frigate/events` and waits for detection events to complete (end event)
- Downloads the cropped snapshot from Frigate's API
- Runs **local TFHub bird model** (~900 species, ~50ms, 100% offline)
  - If confidence ≥ 0.85 → use result immediately
  - If confidence < 0.85 → proceed to Stage 3
- Runs **iNaturalist CV API** (rate-limited, 109K+ taxa, location-aware)
  - Geo-filters results using your feeder's coordinates (drastically improves accuracy)
  - Returns top 5 candidates with confidence scores
  - Captures both common and scientific names

**Stage 3: Logging & Publishing**
- If confidence ≥ `MIN_CONFIDENCE_LOG` (0.3):
  - Archives the snapshot to `data/snapshots/YYYY-MM-DD/`
  - Writes record to `data/detections.csv` (append-only)
  - Writes record to `data/detections.db` (SQLite, indexed)
  - Publishes to MQTT topics:
    - `birdfeeder/detection` (all detections)
    - `birdfeeder/detection/{species-slug}` (per-species, e.g., `northern-cardinal`)
    - `birdfeeder/new_species` (first time a species is seen)
- If confidence < `MIN_CONFIDENCE_NOTIFY` (0.7):
  - Also saves to `data/corrections/` for manual review

### CSV Log Format

Every detection produces one row in `data/detections.csv`:

```csv
timestamp,date,time,species_common,species_scientific,confidence,source,classifier,duration_sec,count,weather,temperature_f,snapshot_path,audio_path,frigate_event_id,reviewed,corrected_species
2026-02-12T19:15:23-05:00,2026-02-12,19:15:23,Northern Cardinal,Cardinalis cardinalis,0.91,visual,inaturalist,NULL,1,NULL,NULL,data/snapshots/2026-02-12/2026-02-12T19-15-23_northern-cardinal_0.91.jpg,NULL,1707732923.456-abcdef,0,NULL
```

| Field | Meaning |
|-------|---------|
| `timestamp` | ISO 8601 timestamp with timezone |
| `date` / `time` | Local date and time (for easy filtering) |
| `species_common` | e.g., "Northern Cardinal" |
| `species_scientific` | e.g., "Cardinalis cardinalis" |
| `confidence` | Model confidence 0.0–1.0 |
| `source` | `visual`, `audio`, or `both` (audio TBD) |
| `classifier` | `inaturalist`, `tfhub_birds`, or `birdnet` |
| `duration_sec` | How long bird was in frame (future) |
| `count` | Number of individuals detected (future) |
| `weather` / `temperature_f` | Optional API data (future) |
| `snapshot_path` | Path to archived image |
| `audio_path` | Path to audio clip (future) |
| `frigate_event_id` | Frigate's internal event ID |
| `reviewed` | 0 or 1 — has human verified? |
| `corrected_species` | If wrong, what was it actually? |

### SQLite Database

The same data is also written to `data/detections.db` with indexes for fast querying:

```bash
# Today's species count
sqlite3 data/detections.db \
  "SELECT species_common, COUNT(*) as visits FROM detections \
   WHERE date(timestamp) = date('now') \
   GROUP BY species_common ORDER BY visits DESC;"

# Species first seen this month
sqlite3 data/detections.db \
  "SELECT species_common, MIN(timestamp) as first_seen FROM detections \
   WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now') \
   GROUP BY species_common;"

# Peak activity hours
sqlite3 data/detections.db \
  "SELECT strftime('%H', timestamp) as hour, COUNT(*) as detections FROM detections \
   GROUP BY hour ORDER BY detections DESC;"
```

### Snapshot Archive

```
data/snapshots/
├── 2026-02-12/
│   ├── 2026-02-12T08-15-23_northern-cardinal_0.91.jpg
│   ├── 2026-02-12T09-47-11_blue-jay_0.88.jpg
│   └── 2026-02-12T14-22-55_house-finch_0.76.jpg
└── 2026-02-13/
    └── ...
```

Filenames follow the pattern: `{ISO_timestamp}_{species-slug}_{confidence}.jpg`

Each snapshot is a high-resolution (1080p) cropped image of just the bird, making it easy to review and correct misidentifications.

### Corrections Folder

```
data/corrections/
├── 1707732923.456-abcdef.jpg  # Unknown or low-confidence (< 0.7)
├── 1707733145.789-ghijkl.jpg
└── ...
```

Low-confidence detections are automatically saved here for manual review. You can:
1. Open the image
2. Identify the species correctly
3. Update `corrected_species` column in the database
4. Use these corrected images to improve future classification

### Storage Requirements

```
frigate/storage/     ~2–3 GB    4 hours continuous recording + 30 days event clips
data/snapshots/      ~100 MB    ~1000 snapshots per 30 days (depends on activity)
data/detections.csv  ~1 MB      ~50,000 rows per year
data/detections.db   ~2 MB      Same data, indexed for queries
```

Continuous recording auto-deletes after 4 hours. Event clips stay for 30 days. You can export old data to CSV and delete the database if disk space is tight.

### Export & Analysis

```bash
# Export last 7 days to CSV
python3 scripts/export_csv.py --days 7 --output weekly.csv

# Export all-time
python3 scripts/export_csv.py --output all_detections.csv

# Query with pandas (Python)
import pandas as pd
df = pd.read_csv('data/detections.csv')
print(df[df['date'] == '2026-02-12'].groupby('species_common').size())
```

---

## Storage

```
data/
├── snapshots/          # Organized by date: YYYY-MM-DD/timestamp_species_conf.jpg
├── corrections/        # Low-confidence images for manual review
├── detections.csv      # Flat-file detection log (append-only)
└── detections.db       # SQLite database (indexed, queryable)

frigate/storage/        # 4 hours of rolling continuous recording + 30-day event clips
```

Continuous recording uses ~2 GB per 4 hours at 1080p. Old footage auto-deletes.

---

## Deployment to Another Laptop

To move this system to a different laptop:

```bash
# 1. Clone the repo
git clone <your-repo> backyardBirdwatch
cd backyardBirdwatch

# 2. Configure environment
cp .env.example .env
nano .env

# 3. Create password file
touch mosquitto/config/password_file
chmod 644 mosquitto/config/password_file

# 4. Start services
docker compose up -d

# 5. Add MQTT user
docker exec -it backyardbirdwatch-mosquitto-1 \
  mosquitto_passwd /mosquitto/config/password_file frigate

# 6. Fix permissions and restart
docker exec -u root backyardbirdwatch-mosquitto-1 \
  chown mosquitto:mosquitto /mosquitto/config/password_file
docker exec -u root backyardbirdwatch-mosquitto-1 \
  chmod 0700 /mosquitto/config/password_file
docker compose restart mosquitto
```

The `.env` file and `mosquitto/config/password_file` are git-ignored, so they won't be committed. You configure them fresh on each machine.

---

## Project Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Infrastructure | ✅ Complete | Docker Compose, Mosquitto, Frigate |
| 2 — Classification | ✅ Complete | Two-stage bird-classifier service |
| 3 — Home Assistant | 🔄 Planned | Lovelace dashboard, automations, alerts |
| 4 — Audio (BirdNET) | 🔄 Planned | Audio-based species ID from recordings |
| 5 — Analysis | 🔄 Planned | Seasonal trends, peak hours, diversity reports |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Container orchestration | Docker Compose | Service management |
| NVR / object detection | Frigate NVR | Motion detection, bird detection, recording |
| MQTT broker | Eclipse Mosquitto | Inter-service messaging |
| Species classification | iNaturalist API + TFHub | Visual ID of birds/wildlife |
| Audio classification | BirdNET (future) | Species ID by sound |
| Home automation | Home Assistant (future) | Dashboards and automations |
| Logging | CSV + SQLite | Detection history and analysis |
| Language | Python 3.11+ | Classifier service |

---

## License

[Add your license here]

---

## Help

For detailed setup, see [Getting Started](docs/getting-started.md).

For issues, see [Troubleshooting](docs/troubleshooting.md).

For architecture questions, see [Architecture](docs/architecture.md).
