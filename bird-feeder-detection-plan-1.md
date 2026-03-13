# Backyard Bird Feeder Detection System — Architecture & Plan

## 1. System Overview

A local, privacy-first bird (and critter) detection and identification system that runs entirely on your home network. The camera watches your feeder 24/7, the laptop processes the video stream, identifies species, logs everything to CSV, and eventually feeds events into Home Assistant for dashboards and automations.

```
┌──────────────┐    RTSP     ┌──────────────────────────────────────────────┐
│  Tapo C200   │────────────▶│          Ubuntu 24.04 Laptop                 │
│  (Wi-Fi)     │  stream     │                                              │
└──────────────┘             │  ┌────────────┐   ┌───────────────────────┐  │
                             │  │  Frigate    │──▶│  Motion-triggered     │  │
                             │  │  NVR        │   │  frame capture        │  │
                             │  └─────┬──────┘   └──────────┬────────────┘  │
                             │        │ MQTT                │               │
                             │        ▼                     ▼               │
                             │  ┌────────────┐   ┌───────────────────────┐  │
                             │  │  Home       │   │  Bird Detection &     │  │
                             │  │  Assistant  │   │  Classification       │  │
                             │  │  (future)   │   │  Pipeline             │  │
                             │  └────────────┘   └──────────┬────────────┘  │
                             │                              │               │
                             │                              ▼               │
                             │                   ┌───────────────────────┐  │
                             │                   │  CSV / SQLite Logger  │  │
                             │                   │  + Snapshot Archive   │  │
                             │                   └───────────────────────┘  │
                             └──────────────────────────────────────────────┘
```

---

## 2. Hardware & Network Setup

### 2.1 Camera: Tapo C200

The Tapo C200 supports RTSP natively, which is what makes this whole pipeline possible.

**RTSP Setup Steps:**

1. Open the Tapo app → Camera Settings → Advanced Settings → Camera Account
2. Create a username and password (6–32 characters, avoid special characters)
3. Your RTSP URLs will be:
   - **High quality (1080p):** `rtsp://<username>:<password>@<camera-ip>:554/stream1`
   - **Low quality (360p):** `rtsp://<username>:<password>@<camera-ip>:554/stream2`
4. Find the camera IP via your router's DHCP list or the Tapo app

**Recommendations:**

- Use `stream2` (low quality) for detection to reduce CPU load — you only need enough resolution to distinguish species, not pixel-perfect video
- Use `stream1` for snapshot capture when a detection occurs (high-res photo for the archive and for species classification)
- Assign a static IP or DHCP reservation to the camera
- Position the camera 1–3 meters from the feeder for best results; the C200's 1080p at that range gives you plenty of detail for bird ID
- The C200 has pan-tilt capability — lock it to a fixed position pointed at the feeder

### 2.2 Laptop: Ubuntu 24.04

No GPU required for the initial setup (CPU inference is fine for a single feeder camera), but if you have an NVIDIA GPU, it will significantly speed up YOLO inference.

**Key software to install:**

- Docker & Docker Compose (for Frigate, Mosquitto, Home Assistant)
- Python 3.11+ (for the custom classification pipeline)
- ffmpeg (for RTSP stream handling and audio extraction)

---

## 3. Detection & Classification Pipeline

This is the core of the system. The architecture uses a **two-stage approach**: detect first (is there *something* at the feeder?), then classify (what species is it?).

### 3.1 Stage 1 — Motion Detection & Object Detection

**Option A: Frigate NVR (Recommended for Home Assistant path)**

Frigate is purpose-built for this. It watches your RTSP stream, uses motion detection to know *when* to run inference, and runs an object detector only on the regions with motion. This keeps CPU usage very low when nothing is happening.

- Frigate's default model (SSD MobileNet) detects COCO classes including `bird` out of the box
- It communicates via MQTT, which is exactly what Home Assistant needs
- It saves snapshots and clips of detection events automatically
- It handles all the stream management, recording, and retention for you

```yaml
# frigate/config.yml (minimal example)
mqtt:
  host: mosquitto  # or your MQTT broker IP
  user: frigate
  password: "{FRIGATE_MQTT_PASSWORD}"

cameras:
  bird_feeder:
    ffmpeg:
      inputs:
        - path: rtsp://<user>:<pass>@<camera-ip>:554/stream2
          roles:
            - detect
        - path: rtsp://<user>:<pass>@<camera-ip>:554/stream1
          roles:
            - record
            - snapshots
    detect:
      width: 640
      height: 480
      fps: 5           # 5 fps is plenty for bird detection
    objects:
      track:
        - bird
        - cat           # predator alert!
      filters:
        bird:
          min_score: 0.4
          min_area: 1000
    snapshots:
      enabled: true
      crop: true        # crops to the detected object — great for classification
    record:
      enabled: true
      retain:
        days: 7
      events:
        retain:
          default: 30
```

**Limitation:** Frigate's default model only knows `bird` — it does not distinguish species. That's where Stage 2 comes in.

**Option B: Custom YOLO Pipeline (if you want more control)**

If you'd rather skip Frigate and build your own, use a pre-trained YOLOv8 model for detection. The standard COCO-trained YOLOv8 will detect `bird` as a class. You can also find community models on Roboflow or Ultralytics Hub that have been trained specifically on bird species.

```python
# Minimal YOLO detection loop
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # nano model, fast on CPU

cap = cv2.VideoCapture("rtsp://<user>:<pass>@<camera-ip>:554/stream2")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, conf=0.4)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        if label == "bird":
            # Crop the detection region
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            # Send crop to Stage 2 classifier...
```

### 3.2 Stage 2 — Species Classification

Once you have a cropped image of a bird, you pass it to a species classifier. Here are the best options ranked by practicality:

#### Option 1: iNaturalist Vision Model (Best overall)

The iNaturalist CV model currently covers 109,000+ taxa and is excellent at North American birds. You can access it via their API:

```python
import requests

def classify_bird_inat(image_path, lat=None, lng=None):
    """Classify a bird image using the iNaturalist API."""
    url = "https://api.inaturalist.org/v1/computervisions/score_image"
    files = {"image": open(image_path, "rb")}
    params = {}
    if lat and lng:
        params["lat"] = lat
        params["lng"] = lng
    response = requests.post(url, files=files, params=params)
    results = response.json().get("results", [])
    return [
        {
            "species": r["taxon"]["preferred_common_name"],
            "scientific": r["taxon"]["name"],
            "score": r["combined_score"],
        }
        for r in results[:5]
    ]
```

**Pros:** Covers birds, mammals (raccoons, squirrels), and basically everything. Location-aware filtering. No local model to manage.
**Cons:** Requires internet; rate-limited (unclear public limits — be respectful, batch calls). Not suitable for real-time per-frame classification, but perfect for event-triggered snapshots.

#### Option 2: Google's TFHub Bird Classifier (Fully local)

Google provides a MobileNet-based bird classifier trained on iNaturalist bird data. This runs entirely locally and is very lightweight.

```bash
pip install tensorflow tensorflow-hub pillow --break-system-packages
```

```python
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model (downloads once, ~16MB)
model = hub.load("https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1")

def classify_bird_local(image_path):
    img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    logits = model(img)
    top_indices = tf.argsort(logits[0], direction="DESCENDING")[:5]
    return top_indices.numpy()
```

**Pros:** Fast, local, no internet needed, runs on CPU.
**Cons:** Limited to ~900 bird species. No mammals (won't help with raccoon/squirrel). Older model; the label mapping can be tricky.

#### Option 3: Hybrid Approach (Recommended)

Use a **local lightweight classifier** for fast first-pass identification and fall back to the **iNaturalist API** for uncertain results or periodic verification.

```
Detection event
    │
    ▼
Local classifier (TFHub birds / custom fine-tuned)
    │
    ├── confidence > 0.85 → Log species directly
    │
    └── confidence < 0.85 → Queue for iNaturalist API call
                                │
                                └── Log with iNat result + flag for review
```

### 3.3 Audio Bonus: BirdNET

If your camera or a nearby microphone picks up audio, BirdNET can identify birds by their songs and calls. This is incredibly powerful for species that are vocal but hard to photograph (wrens, for example).

```bash
pip install birdnet --break-system-packages
```

```python
import birdnet

model = birdnet.load("acoustic", "2.4", "tf")

# Predict species from audio
predictions = model.predict(
    "feeder_audio_clip.wav",
    custom_species_list="species_list.txt",  # optional filter
)
predictions.to_csv("audio_detections.csv")
```

BirdNET can also filter by location and week of year:

```python
# Get species expected at your location this week
geo_model = birdnet.load("geo", "2.4", "tf")
expected = geo_model.predict(lat=40.0, lng=-74.0, week=12)
```

**Integration idea:** Run BirdNET on audio segments extracted from Frigate recordings to get a second modality of species ID. Cross-referencing visual + audio gives you much higher confidence.

---

## 4. Logging & Data Pipeline

### 4.1 CSV Log Structure

Every detection event gets a row:

```csv
timestamp,date,time,species_common,species_scientific,confidence,source,
duration_sec,count,weather,temperature_f,snapshot_path,audio_path,notes,
reviewed,corrected_species
```

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 UTC timestamp |
| `date` / `time` | Local date and time (for easy filtering) |
| `species_common` | e.g., "Northern Cardinal" |
| `species_scientific` | e.g., "Cardinalis cardinalis" |
| `confidence` | Model confidence score (0.0–1.0) |
| `source` | `visual`, `audio`, or `both` |
| `duration_sec` | How long the bird was tracked in frame |
| `count` | Number of individuals detected simultaneously |
| `weather` | Optional: pulled from a weather API |
| `temperature_f` | Optional: outdoor temp at time of detection |
| `snapshot_path` | Path to the cropped detection image |
| `audio_path` | Path to audio clip if available |
| `notes` | Auto-generated notes (e.g., "feeding", "perched") |
| `reviewed` | Boolean: has a human verified this? |
| `corrected_species` | If reviewed and wrong, what was it actually? |

### 4.2 SQLite for Querying

For long-term analysis, mirror the CSV into SQLite:

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    species_common TEXT,
    species_scientific TEXT,
    confidence REAL,
    source TEXT,
    duration_sec REAL,
    count INTEGER DEFAULT 1,
    temperature_f REAL,
    snapshot_path TEXT,
    reviewed INTEGER DEFAULT 0,
    corrected_species TEXT
);

-- Example queries for seasonal analysis:
-- Species diversity by month
SELECT strftime('%Y-%m', timestamp) AS month,
       COUNT(DISTINCT species_common) AS species_count
FROM detections GROUP BY month;

-- First/last sighting of each species per year
SELECT species_common,
       MIN(timestamp) AS first_seen,
       MAX(timestamp) AS last_seen
FROM detections
WHERE strftime('%Y', timestamp) = '2026'
GROUP BY species_common;

-- Peak activity hours
SELECT strftime('%H', timestamp) AS hour,
       COUNT(*) AS detections
FROM detections GROUP BY hour ORDER BY detections DESC;
```

### 4.3 Snapshot Archive

```
data/
├── snapshots/
│   ├── 2026-02-12/
│   │   ├── 2026-02-12T08-15-23_cardinal_0.92.jpg
│   │   ├── 2026-02-12T08-17-01_bluejay_0.88.jpg
│   │   └── 2026-02-12T09-02-44_squirrel_0.95.jpg
│   └── 2026-02-13/
│       └── ...
├── audio_clips/
│   └── ...
├── detections.csv
├── detections.db
└── corrections/          # images flagged for review
    └── ...
```

---

## 5. Home Assistant Integration

### 5.1 Architecture

```
┌──────────┐         ┌───────────┐         ┌──────────────┐
│ Frigate  │──MQTT──▶│ Mosquitto │──MQTT──▶│ Home         │
│ NVR      │         │ Broker    │         │ Assistant    │
└──────────┘         └───────────┘         └──────┬───────┘
                                                  │
                              ┌────────────────────┤
                              │                    │
                              ▼                    ▼
                     ┌─────────────┐      ┌──────────────┐
                     │ Lovelace    │      │ Automations  │
                     │ Dashboard   │      │ & Alerts     │
                     └─────────────┘      └──────────────┘
```

### 5.2 Docker Compose Stack

```yaml
# docker-compose.yml
version: "3.9"

services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"
    volumes:
      - ./mosquitto/config:/mosquitto/config
      - ./mosquitto/data:/mosquitto/data
    restart: unless-stopped

  frigate:
    image: ghcr.io/blakeblackshear/frigate:stable
    privileged: true
    ports:
      - "5000:5000"   # Frigate UI
      - "8554:8554"   # RTSP re-stream
      - "8555:8555"   # WebRTC
    volumes:
      - ./frigate/config.yml:/config/config.yml
      - ./frigate/storage:/media/frigate
      - /etc/localtime:/etc/localtime:ro
    environment:
      - FRIGATE_RTSP_PASSWORD=${RTSP_PASSWORD}
    restart: unless-stopped
    depends_on:
      - mosquitto

  homeassistant:
    image: ghcr.io/home-assistant/home-assistant:stable
    ports:
      - "8123:8123"
    volumes:
      - ./homeassistant/config:/config
      - /etc/localtime:/etc/localtime:ro
    restart: unless-stopped
    depends_on:
      - mosquitto

  bird-classifier:
    build: ./bird-classifier
    volumes:
      - ./data:/data
      - ./frigate/storage:/media/frigate:ro
    environment:
      - MQTT_HOST=mosquitto
      - MQTT_PORT=1883
    restart: unless-stopped
    depends_on:
      - mosquitto
      - frigate
```

### 5.3 Custom MQTT Topics

The bird-classifier service listens for Frigate bird detection events and publishes enriched results:

```
# Frigate publishes (built-in):
frigate/events                          # all detection events

# Bird classifier publishes (custom):
birdfeeder/detection                    # species-level detection
birdfeeder/detection/cardinal           # per-species topic
birdfeeder/daily_summary                # end-of-day summary
birdfeeder/new_species                  # first-of-season alerts
```

**Example MQTT payload:**

```json
{
  "timestamp": "2026-02-12T08:15:23-05:00",
  "species": "Northern Cardinal",
  "scientific_name": "Cardinalis cardinalis",
  "confidence": 0.92,
  "source": "visual",
  "snapshot_url": "/media/frigate/snapshots/bird_feeder/2026-02-12T08-15-23.jpg",
  "count": 1
}
```

### 5.4 Home Assistant Automations

```yaml
# Example: Send notification for rare/first-of-season species
automation:
  - alias: "New Bird Species Alert"
    trigger:
      platform: mqtt
      topic: "birdfeeder/new_species"
    action:
      - service: notify.mobile_app
        data:
          title: "New bird species spotted!"
          message: >
            {{ trigger.payload_json.species }} detected at the feeder
            (confidence: {{ trigger.payload_json.confidence }})
          data:
            image: "{{ trigger.payload_json.snapshot_url }}"

  - alias: "Predator Alert"
    trigger:
      platform: mqtt
      topic: "frigate/events"
    condition:
      - condition: template
        value_template: "{{ trigger.payload_json.after.label == 'cat' }}"
    action:
      - service: notify.mobile_app
        data:
          title: "Cat near the feeder!"
          message: "A cat was detected near the bird feeder."
```

---

## 6. Phased Implementation Plan

### Phase 1: Camera + Basic Detection (Week 1)

**Goal:** Get the RTSP stream working and detect "bird" as a class.

1. Set up camera account in Tapo app, verify RTSP works with VLC:
   `vlc rtsp://<user>:<pass>@<camera-ip>:554/stream2`
2. Install Docker on Ubuntu laptop
3. Deploy Mosquitto + Frigate via Docker Compose
4. Configure Frigate with the Tapo C200 RTSP streams
5. Verify Frigate detects `bird` class in its web UI (port 5000)
6. Set up cropped snapshot saving

**Deliverable:** Frigate running, detecting birds, saving snapshots.

### Phase 2: Species Classification (Week 2–3)

**Goal:** Identify bird species from Frigate snapshots.

1. Build the `bird-classifier` Python service
2. Subscribe to Frigate MQTT events for `bird` detections
3. Grab the cropped snapshot → run through iNaturalist API
4. Publish species-level MQTT messages
5. Log every detection to CSV and SQLite
6. Set up the snapshot archive directory structure

**Deliverable:** Per-species identification with confidence scores, logged to CSV/SQLite.

### Phase 3: Home Assistant Integration (Week 3–4)

**Goal:** Dashboards and automations.

1. Deploy Home Assistant via Docker Compose
2. Install Frigate integration via HACS
3. Configure MQTT integration pointing to Mosquitto
4. Build a Lovelace dashboard with:
   - Live camera feed
   - Today's species list
   - Detection count chart
   - Recent snapshots gallery
5. Set up automations (new species alerts, daily summary, predator alerts)

**Deliverable:** Fully integrated HA dashboard with live detection feed.

### Phase 4: Audio Integration (Week 4–5)

**Goal:** Add BirdNET audio identification as a second modality.

1. If the Tapo C200 audio quality is sufficient, extract audio segments from Frigate recordings using ffmpeg
2. Otherwise, add a USB microphone near the feeder
3. Run BirdNET on audio clips around detection timestamps
4. Cross-reference visual and audio IDs for higher confidence
5. Log audio detections alongside visual ones

**Deliverable:** Dual-modality (visual + audio) species identification.

### Phase 5: Refinement & Seasonal Analysis (Ongoing)

**Goal:** Improve accuracy and build behavioral insights.

1. Review the `corrections/` folder weekly — fix misidentifications
2. Track accuracy metrics over time
3. Once you have 50+ corrected images per common species, consider fine-tuning a local classifier
4. Build seasonal analysis queries:
   - Migration arrival/departure dates
   - Peak feeding hours by species
   - Weather correlation analysis
   - Species diversity trends
5. Optional: Set up a Grafana dashboard for rich visualizations

---

## 7. Species Reference for Your Area

Based on the species you mentioned, here's a quick reference with expected detection characteristics:

| Species | Visual ID Difficulty | Audio ID | Season |
|---------|---------------------|----------|--------|
| Northern Cardinal | Easy (bright red male) | Excellent | Year-round |
| American Robin | Easy | Good | Spring–Fall |
| House/Gold Finch | Moderate (similar shapes) | Good | Year-round |
| Dark-eyed Junco | Moderate | Fair | Winter |
| Tufted Titmouse | Moderate | Excellent | Year-round |
| Blue Jay | Easy (large, blue) | Excellent | Year-round |
| Sparrows (various) | Hard (many similar spp.) | Good with BirdNET | Year-round |
| Brown-headed Cowbird | Moderate | Fair | Spring–Fall |
| European Starling | Easy (iridescent) | Good | Year-round |
| Woodpeckers | Moderate | Good | Year-round |
| Black-capped Chickadee | Easy (distinctive mask) | Excellent | Year-round |
| Carolina Wren | Moderate (small, fast) | Excellent | Year-round |
| Raccoon | Easy (large, nocturnal) | N/A | Year-round |
| Squirrel (gray/fox) | Easy (large, diurnal) | N/A | Year-round |

**Tip:** Sparrows are the hardest group for visual classification. BirdNET audio ID is often more reliable than visual for distinguishing Song Sparrow, White-throated Sparrow, House Sparrow, etc.

---

## 8. Key Technology Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Camera | Tapo C200 (RTSP) | Video capture |
| Stream management | Frigate NVR | Motion detection, object detection, recording |
| Object detection | Frigate's SSD MobileNet or YOLOv8 | Detect "bird" class |
| Species classification | iNaturalist API + TFHub birds model | Identify species |
| Audio classification | BirdNET (Python `birdnet` package) | Identify species by sound |
| Message broker | Mosquitto (MQTT) | Inter-service communication |
| Home automation | Home Assistant | Dashboards, alerts, automations |
| Logging | CSV + SQLite | Detection history and analysis |
| Containerization | Docker Compose | Service orchestration |

---

## 9. Useful Links

- **Frigate NVR:** https://docs.frigate.video
- **Frigate HA Integration:** https://github.com/blakeblackshear/frigate-hass-integration
- **BirdNET Python Package:** https://pypi.org/project/birdnet/
- **BirdNET-Analyzer:** https://birdnet.cornell.edu/analyzer/
- **iNaturalist API:** https://api.inaturalist.org/v1/docs/
- **Ultralytics YOLOv8:** https://docs.ultralytics.com
- **Tapo RTSP Setup:** https://www.tp-link.com/us/support/faq/2680/
- **Mosquitto MQTT Broker:** https://mosquitto.org
