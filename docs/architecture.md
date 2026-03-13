# System Architecture

## Overview

```
Tapo C200 ──RTSP──▶ Frigate NVR ──MQTT──▶ Bird Classifier ──▶ CSV / SQLite
(stream2: detect)       │             frigate/events    │
(stream1: record)       │                               │ MQTT publish
                        │                               ▼
                   Mosquitto ◀────────── birdfeeder/* topics
```

All services run in Docker containers on a single Ubuntu laptop, communicating over a Docker bridge network. Nothing leaves the home network except calls to the iNaturalist API.

---

## Services

### Mosquitto (MQTT Broker)
- Eclipse Mosquitto 2
- Port 1883 (internal + host)
- Authenticated — password file at `mosquitto/config/password_file`
- Persistent message storage in `mosquitto/data/`

### Frigate NVR
- `ghcr.io/blakeblackshear/frigate:stable`
- Port 5000 (web UI), 8554 (RTSP re-stream), 8555 (WebRTC)
- Consumes **stream2** (360p) for motion detection and object detection at 5 fps
- Consumes **stream1** (1080p) for recording and high-res snapshots
- Default SSD MobileNet model detects `bird` and `cat` from the COCO class set
- Publishes detection events to `frigate/events` on Mosquitto
- Saves cropped detection snapshots to `frigate/storage/`

### Bird Classifier
- Custom Python 3.11 service, built from `bird-classifier/Dockerfile`
- Subscribes to `frigate/events`, filters for `bird` and `cat` labels
- Runs the two-stage classification pipeline (see below)
- Publishes enriched detections to `birdfeeder/*` topics
- Logs every detection to CSV and SQLite

---

## Two-Stage Classification Pipeline

```
Frigate "end" event received
        │
        ▼
Download cropped snapshot from Frigate API
        │
        ▼
Stage 1: Local TFHub model (fast, ~50ms, ~900 species)
        │
   conf ≥ 0.85 ──────────────────────────────▶ Use local result
        │
   conf < 0.85
        │
        ▼
Stage 2: iNaturalist CV API (rate-limited, 109K+ taxa, location-aware)
        │
   iNat success ──────────────────────────────▶ Use iNat result
        │
   iNat failed + local result available ──────▶ Use local result
        │
   No result ──────────────────────────────────▶ species = "Unknown"
        │
        ▼
conf < MIN_CONFIDENCE_LOG (0.3) ────────────▶ Drop (no log)
        │
conf ≥ MIN_CONFIDENCE_LOG
        │
        ├── conf < MIN_CONFIDENCE_NOTIFY (0.7)
        │   └── Save to data/corrections/ + publish to birdfeeder/unknown
        │
        └── conf ≥ MIN_CONFIDENCE_NOTIFY
            └── Archive snapshot + publish to birdfeeder/detection
                + check for new species (birdfeeder/new_species)
```

**Why classify only on "end" events?**
A single bird visit generates multiple Frigate events (new, update, end). Processing only the final "end" event avoids duplicate log entries and unnecessary API calls. The "end" event also has the best snapshot — Frigate picks the clearest frame over the course of the detection.

---

## MQTT Topics

| Topic | Publisher | Description |
|-------|-----------|-------------|
| `frigate/events` | Frigate | Raw detection event JSON (all labels) |
| `birdfeeder/detection` | bird-classifier | Every confirmed species detection |
| `birdfeeder/detection/{species-slug}` | bird-classifier | Per-species detections (e.g. `birdfeeder/detection/northern-cardinal`) |
| `birdfeeder/new_species` | bird-classifier | First time a species is seen (this season) |
| `birdfeeder/predator_alert` | bird-classifier | Cat detection near feeder |
| `birdfeeder/unknown` | bird-classifier | Low-confidence detections needing review |
| `birdfeeder/daily_summary` | bird-classifier | End-of-day JSON summary (future) |

---

## Data Storage

```
data/
├── snapshots/
│   └── 2026-02-12/
│       ├── 2026-02-12T08-15-23_northern-cardinal_0.91.jpg
│       └── 2026-02-12T08-17-01_blue-jay_0.88.jpg
├── corrections/          # Low-confidence snapshots for human review
│   └── 1707732923.456-abcdef.jpg
├── detections.csv        # Flat CSV log (append-only)
└── detections.db         # SQLite database (indexed, queryable)
```

Snapshot filenames follow the pattern: `{ISO_timestamp}_{species-slug}_{confidence}.jpg`

---

## Network Ports

| Port | Service | Purpose |
|------|---------|---------|
| 1883 | Mosquitto | MQTT (plaintext, LAN only) |
| 5000 | Frigate | Web UI and REST API |
| 8554 | Frigate | RTSP re-stream |
| 8555 | Frigate | WebRTC (for browser live view) |

All ports are bound to localhost (`127.0.0.1`) by default on the host — accessible from the LAN via the laptop's IP address.

---

## Planned Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Infrastructure | Complete | Docker Compose, Mosquitto, Frigate |
| 2 — Classification | Complete | Two-stage bird-classifier service |
| 3 — Home Assistant | Planned | Lovelace dashboard, automations, alerts |
| 4 — Audio (BirdNET) | Planned | Audio-based species ID from Frigate recordings |
| 5 — Analysis | Planned | Seasonal trends, peak hours, diversity reports |
