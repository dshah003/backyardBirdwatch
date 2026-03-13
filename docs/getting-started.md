# Getting Started

A step-by-step guide to getting the backyard bird feeder detection system running from scratch.

## Prerequisites

- Tapo C200 camera with a camera account created (see [Configuration](configuration.md))
- Ubuntu 24.04 laptop on the same Wi-Fi network as the camera
- Docker and Docker Compose installed (`docker --version` should work)
- `ffprobe` for stream testing: `sudo apt install ffmpeg`

---

## Step 1: Clone and Configure

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your values
nano .env
```

Fill in every field in `.env`:

| Variable | Where to find it |
|----------|-----------------|
| `CAMERA_IP` | Router DHCP list or Tapo app → Device Info |
| `RTSP_USER` | Tapo app → Camera Settings → Advanced Settings → Camera Account |
| `RTSP_PASSWORD` | Same as above |
| `MQTT_USER` | Choose a username (e.g. `frigate`) |
| `MQTT_PASSWORD` | Choose a strong password |
| `LATITUDE` / `LONGITUDE` | Your approximate GPS coordinates (improves iNat accuracy) |

---

## Step 2: Test the Camera

Verify both RTSP streams are accessible before starting Docker:

```bash
./scripts/test_rtsp.sh
```

You should see stream details for both `stream1` and `stream2`. If it fails, see [Troubleshooting](troubleshooting.md#rtsp-connection-fails).

---

## Step 3: Set Up Mosquitto Authentication

Mosquitto requires the password file to **exist on disk before it starts** — otherwise it crashes on launch.

```bash
# 1. Create an empty password file — must exist before Mosquitto starts
touch mosquitto/config/password_file
chmod 644 mosquitto/config/password_file   # readable so container can start

# 2. Start Mosquitto
docker compose up -d mosquitto

# 3. Add your MQTT user (omit -c — that recreates the file from scratch)
docker exec -it backyardbirdwatch-mosquitto-1 \
  mosquitto_passwd /mosquitto/config/password_file frigate

# 4. Fix ownership and permissions to silence future warnings
docker exec -u root backyardbirdwatch-mosquitto-1 \
  chown mosquitto:mosquitto /mosquitto/config/password_file
docker exec -u root backyardbirdwatch-mosquitto-1 \
  chmod 0700 /mosquitto/config/password_file

# 5. Restart so Mosquitto reloads the populated, correctly-owned file
docker compose restart mosquitto
```

Verify it's healthy:

```bash
docker compose ps mosquitto
# Should show: healthy
```

---

## Step 4: Start the Full Stack

```bash
docker compose up -d
```

This starts three services:
- **mosquitto** — MQTT broker on port 1883
- **frigate** — NVR and object detector on port 5000
- **bird-classifier** — Species classification pipeline

Check that everything is running:

```bash
docker compose ps
docker compose logs -f
```

---

## Step 5: Verify Frigate is Detecting

1. Open the Frigate UI at **http://localhost:5000**
2. You should see a live feed from `bird_feeder` camera
3. Point the camera at your feeder and wait for a bird
4. Frigate should draw a bounding box labeled `bird` with a confidence score

If no detections appear after birds are clearly in frame, see [Troubleshooting](troubleshooting.md#no-detections-in-frigate).

---

## Step 6: Verify Classification is Working

Watch the classifier logs in real time:

```bash
docker compose logs -f bird-classifier
```

When Frigate detects a bird and the event ends, you should see log lines like:

```
2026-02-12T08:15:23 [bird-classifier] INFO: Processing bird detection event: 1707732923.456-abcdef
2026-02-12T08:15:24 [bird-classifier] INFO: iNaturalist result: Northern Cardinal (0.91)
2026-02-12T08:15:24 [bird-classifier] INFO: Logged detection: Northern Cardinal (0.91) via inaturalist
```

Check the data directory for logged detections:

```bash
# View today's detections
sqlite3 data/detections.db \
  "SELECT timestamp, species_common, confidence, classifier FROM detections ORDER BY timestamp DESC LIMIT 10;"

# View snapshot archive
ls data/snapshots/$(date +%Y-%m-%d)/
```

---

## What Happens Next

The system runs continuously. Each bird detection:

1. Triggers a Frigate snapshot (cropped to the bird)
2. Gets classified (local model → iNaturalist API)
3. Is logged to `data/detections.csv` and `data/detections.db`
4. Publishes to MQTT topic `birdfeeder/detection`
5. Snapshot is archived to `data/snapshots/YYYY-MM-DD/`

Low-confidence detections are saved to `data/corrections/` for your review.

---

## Useful Commands

```bash
# Restart a single service
docker compose restart bird-classifier

# View live MQTT traffic
docker compose exec mosquitto mosquitto_sub -u $MQTT_USER -P $MQTT_PASSWORD -t "birdfeeder/#" -v

# Today's species count
sqlite3 data/detections.db \
  "SELECT species_common, COUNT(*) as visits FROM detections WHERE date(timestamp)=date('now') GROUP BY species_common ORDER BY visits DESC;"

# Export last 7 days to CSV
python3 scripts/export_csv.py --days 7 --output weekly.csv
```
