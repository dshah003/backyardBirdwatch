# Configuration Reference

All service configuration is managed through the `.env` file and two service config files.

---

## Environment Variables (`.env`)

Copy `.env.example` to `.env` and fill in your values. Never commit `.env` to version control.

### Camera

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_IP` | — | Static IP of the Tapo C200 on your LAN |
| `RTSP_USER` | — | Camera account username (set in Tapo app) |
| `RTSP_PASSWORD` | — | Camera account password |

**Setting up the camera account:**
1. Tapo app → select your C200 → Settings (gear icon)
2. Advanced Settings → Camera Account
3. Create a username and password (6–32 chars, avoid `@` and `:`)
4. Assign a static IP via your router's DHCP reservation table

### MQTT

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_USER` | — | Username for Mosquitto authentication |
| `MQTT_PASSWORD` | — | Password for Mosquitto authentication |

### Location

| Variable | Default | Description |
|----------|---------|-------------|
| `LATITUDE` | — | Decimal latitude of your feeder (e.g. `40.7128`) |
| `LONGITUDE` | — | Decimal longitude (e.g. `-74.0060`) |

Location is passed to the iNaturalist API for geo-filtered results, which significantly improves accuracy by narrowing candidates to species found in your region.

### Classification Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `INAT_RATE_LIMIT_PER_MIN` | `30` | Max iNaturalist API calls per minute |
| `MIN_CONFIDENCE_LOG` | `0.3` | Detections below this threshold are dropped entirely |
| `MIN_CONFIDENCE_NOTIFY` | `0.7` | Detections below this go to `corrections/` for review |

**Tuning guidance:**
- If you're getting too many false positives (non-bird images logged), increase `MIN_CONFIDENCE_LOG` to `0.5`
- If you're missing detections of hard-to-photograph species (wrens, sparrows), lower `MIN_CONFIDENCE_NOTIFY` to `0.5` and review the `corrections/` folder more often
- The iNaturalist rate limit default (30/min) is conservative; the public API is undocumented but generally tolerant of up to 60/min from a single IP

---

## Frigate Config (`frigate/config.yml`)

Frigate uses `{VARIABLE}` syntax to substitute values from environment variables set in `docker-compose.yml`.

### Key settings

```yaml
detect:
  width: 640
  height: 480
  fps: 5          # 5 fps is sufficient for bird activity; higher = more CPU
```

**`fps` tuning:** At 5 fps, a 2-second bird visit generates ~10 frames. Increasing to 10 fps doubles CPU usage with minimal benefit for detection quality.

```yaml
objects:
  track:
    - bird
    - cat
  filters:
    bird:
      min_score: 0.4    # Minimum object detection confidence
      min_area: 1000    # Minimum bounding box area (pixels²), filters out distant/tiny detections
```

**`min_area` tuning:** 1000 px² corresponds to roughly a 32×32 pixel box. If you're getting false detections from small objects (leaves blowing, insects), increase to `2000`–`4000`.

```yaml
snapshots:
  enabled: true
  crop: true        # Crops to the detected object — required for the classifier
```

`crop: true` is essential. The classifier receives the cropped bird image, not the full frame.

```yaml
record:
  enabled: true
  retain:
    days: 7         # Continuous recording retention
  events:
    retain:
      default: 30   # Retain event clips for 30 days
```

Adjust retention based on available disk space. At 360p/5fps the detect stream doesn't record — only the 1080p stream1 is recorded.

---

## Mosquitto Config (`mosquitto/config/mosquitto.conf`)

```
listener 1883
allow_anonymous false
password_file /mosquitto/config/password_file
persistence true
persistence_location /mosquitto/data/
log_dest stdout
```

The `allow_anonymous false` line requires all clients to authenticate. The `password_file` is created with:

```bash
docker compose exec mosquitto mosquitto_passwd -c /mosquitto/config/password_file <username>
```

To add additional users without overwriting the file, omit `-c`:
```bash
docker compose exec mosquitto mosquitto_passwd /mosquitto/config/password_file <new_username>
```

---

## Bird Classifier Environment Variables

These are set in `docker-compose.yml` from the root `.env` file. You generally don't need to edit them directly.

| Variable | Description |
|----------|-------------|
| `MQTT_HOST` | Hostname of the MQTT broker (default: `mosquitto`) |
| `MQTT_PORT` | MQTT port (default: `1883`) |
| `FRIGATE_URL` | Frigate API base URL (default: `http://frigate:5000`) |
| `DATA_DIR` | Where to store detections, snapshots, and logs (default: `/data`) |
| `FRIGATE_MEDIA_DIR` | Mount path of Frigate storage (default: `/media/frigate`) |

---

## species_list.txt

Located at `bird-classifier/species_list.txt`. Contains one species common name per line.

This file is currently informational — it documents the species expected at your feeder. It is not used as a hard filter; the classifier will log any species it identifies, regardless of whether it appears in this file.

In a future phase, this list can be passed to BirdNET as a custom species filter to improve audio classification accuracy.

To update the list: edit the file and rebuild the container:
```bash
docker compose build bird-classifier
docker compose up -d bird-classifier
```
