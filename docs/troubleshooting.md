# Troubleshooting

## RTSP Connection Fails

**Symptom:** `scripts/test_rtsp.sh` fails or Frigate shows no camera feed.

**Checks:**
1. Can you ping the camera?
   ```bash
   ping $CAMERA_IP
   ```
2. Is the camera account set up in the Tapo app (not just the Tapo account password)?
   - Tapo app → Camera → Settings → Advanced Settings → Camera Account
3. Does VLC connect?
   ```
   vlc rtsp://<user>:<pass>@<camera-ip>:554/stream2
   ```
4. Are there special characters (`@`, `:`, `/`) in the RTSP password? URL-encode them or choose a simpler password.
5. Is the camera on the same network/VLAN as the laptop?

---

## Mosquitto Won't Start

**Symptom:** `docker compose ps` shows mosquitto as unhealthy or restarting.

**Check the logs:**
```bash
docker compose logs mosquitto
```

**Common causes:**

- **Password file missing or unreadable:** Mosquitto crashes at startup if `password_file` doesn't exist or the container user can't read it. Create it on the host first, then add the user:
  ```bash
  touch mosquitto/config/password_file
  chmod 644 mosquitto/config/password_file
  docker compose restart mosquitto
  # Once running, add the user (no -c flag — that would recreate the file)
  docker exec -it backyardbirdwatch-mosquitto-1 \
    mosquitto_passwd /mosquitto/config/password_file frigate
  docker compose restart mosquitto
  ```

- **Config file permissions:** Mosquitto refuses to start if `mosquitto.conf` is world-writable.
  ```bash
  chmod 644 mosquitto/config/mosquitto.conf
  ```

---

## Frigate Shows No Detections

**Symptom:** Frigate UI loads but no bounding boxes appear even with birds in frame.

**Checks:**

1. **Is the detect stream working?** In the Frigate UI, the camera should show a live feed. If it's black or says "offline", fix the RTSP stream first.

2. **Is the object detector loaded?** Check logs:
   ```bash
   docker compose logs frigate | grep -i detector
   ```

3. **Is the confidence threshold too high?** Try lowering in `frigate/config.yml`:
   ```yaml
   filters:
     bird:
       min_score: 0.3   # was 0.4
   ```
   Then restart: `docker compose restart frigate`

4. **Is `min_area` filtering out detections?** If your feeder is far from the camera, birds may appear smaller than 1000 px². Try `min_area: 500`.

5. **CPU too slow?** On very low-end hardware, the detector may lag. Try `fps: 2` in the detect config.

---

## Bird Classifier Not Logging Detections

**Symptom:** Frigate detects birds, but nothing appears in `data/detections.csv` or the classifier logs show nothing.

**Check classifier logs:**
```bash
docker compose logs -f bird-classifier
```

**Common causes:**

- **MQTT not authenticated:** Look for connection errors. Verify `MQTT_USER` and `MQTT_PASSWORD` match what you set in the Mosquitto password file.

- **No "end" events from Frigate:** The classifier only processes `end` events. Make sure detections are completing (the bird leaves the frame and Frigate closes the event). In the Frigate UI, completed events appear in the Events tab.

- **Frigate API unreachable:** The classifier downloads snapshots from `http://frigate:5000/api/events/{id}/snapshot.jpg`. Check:
  ```bash
  docker compose exec bird-classifier curl -s http://frigate:5000/api/version
  ```

- **iNaturalist API rate limit hit:** Look for `429` or timeout errors in the logs. If too many events are queued, the classifier may fall behind. The rate limit is configurable via `INAT_RATE_LIMIT_PER_MIN`.

---

## iNaturalist Always Returns "Unknown"

**Symptom:** All iNat results come back empty or with very low confidence.

**Checks:**

1. **Is the snapshot usable?** View a snapshot from `data/corrections/` — is the bird visible and in focus? The C200 at 1080p should produce clear images at 1–3m.

2. **Is the crop reasonable?** If Frigate's bounding box is too large (includes the whole frame) or too small, iNat will struggle. Check `min_area` in Frigate config.

3. **Are coordinates set?** Geo-filtering greatly improves results. Confirm `LATITUDE` and `LONGITUDE` are set in `.env`.

4. **Test manually:**
   ```bash
   curl -X POST "https://api.inaturalist.org/v1/computervisions/score_image" \
     -F "image=@data/corrections/<some-file>.jpg" \
     -F "lat=$LATITUDE" -F "lng=$LONGITUDE" | python3 -m json.tool
   ```

---

## Disk Space Running Out

**Symptom:** Services start failing or logs show write errors.

Frigate recordings are the main culprit. Reduce retention:

```yaml
# frigate/config.yml
record:
  retain:
    days: 3        # was 7
  events:
    retain:
      default: 14  # was 30
```

Check current disk usage:
```bash
du -sh frigate/storage/
du -sh data/snapshots/
```

To manually clear old Frigate recordings:
```bash
find frigate/storage/recordings -name "*.mp4" -mtime +7 -delete
```

---

## Viewing Live MQTT Traffic

Useful for verifying the full pipeline end-to-end:

```bash
# Subscribe to all birdfeeder topics
docker compose exec mosquitto \
  mosquitto_sub -u $MQTT_USER -P $MQTT_PASSWORD -t "birdfeeder/#" -v

# Subscribe to raw Frigate events
docker compose exec mosquitto \
  mosquitto_sub -u $MQTT_USER -P $MQTT_PASSWORD -t "frigate/events" -v
```

---

## Rebuilding After Code Changes

If you edit any file in `bird-classifier/`:

```bash
docker compose build bird-classifier
docker compose up -d bird-classifier
```

If you edit `frigate/config.yml` or `mosquitto/config/mosquitto.conf`:

```bash
docker compose restart frigate   # or mosquitto
```
