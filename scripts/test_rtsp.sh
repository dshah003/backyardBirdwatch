#!/usr/bin/env bash
# Test RTSP connectivity to the Tapo C200 camera.
# Usage: ./scripts/test_rtsp.sh
#
# Requires: ffprobe (part of ffmpeg)
# Reads credentials from .env file in the project root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Copy .env.example to .env and fill in your camera credentials."
    exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

if [ -z "${CAMERA_IP:-}" ] || [ -z "${RTSP_USER:-}" ] || [ -z "${RTSP_PASSWORD:-}" ]; then
    echo "ERROR: CAMERA_IP, RTSP_USER, and RTSP_PASSWORD must be set in .env"
    exit 1
fi

echo "Testing RTSP stream (low quality - stream2)..."
echo "URL: rtsp://$RTSP_USER:****@$CAMERA_IP:554/stream2"
echo ""

if ffprobe -v quiet -print_format json -show_streams \
    "rtsp://$RTSP_USER:$RTSP_PASSWORD@$CAMERA_IP:554/stream2" 2>/dev/null; then
    echo ""
    echo "Stream2 (detect) OK"
else
    echo "FAILED: Could not connect to stream2"
    echo "Check that:"
    echo "  1. Camera is powered on and connected to Wi-Fi"
    echo "  2. Camera account is set up in the Tapo app"
    echo "  3. CAMERA_IP, RTSP_USER, RTSP_PASSWORD are correct in .env"
    exit 1
fi

echo ""
echo "Testing RTSP stream (high quality - stream1)..."
echo "URL: rtsp://$RTSP_USER:****@$CAMERA_IP:554/stream1"
echo ""

if ffprobe -v quiet -print_format json -show_streams \
    "rtsp://$RTSP_USER:$RTSP_PASSWORD@$CAMERA_IP:554/stream1" 2>/dev/null; then
    echo ""
    echo "Stream1 (record/snapshots) OK"
else
    echo "FAILED: Could not connect to stream1"
    exit 1
fi

echo ""
echo "Both RTSP streams are working!"
