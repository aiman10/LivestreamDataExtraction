#!/bin/bash
# setup_runpod.sh
# Run this once in the RunPod web terminal or via SSH.
# Handles all deps, clones the repo, and launches the pipeline in the background.
#
# Usage:
#   bash setup_runpod.sh

set -e

echo "=============================================="
echo "  Dublin Livestream - RunPod Setup Script"
echo "=============================================="
echo ""

# --- 1. System packages ---
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    screen \
    git

# --- 2. Clone repo into persistent storage ---
# /workspace is RunPod's persistent volume - survives pod restarts
echo ""
echo "[2/6] Cloning repo into /workspace (persistent storage)..."
cd /workspace

if [ -d "LivestreamDataExtraction" ]; then
    echo "  Repo already exists, pulling latest..."
    cd LivestreamDataExtraction
    git pull
    cd ..
else
    git clone https://github.com/aiman10/LivestreamDataExtraction.git
fi

cd LivestreamDataExtraction

# --- 3. Python dependencies ---
echo ""
echo "[3/6] Installing Python dependencies..."
pip install --quiet --upgrade pip

# Install from requirements.txt first
if [ -f requirements.txt ]; then
    echo "  Installing from requirements.txt..."
    pip install --quiet -r requirements.txt
fi

# Ensure headless OpenCV (no display needed), yt-dlp for YouTube stream
pip install --quiet \
    opencv-python-headless \
    yt-dlp \
    influxdb-client \
    ultralytics

# --- 4. Verify GPU + YOLO models ---
echo ""
echo "[4/6] Checking GPU and YOLO model files..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No GPU found - running on CPU')
import os
for model in ['yolov8n.pt', 'yolov8m-pose.pt']:
    status = 'FOUND' if os.path.exists(model) else 'MISSING'
    print(f'  {model}: {status}')
"

# --- 5. Write the headless runner ---
echo ""
echo "[5/6] Writing run_headless.py..."
cat > run_headless.py << 'PYEOF'
"""
run_headless.py
Patches out OpenCV GUI calls so main.py runs on a headless server.
Place this in the repo root and run it instead of main.py.
"""
import cv2

# No-op all display functions
cv2.imshow = lambda *a, **k: None
cv2.waitKey = lambda *a, **k: -1   # returns -1 = no key pressed, loop continues
cv2.destroyAllWindows = lambda *a, **k: None
cv2.namedWindow = lambda *a, **k: None
cv2.resizeWindow = lambda *a, **k: None

print("[headless] OpenCV display patched - no screen needed")
print()

import main
main.main()
PYEOF

# --- 6. Launch in a screen session ---
echo ""
echo "[6/6] Launching pipeline in screen session 'dublin'..."

# Kill any existing session with same name
screen -S dublin -X quit 2>/dev/null || true

screen -dmS dublin bash -c "
    cd /workspace/LivestreamDataExtraction
    echo 'Pipeline starting at '\$(date)
    python3 run_headless.py 2>&1 | tee pipeline.log
    echo 'Pipeline stopped at '\$(date)
"

sleep 2

# Confirm it's running
if screen -list | grep -q "dublin"; then
    echo ""
    echo "=============================================="
    echo "  Pipeline is RUNNING in screen 'dublin'"
    echo ""
    echo "  Watch live output:"
    echo "    screen -r dublin"
    echo "    (Ctrl+A then D to detach)"
    echo ""
    echo "  Check log:"
    echo "    tail -f /workspace/LivestreamDataExtraction/pipeline.log"
    echo ""
    echo "  Stop pipeline:"
    echo "    screen -S dublin -X quit"
    echo ""
    echo "  Data is saved to:"
    echo "    /workspace/LivestreamDataExtraction/data/"
    echo "=============================================="
else
    echo "  ERROR: screen session did not start. Check for errors above."
    exit 1
fi
