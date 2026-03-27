# Livestream Analytics Pipeline

**IoT & Big Data - Postgraduate Program**
**(Big) Data Extraction from Livestreams**

A Python application that connects to public city/street livestreams and extracts useful big data information in real time using computer vision. The extracted data is stored as structured time-series CSV and visualized through a Grafana dashboard.

## Architecture

```
Livestream  ->  OpenCV + YOLO  ->  Event Engine  ->  CSV / InfluxDB  ->  Grafana
 (Sensor)      (Edge Processing)  (Stream Proc.)     (Storage)         (Dashboard)
```

**Data extracted per frame:**

- Person, vehicle, bicycle, umbrella, backpack, dog counts (YOLOv8)
- Motion percentage and activity level (background subtraction)
- Brightness, contrast, color temperature, saturation (scene analysis)
- Dominant pedestrian flow direction and speed (optical flow)
- Day/night classification, rain detection (umbrella proxy)
- Scene state (quiet/normal/busy/critical via state machine)

**Events generated:**

- Crowd density warnings and critical alerts
- Rain onset/offset (umbrella-based)
- Sunrise/sunset transitions
- Activity spikes (rolling anomaly detection)
- Scene state transitions

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Docker and Docker Compose (for Grafana dashboard)
- A webcam or public livestream URL

### Step 1: Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/livestream-analytics.git
cd livestream-analytics
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` - computer vision and video processing
- `numpy`, `pandas` - numerical computing and data handling
- `yt-dlp` - YouTube livestream URL extraction
- `ultralytics` - YOLOv8 object detection
- `influxdb-client` - InfluxDB integration for Grafana

The YOLOv8 nano model (~6MB) downloads automatically on first run.

### Step 4: Configure your stream source

Edit `config.py` and set your livestream URL:

```python
# Option A: Direct MJPEG stream (simplest)
STREAM_URL = "http://wmccpinetop.axiscam.net/mjpg/video.mjpg"
IS_YOUTUBE = False

# Option B: YouTube livestream
STREAM_URL = "https://www.youtube.com/watch?v=YOUR_LIVE_ID"
IS_YOUTUBE = True
```

Find public streams at: https://webcams24.live/

## Usage

### Run the pipeline

```bash
# Default (uses config.py settings, shows video window)
python main.py

# With a specific MJPEG stream
python main.py --url "http://example.com/stream.mjpg"

# With a YouTube livestream
python main.py --youtube "https://www.youtube.com/watch?v=LIVE_ID"

# Headless mode (no video window, for servers)
python main.py --no-display

# Without YOLO (faster, only motion/scene analysis)
python main.py --no-yolo
```

### Controls during operation

- Press **q** to stop the pipeline
- Press **s** to save a screenshot

### Output files

After running, data is saved to:
- `data/metrics.csv` - continuous time-series metrics (1 row per second)
- `data/events.csv` - discrete events (threshold crossings, state changes)

## Grafana Dashboard Setup

### Step 1: Start InfluxDB and Grafana

```bash
docker-compose up -d
```

### Step 2: Import CSV data into InfluxDB

First update the token in `config.py`:

```python
INFLUXDB_TOKEN = "my-super-secret-token"
```

Then run the importer:

```bash
python influx_importer.py
```

### Step 3: Configure Grafana

1. Open http://localhost:3000 (login: admin/admin)
2. Go to **Connections** > **Data Sources** > **Add data source**
3. Select **InfluxDB**
4. Configure:
   - Query Language: **Flux**
   - URL: `http://influxdb:8086`
   - Organization: `myorg`
   - Token: `my-super-secret-token`
   - Default Bucket: `video_analytics`
5. Click **Save & Test**

### Step 4: Create dashboard panels

Example Flux query for person count over time:

```flux
from(bucket: "video_analytics")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "scene_metrics")
  |> filter(fn: (r) => r._field == "person_count")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
```

## Project Structure

```
livestream-analytics/
  main.py                 # Pipeline orchestrator
  config.py               # All settings and thresholds
  stream_capture.py       # Threaded livestream reader
  feature_extractors.py   # CV analysis (scene, motion, flow, YOLO)
  event_engine.py         # Threshold, anomaly, and state machine events
  data_writer.py          # CSV output handler
  influx_importer.py      # CSV to InfluxDB loader
  requirements.txt        # Python dependencies
  docker-compose.yml      # Grafana + InfluxDB containers
  data/
    metrics.csv           # Generated time-series data
    events.csv            # Generated event log
```

## IoT / Big Data Context

This pipeline demonstrates a complete IoT data flow:

1. **Perception Layer**: The webcam acts as a multi-modal IoT sensor
2. **Edge Processing**: OpenCV + YOLO process video locally, reducing ~150 MB/min of raw video to ~1 KB/min of structured metrics (99.999% data reduction)
3. **Stream Processing**: The event engine detects meaningful state changes in real time
4. **Storage Layer**: CSV/InfluxDB provides append-only time-series storage
5. **Presentation Layer**: Grafana dashboards enable monitoring and alerting

The pipeline addresses the 3 Vs of Big Data:
- **Volume**: Thousands of data points per hour from continuous sampling
- **Velocity**: Real-time processing and event generation
- **Variety**: Structured metrics, semi-structured events, and raw video coexist

## Technologies Used

- Python 3, OpenCV, NumPy
- YOLOv8 (ultralytics) for object detection
- Background subtraction (MOG2) for motion analysis
- Farneback optical flow for direction analysis
- InfluxDB for time-series storage
- Grafana for visualization and dashboards
- Docker / Docker Compose
