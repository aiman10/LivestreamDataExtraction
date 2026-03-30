"""
Configuration for the Livestream Analytics Pipeline.
Adjust these values to match your stream source and hardware capabilities.
"""

# ============================================================
# STREAM SOURCE
# ============================================================
# Option 1: Direct MJPEG stream (easiest, works out of the box)
# STREAM_URL = "http://wmccpinetop.axiscam.net/mjpg/video.mjpg"

# Option 2: YouTube livestream URL (requires yt-dlp installed)
# STREAM_URL = "https://www.youtube.com/watch?v=3nyPER2kzqk"

# Option 3: RTSP camera
# STREAM_URL = "rtsp://username:password@camera-ip:554/stream"

# Default: a public MJPEG stream (replace with your chosen source)
STREAM_URL = "https://www.youtube.com/watch?v=3nyPER2kzqk"

# Set to True if STREAM_URL is a YouTube link
IS_YOUTUBE = True

# YouTube resolution cap (lower = faster processing)
YOUTUBE_RESOLUTION = "480p"

# ============================================================
# FRAME PROCESSING INTERVALS
# ============================================================
# How often (in frames) to run each analysis layer.
# At 30 fps: every 1 frame = 33ms, every 30 frames = 1s

SCENE_EVERY = 30        # Scene-level metrics (brightness, color temp)
MOTION_EVERY = 1        # Background subtraction (cheap, run every frame)
FLOW_EVERY = 5          # Optical flow direction analysis
YOLO_EVERY = 30         # Object detection with YOLO (expensive)
LOG_EVERY = 30          # Write one row to CSV per this many frames

# ============================================================
# YOLO CONFIGURATION
# ============================================================
YOLO_MODEL = "yolov8n.pt"   # nano model, ~6MB, fast on CPU
YOLO_CONFIDENCE = 0.4        # Minimum detection confidence (0.0-1.0)

# ============================================================
# EVENT ENGINE THRESHOLDS
# ============================================================
# Crowd thresholds
CROWD_WARNING_THRESHOLD = 25
CROWD_CRITICAL_THRESHOLD = 40

# Activity spike: fire event when motion_pct exceeds this multiplier of rolling avg
ACTIVITY_SPIKE_MULTIPLIER = 2.0

# Event cooldown in seconds (prevents spam)
EVENT_COOLDOWN_SECONDS = 30

# Rolling anomaly detector
ANOMALY_WINDOW_SIZE = 12     # Number of samples in the rolling window
ANOMALY_NUM_STD = 2.0        # Standard deviations for anomaly boundary

# Scene state machine persistence (consecutive readings before state change)
STATE_PERSISTENCE_COUNT = 3

# State machine person_count thresholds
STATE_QUIET_TO_NORMAL = 5
STATE_NORMAL_TO_BUSY = 20
STATE_BUSY_TO_CRITICAL = 40
STATE_CRITICAL_TO_BUSY = 30
STATE_BUSY_TO_NORMAL = 15
STATE_NORMAL_TO_QUIET = 3

# ============================================================
# ENGAGEMENT / FRIENDLINESS DETECTION
# ============================================================
ENGAGEMENT_EVERY = 5            # Run engagement analysis every N frames

# Wave detection (MediaPipe Pose on YOLO person crops)
WAVE_MIN_BOX_WIDTH = 50         # Minimum person bbox width in pixels
WAVE_MIN_BOX_HEIGHT = 100       # Minimum person bbox height in pixels
WAVE_MAX_PERSONS = 5            # Max persons to run pose estimation on per frame
WAVE_WRIST_ABOVE_SHOULDER_THRESHOLD = 0.15  # Normalized y-distance wrist must be above shoulder
WAVE_LATERAL_MOVEMENT_THRESHOLD = 0.03      # Min lateral oscillation to count as wave
WAVE_OSCILLATION_WINDOW = 10    # Frames to track wrist x-positions for oscillation
WAVE_MIN_DIRECTION_CHANGES = 2  # Min direction reversals to count as wave
WAVE_COOLDOWN_SECONDS = 3       # Cooldown per tracked person before re-counting a wave

# Photo-stop detection (stationary person tracking)
PHOTO_STOP_STATIONARY_FRAMES = 15       # Consecutive frames a person must be still
PHOTO_STOP_MOVEMENT_THRESHOLD = 20      # Max centroid pixel movement to count as stationary
PHOTO_STOP_MATCH_DISTANCE = 50          # Max centroid distance for cross-frame matching

# Friendliness index
FRIENDLINESS_WINDOW_MINUTES = 5         # Rolling window duration
FRIENDLINESS_MAX_WAVES = 10             # Wave count that maps to 100 on wave component
FRIENDLINESS_MAX_STOPS = 8              # Photo-stop count that maps to 100 on stop component
FRIENDLINESS_WAVE_WEIGHT = 0.6          # Weight of wave component in combined score
FRIENDLINESS_STOP_WEIGHT = 0.4          # Weight of photo-stop component in combined score

# Engagement event thresholds
ENGAGEMENT_SPIKE_THRESHOLD = 70         # Friendliness index above this triggers spike event
ENGAGEMENT_DROP_THRESHOLD = 20          # Friendliness index below this triggers drop event
ENGAGEMENT_DROP_PREVIOUS_MIN = 40       # Previous index must be above this for drop to fire

# ============================================================
# DATA STORAGE
# ============================================================
METRICS_CSV_PATH = "data/metrics.csv"
EVENTS_CSV_PATH = "data/events.csv"

# ============================================================
# INFLUXDB (for Grafana integration)
# ============================================================
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "YOUR_TOKEN_HERE"
INFLUXDB_ORG = "myorg"
INFLUXDB_BUCKET = "video_analytics"

# ============================================================
# DISPLAY
# ============================================================
SHOW_VIDEO = True            # Set False for headless/server mode
DISPLAY_WIDTH = 960          # Resize display window width (None = original)
DISPLAY_HEIGHT = 540         # Resize display window height (None = original)
OVERLAY_METRICS = True       # Show live metrics overlay on video
