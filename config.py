"""
config.py
All settings for Phase 1: livestream capture + YOLO detection display.
"""

# -- Stream ------------------------------------------------------------------
STREAM_URL = "https://www.youtube.com/watch?v=3nyPER2kzqk"  # EarthCam Dublin
IS_YOUTUBE = True
YOUTUBE_RESOLUTION = "480p"

# -- YOLO --------------------------------------------------------------------
YOLO_MODEL = "yolov8n.pt"      # Nano model, ~6 MB, auto-downloads first run
YOLO_CONFIDENCE = 0.4           # Minimum detection confidence
YOLO_EVERY = 2                  # Run detection every N frames (1 = every frame)

# -- Background / Small Person Detection ------------------------------------
SMALL_PERSON_CONF_THRESHOLD = 0.15   # Relaxed confidence for small person boxes
SMALL_PERSON_HEIGHT_RATIO   = 0.15   # Box height < 15% of frame height = "small"
BACKGROUND_ZONE_RATIO       = 0.60   # Top 60% of frame treated as background zone

# -- Display -----------------------------------------------------------------
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
SHOW_LABELS = True              # Draw class name + confidence on boxes
SHOW_COUNTS = True              # Show object count overlay

# -- Wave Detection ----------------------------------------------------------
WAVE_ENABLED         = True
WAVE_MODEL           = "yolov8m-pose.pt"  # Medium pose model — best wrist accuracy
WAVE_EVERY           = 2                   # Run pose inference every N frames
WAVE_CONFIDENCE      = 0.40
WAVE_KEYPOINT_CONF   = 0.40               # Min keypoint confidence to trust
WAVE_RAISE_FRACTION  = 0.15               # Wrist must be this fraction of body height above shoulder
WAVE_MIN_AMPLITUDE   = 30                 # Min horizontal wrist travel in pixels
WAVE_MIN_REVERSALS   = 3                  # Direction changes needed to confirm oscillation
WAVE_MIN_SAMPLES     = 8                  # Min raised-hand samples in history
WAVE_CONFIRM_FRAMES  = 3                  # Consecutive detections before wave is confirmed
WAVE_HOLD_SECONDS    = 2.0                # Seconds to hold wave indicator after last detection
WAVE_TRACKER_IOU     = 0.35               # IoU threshold for track matching
WAVE_TRACKER_MISSING = 15                 # Frames before a track is discarded
WAVE_SHOW_SKELETON   = True               # Draw arm skeleton on detected persons
WAVE_LOG_CSV         = "wave_detections.csv"
WAVE_LOG_JSON        = "wave_detections.jsonl"

# -- Photo-Taking Detection --------------------------------------------------
PHOTO_ENABLED          = True
PHOTO_MODEL            = "yolov8m-pose.pt"   # Shared with wave detector
PHOTO_EVERY            = 2                    # Run every N frames
PHOTO_CONFIDENCE       = 0.40
PHOTO_KEYPOINT_CONF    = 0.40                 # Min keypoint confidence to trust
PHOTO_STILL_PIXELS     = 20                   # Max centroid delta between consecutive frames (px)
PHOTO_HEAD_TOLERANCE   = 60                   # Nose-to-wrist-midpoint X tolerance (px)
PHOTO_MIN_DURATION_SEC = 1.5                  # Seconds all conditions must hold continuously
PHOTO_HISTORY_FRAMES   = 30                   # Rolling window size (pose inference frames, ~2 s)
PHOTO_DISPLAY_CONF     = 0.75                 # Min confidence fraction to show label
PHOTO_HOLD_SECONDS     = 2.0                  # Seconds to hold indicator after detection
PHOTO_TRACKER_IOU      = 0.35                 # IoU threshold for track matching
PHOTO_TRACKER_MISSING  = 15                   # Pose frames before a track is discarded
PHOTO_LOG_CSV          = "photo_detections.csv"
PHOTO_LOG_JSON         = "photo_detections.jsonl"

# -- Crowd Safety & Anomaly Detection ----------------------------------------
CROWD_SAFETY_ENABLED        = True
CROWD_SAFETY_EVERY          = 2       # Analyze every N YOLO cycles
CROWD_GRID_COLS             = 5       # Horizontal grid cells
CROWD_GRID_ROWS             = 3       # Vertical grid cells
CROWD_DENSITY_THRESHOLD     = 4       # People per cell → choke point
CROWD_SURGE_PCT             = 80      # % increase in count → surge alert
CROWD_DISPERSAL_PCT         = 60      # % decrease in count → dispersal alert
CROWD_HISTORY_SECONDS       = 10      # Rolling window for baseline (seconds)
CROWD_GATHERING_FRAMES      = 8       # Consecutive dense frames → gathering alert

# -- Data Logging ------------------------------------------------------------
DATA_LOG_ENABLED    = True
DATA_LOG_DIR        = "data"
DATA_LOG_FILE       = "livestream_analytics.csv"
