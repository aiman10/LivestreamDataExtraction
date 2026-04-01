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
