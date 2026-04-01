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
