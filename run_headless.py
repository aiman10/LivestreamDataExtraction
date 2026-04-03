"""
run_headless.py
Drop this in your repo root and run it instead of main.py on Vast.ai.
It patches out all OpenCV display calls so the pipeline runs without a screen.

Usage:
    python run_headless.py
    python run_headless.py --youtube "OTHER_YT_URL"
    python run_headless.py --url "rtsp://..."
"""

import cv2
import types

# --- Patch OpenCV display functions to no-ops BEFORE importing main ---
cv2.imshow = lambda *args, **kwargs: None
cv2.waitKey = lambda *args, **kwargs: -1
cv2.destroyAllWindows = lambda *args, **kwargs: None
cv2.namedWindow = lambda *args, **kwargs: None
cv2.resizeWindow = lambda *args, **kwargs: None

# Also patch resize so it still works but doesn't try to display
_real_resize = cv2.resize
cv2.resize = _real_resize  # resize itself is fine, keep it

print("[headless] OpenCV display patched - running without GUI")
print("[headless] Data logging, InfluxDB, and CSV output unaffected")
print()

# Now import and run main normally
import main
main.main()
