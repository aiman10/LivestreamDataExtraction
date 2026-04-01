"""
detection_logger.py
Appends detection events to CSV and JSON Lines files under the detections/ folder.
"""

import csv
import json
import os
import time

import config

_FIELDS = ["timestamp", "event", "frame_idx", "track_id", "confidence",
           "x1", "y1", "x2", "y2"]


class DetectionLogger:
    """
    Writes one row per detection event to:
      detections/<PHOTO_LOG_CSV>   — comma-separated, with header
      detections/<PHOTO_LOG_JSON>  — JSON Lines (one object per line)
    """

    def __init__(self):
        os.makedirs("detections", exist_ok=True)
        csv_path  = os.path.join("detections", config.PHOTO_LOG_CSV)
        self._json_path = os.path.join("detections", config.PHOTO_LOG_JSON)

        write_header = not os.path.exists(csv_path)
        self._csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_fh, fieldnames=_FIELDS)
        if write_header:
            self._writer.writeheader()

    def log(self, event: str, frame_idx: int, track_id: int,
            confidence: float, box: tuple) -> None:
        x1, y1, x2, y2 = box
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        row = {
            "timestamp":  ts,
            "event":      event,
            "frame_idx":  frame_idx,
            "track_id":   track_id,
            "confidence": round(confidence, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        }
        self._writer.writerow(row)
        self._csv_fh.flush()

        with open(self._json_path, "a", encoding="utf-8") as jf:
            json.dump(row, jf)
            jf.write("\n")

    def close(self) -> None:
        self._csv_fh.close()
