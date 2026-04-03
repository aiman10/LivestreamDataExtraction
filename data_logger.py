"""
data_logger.py
Writes one CSV row per YOLO detection cycle, capturing every metric
produced by the analytics pipeline for later visualization and analysis.
"""

import csv
import json
import os

import config

_COLUMNS = [
    # -- Time context --
    "timestamp",                # ISO 8601 with timezone
    "dublin_time",              # HH:MM:SS local
    "time_period",              # Night / Morning / Afternoon / Evening / Late Night
    "frame_number",

    # -- Detection counts --
    "person_count",
    "background_person_count",
    "vehicle_count",
    "bicycle_count",
    "umbrella_count",
    "total_objects",
    "crowd_level",              # NORMAL / CROWDED / VERY CROWDED

    # -- Behaviour events --
    "waving_count",
    "photo_taking_count",

    # -- Crowd safety --
    "safety_status",            # NORMAL / WARNING / CRITICAL
    "choke_point_count",
    "grid_max_density",         # Highest single-cell person count
    "baseline_count",           # Rolling average person count
    "active_alerts",            # Semicolon-separated alert types

    # -- Crowd density grid (for spatial analysis) --
    "grid_density_json",        # Full grid as JSON, e.g. [[0,1],[2,0],[0,0]]

    # -- Performance --
    "fps",
]


class DataLogger:
    """
    Append-only CSV logger.  Opens the file once at init, writes a header
    if the file is new, then accepts one ``log()`` call per YOLO cycle.

    Usage:
        logger = DataLogger()
        logger.log(timestamp=..., dublin_time=..., ...)
        logger.close()
    """

    def __init__(self):
        os.makedirs(config.DATA_LOG_DIR, exist_ok=True)
        self._path = os.path.join(config.DATA_LOG_DIR, config.DATA_LOG_FILE)

        write_header = not os.path.exists(self._path)
        self._fh = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=_COLUMNS)
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

        print(f"[DataLogger] Logging to {self._path}")

    def log(
        self,
        timestamp: str,
        dublin_time: str,
        time_period: str,
        frame_number: int,
        summary: dict,
        crowd_level: str,
        waving_count: int,
        photo_taking_count: int,
        safety_status: str = "NORMAL",
        choke_point_count: int = 0,
        grid_max_density: int = 0,
        baseline_count: float = 0.0,
        active_alerts: str = "",
        grid_density_json: str = "[]",
        fps: float = 0.0,
    ) -> None:
        """Write one row with all analytics data for the current cycle."""
        row = {
            "timestamp":              timestamp,
            "dublin_time":            dublin_time,
            "time_period":            time_period,
            "frame_number":           frame_number,
            "person_count":           summary.get("person_count", 0),
            "background_person_count": summary.get("background_person_count", 0),
            "vehicle_count":          summary.get("vehicle_count", 0),
            "bicycle_count":          summary.get("bicycle_count", 0),
            "umbrella_count":         summary.get("umbrella_count", 0),
            "total_objects":          summary.get("total_objects", 0),
            "crowd_level":            crowd_level,
            "waving_count":           waving_count,
            "photo_taking_count":     photo_taking_count,
            "safety_status":          safety_status,
            "choke_point_count":      choke_point_count,
            "grid_max_density":       grid_max_density,
            "baseline_count":         round(baseline_count, 1),
            "active_alerts":          active_alerts,
            "grid_density_json":      grid_density_json,
            "fps":                    round(fps, 1),
        }
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
        print(f"[DataLogger] Closed {self._path}")
