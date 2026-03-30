"""
data_writer.py
Handles structured CSV output for time-series metrics and discrete events.
Two separate files follow IoT best practices:
  - metrics.csv: Continuous sensor readings (sampled every few seconds)
  - events.csv: Discrete state changes and threshold crossings
"""

import csv
import os

import config


# Column definitions for each CSV file
METRICS_FIELDS = [
    "timestamp",
    "person_count",
    "vehicle_count",
    "bicycle_count",
    "umbrella_count",
    "backpack_count",
    "suitcase_count",
    "dog_count",
    "is_raining",
    "pedestrian_vehicle_ratio",
    "crowd_density",
    "motion_pct",
    "moving_object_count",
    "activity_level",
    "dominant_flow_dir",
    "avg_flow_speed",
    "brightness",
    "contrast",
    "saturation",
    "color_temp",
    "is_daytime",
    "green_ratio",
    "scene_state",
    "wave_count",
    "total_waves",
    "photo_stop_count",
    "total_photo_stops",
    "friendliness_index",
    "friendliness_level",
]

EVENTS_FIELDS = [
    "timestamp",
    "event_type",
    "severity",
    "metric_value",
    "threshold",
    "description",
    "scene_state",
]


class DataWriter:
    """
    Writes metrics and events to separate CSV files.
    
    Uses append mode so the pipeline can be stopped and restarted
    without losing data. Headers are written only when creating
    a new file. This follows the IoT append-only data pattern.
    
    Usage:
        writer = DataWriter()
        writer.write_metric({"timestamp": "...", "person_count": 12, ...})
        writer.write_event({"timestamp": "...", "event_type": "crowd_warning", ...})
        writer.close()
    """

    def __init__(self, metrics_path=None, events_path=None):
        """
        Args:
            metrics_path: Path to metrics CSV file
            events_path: Path to events CSV file
        """
        self.metrics_path = metrics_path or config.METRICS_CSV_PATH
        self.events_path = events_path or config.EVENTS_CSV_PATH

        # Ensure data directory exists
        for path in [self.metrics_path, self.events_path]:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

        # Write headers for new files
        self._ensure_headers(self.metrics_path, METRICS_FIELDS)
        self._ensure_headers(self.events_path, EVENTS_FIELDS)

        # Open files in append mode and keep them open for performance
        self._metrics_file = open(self.metrics_path, "a", newline="")
        self._events_file = open(self.events_path, "a", newline="")

        self._metrics_writer = csv.DictWriter(
            self._metrics_file, fieldnames=METRICS_FIELDS, extrasaction="ignore"
        )
        self._events_writer = csv.DictWriter(
            self._events_file, fieldnames=EVENTS_FIELDS, extrasaction="ignore"
        )

        self.metrics_count = 0
        self.events_count = 0

        print(f"[DataWriter] Metrics -> {self.metrics_path}")
        print(f"[DataWriter] Events  -> {self.events_path}")

    def _ensure_headers(self, path, fields):
        """Write CSV headers if the file doesn't exist or is empty."""
        write_header = False

        if not os.path.exists(path):
            write_header = True
        elif os.path.getsize(path) == 0:
            write_header = True

        if write_header:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()

    def write_metric(self, row):
        """
        Write a single metric row to metrics.csv.
        
        Args:
            row: dict with keys matching METRICS_FIELDS.
                 Missing keys will be written as empty strings.
                 Extra keys are silently ignored.
        """
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()
        self.metrics_count += 1

    def write_event(self, event):
        """
        Write a single event row to events.csv.
        
        Args:
            event: dict with keys matching EVENTS_FIELDS.
        """
        self._events_writer.writerow(event)
        self._events_file.flush()
        self.events_count += 1

    def write_events(self, events):
        """Write multiple events at once."""
        for event in events:
            self.write_event(event)

    def get_stats(self):
        """Return counts of written rows."""
        return {
            "metrics_rows": self.metrics_count,
            "events_rows": self.events_count,
        }

    def close(self):
        """Close file handles."""
        self._metrics_file.close()
        self._events_file.close()
        print(f"[DataWriter] Closed. Wrote {self.metrics_count} metrics, "
              f"{self.events_count} events.")
