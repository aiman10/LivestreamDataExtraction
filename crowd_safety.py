"""
crowd_safety.py
Grid-based crowd density analysis for detecting choke points,
surges, dispersals, and unusual gathering patterns.

Works entirely from YOLO person detections — no tracking dependency.
"""

import time
from collections import deque

import cv2
import numpy as np

import config


class SafetyAlert:
    """One active safety alert."""
    __slots__ = ("alert_type", "severity", "message", "cell", "timestamp")

    def __init__(self, alert_type: str, severity: str, message: str,
                 cell: tuple | None = None):
        self.alert_type = alert_type    # CHOKE_POINT / CROWD_SURGE / DISPERSAL / GATHERING
        self.severity = severity        # WARNING / CRITICAL
        self.message = message
        self.cell = cell                # (col, row) or None for frame-wide
        self.timestamp = time.time()


class SafetyReport:
    """Result of one analysis cycle."""
    __slots__ = ("grid_density", "alerts", "total_count",
                 "baseline_count", "choke_cells", "status")

    def __init__(self):
        self.grid_density: list[list[int]] = []
        self.alerts: list[SafetyAlert] = []
        self.total_count: int = 0
        self.baseline_count: float = 0.0
        self.choke_cells: list[tuple] = []
        self.status: str = "NORMAL"     # NORMAL / WARNING / CRITICAL


# Density-to-colour mapping (BGR) for the heatmap overlay
_DENSITY_COLORS = [
    (0,   0,   0  ),   # 0 — transparent (no tint)
    (0,   80,  0  ),   # 1 — dim green
    (0,   160, 0  ),   # 2 — green
    (0,   200, 200),   # 3 — yellow
    (0,   140, 255),   # 4 — orange
    (0,   60,  255),   # 5 — red-orange
]
_COLOR_MAX = (0, 0, 255)  # 6+ — red


def _color_for_density(count: int) -> tuple:
    if count >= len(_DENSITY_COLORS):
        return _COLOR_MAX
    return _DENSITY_COLORS[count]


class CrowdSafetyAnalyzer:
    """
    Divides the frame into a grid and tracks per-cell person density
    over time to detect safety-relevant crowd patterns.

    Usage:
        analyzer = CrowdSafetyAnalyzer()
        report = analyzer.analyze(detections, frame_shape)
        frame = analyzer.draw_overlay(frame, report)
    """

    def __init__(self):
        self.cols = config.CROWD_GRID_COLS
        self.rows = config.CROWD_GRID_ROWS
        self.density_thresh = config.CROWD_DENSITY_THRESHOLD
        self.surge_pct = config.CROWD_SURGE_PCT
        self.dispersal_pct = config.CROWD_DISPERSAL_PCT
        self.gathering_frames = config.CROWD_GATHERING_FRAMES

        # Rolling history  — each entry is (timestamp, total_count, grid_snapshot)
        self._history: deque = deque(maxlen=200)

        # Per-cell consecutive-dense counter for gathering detection
        self._cell_dense_streak = [[0] * self.cols for _ in range(self.rows)]

    # ------------------------------------------------------------------
    def analyze(self, detections: list[dict], frame_shape: tuple) -> SafetyReport:
        """
        Run crowd-safety analysis on the current frame's person detections.

        Args:
            detections: list of YOLO detection dicts (all classes).
            frame_shape: (height, width, channels) of the frame.

        Returns:
            SafetyReport with grid density, alerts, and overall status.
        """
        report = SafetyReport()
        h, w = frame_shape[:2]
        cell_w = w / self.cols
        cell_h = h / self.rows

        # Build density grid from person centroids
        grid = [[0] * self.cols for _ in range(self.rows)]
        person_count = 0
        for det in detections:
            if det["class"] != "person":
                continue
            person_count += 1
            x1, y1, x2, y2 = det["box"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            col = min(int(cx / cell_w), self.cols - 1)
            row = min(int(cy / cell_h), self.rows - 1)
            grid[row][col] += 1

        report.grid_density = grid
        report.total_count = person_count

        now = time.time()
        self._history.append((now, person_count, grid))

        # --- Baseline from rolling window ---
        cutoff = now - config.CROWD_HISTORY_SECONDS
        window_counts = [c for t, c, _ in self._history if t >= cutoff]
        baseline = sum(window_counts) / len(window_counts) if window_counts else 0
        report.baseline_count = baseline

        alerts: list[SafetyAlert] = []

        # --- Choke points & gathering ---
        choke_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                count = grid[r][c]
                if count >= self.density_thresh:
                    choke_cells.append((c, r))
                    self._cell_dense_streak[r][c] += 1

                    if self._cell_dense_streak[r][c] == 1:
                        alerts.append(SafetyAlert(
                            "CHOKE_POINT", "WARNING",
                            f"High density in zone ({c},{r}): {count} people",
                            cell=(c, r)))

                    if self._cell_dense_streak[r][c] >= self.gathering_frames:
                        alerts.append(SafetyAlert(
                            "GATHERING", "CRITICAL",
                            f"Persistent crowd in zone ({c},{r}) for "
                            f"{self._cell_dense_streak[r][c]} cycles",
                            cell=(c, r)))
                else:
                    self._cell_dense_streak[r][c] = 0

        report.choke_cells = choke_cells

        # --- Surge / dispersal (need enough history) ---
        if len(window_counts) >= 3 and baseline > 0:
            pct_change = ((person_count - baseline) / baseline) * 100

            if pct_change >= self.surge_pct:
                alerts.append(SafetyAlert(
                    "CROWD_SURGE", "CRITICAL",
                    f"Crowd surge: {person_count} people "
                    f"(+{pct_change:.0f}% vs baseline {baseline:.0f})"))

            if pct_change <= -self.dispersal_pct:
                alerts.append(SafetyAlert(
                    "DISPERSAL", "WARNING",
                    f"Rapid dispersal: {person_count} people "
                    f"({pct_change:.0f}% vs baseline {baseline:.0f})"))

        report.alerts = alerts

        # Overall status
        if any(a.severity == "CRITICAL" for a in alerts):
            report.status = "CRITICAL"
        elif alerts:
            report.status = "WARNING"
        else:
            report.status = "NORMAL"

        return report

    # ------------------------------------------------------------------
    def draw_overlay(self, frame: np.ndarray, report: SafetyReport) -> np.ndarray:
        """Draw density heatmap grid and alert banner onto the frame."""
        h, w = frame.shape[:2]
        cell_w = w / self.cols
        cell_h = h / self.rows

        # Semi-transparent density heatmap
        overlay = frame.copy()
        for r in range(self.rows):
            for c in range(self.cols):
                count = report.grid_density[r][c]
                if count == 0:
                    continue
                color = _color_for_density(count)
                x1 = int(c * cell_w)
                y1 = int(r * cell_h)
                x2 = int((c + 1) * cell_w)
                y2 = int((r + 1) * cell_h)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Choke-point cell borders (solid red)
        for c, r in report.choke_cells:
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Density number inside the cell
            label = str(report.grid_density[r][c])
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(frame, label, (tx, ty), font, 0.6, (0, 0, 255), 2)

        # Alert banner (top-right)
        if report.alerts:
            top_alert = report.alerts[-1]
            if top_alert.severity == "CRITICAL":
                banner_color = (0, 0, 200)
                text_color = (255, 255, 255)
            else:
                banner_color = (0, 140, 255)
                text_color = (0, 0, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            msg = top_alert.message
            (tw, th), _ = cv2.getTextSize(msg, font, 0.45, 1)

            bx = w - tw - 24
            by = 8
            cv2.rectangle(frame, (bx, by), (bx + tw + 16, by + th + 14),
                          banner_color, -1)
            cv2.putText(frame, msg, (bx + 8, by + th + 6),
                        font, 0.45, text_color, 1)

        return frame
