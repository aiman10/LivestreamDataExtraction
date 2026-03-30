"""
engagement_detector.py
Friendliness & Engagement detection layer.

Three components:
  1. WaveDetector: MediaPipe Pose-based wave gesture detection
  2. PhotoStopDetector: Stationary person tracking for photo-taking behavior
  3. FriendlinessIndex: Rolling engagement score from wave and photo-stop counts
"""

import math
import time
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np

import config

# MediaPipe landmark indices for pose estimation
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_WRIST = 15
_RIGHT_WRIST = 16


class WaveDetector:
    """
    Detects waving gestures by running MediaPipe Pose on cropped
    YOLO person bounding boxes.

    Algorithm:
      1. Filter person bboxes by minimum size, take top N by area.
      2. Assign temporary person IDs via nearest-centroid matching.
      3. Crop each person region, run MediaPipe Pose.
      4. Check if either wrist is raised above the shoulder.
      5. Track lateral (x) wrist position over a sliding window.
      6. Count direction changes - if enough reversals, classify as wave.
      7. Apply per-person cooldown to avoid double-counting.
    """

    def __init__(self):
        import mediapipe as mp

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Per-person wrist x-position history (person_id -> deque of x values)
        self.wrist_history: dict[int, deque] = {}
        # Per-person cooldown timestamps (person_id -> last wave time)
        self.cooldowns: dict[int, float] = {}
        # Centroid tracker state
        self._prev_centroids: dict[int, tuple] = {}
        self._next_id: int = 0

        self.total_waves: int = 0

    def detect(self, frame, person_detections: list, timestamp: datetime) -> dict:
        """
        Run wave detection on person crops from the current frame.

        Args:
            frame: Full BGR image
            person_detections: List of detection dicts with "bbox": (x1, y1, x2, y2)
            timestamp: Current UTC datetime

        Returns:
            dict with wave_count (this call) and total_waves (cumulative)
        """
        now_mono = time.monotonic()
        wave_count = 0

        # Filter by minimum box size and sort by area descending
        valid_boxes = []
        for det in person_detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            if w >= config.WAVE_MIN_BOX_WIDTH and h >= config.WAVE_MIN_BOX_HEIGHT:
                valid_boxes.append(det)

        valid_boxes.sort(
            key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
            reverse=True,
        )
        valid_boxes = valid_boxes[:config.WAVE_MAX_PERSONS]

        # Assign person IDs via centroid matching
        person_ids = self._match_centroids(valid_boxes)

        # Track which IDs are active this frame
        active_ids = set()

        for det, pid in zip(valid_boxes, person_ids):
            active_ids.add(pid)
            x1, y1, x2, y2 = det["bbox"]

            # Crop person region from frame
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self.pose.process(crop_rgb)

            if not results.pose_landmarks:
                # No pose detected, clear history for this person
                self.wrist_history.pop(pid, None)
                continue

            landmarks = results.pose_landmarks.landmark

            # Check both sides for raised wrist
            raised_wrist_x = None
            threshold = config.WAVE_WRIST_ABOVE_SHOULDER_THRESHOLD

            # Left side: wrist above shoulder means wrist.y < shoulder.y
            left_shoulder_y = landmarks[_LEFT_SHOULDER].y
            left_wrist_y = landmarks[_LEFT_WRIST].y
            left_wrist_x = landmarks[_LEFT_WRIST].x

            right_shoulder_y = landmarks[_RIGHT_SHOULDER].y
            right_wrist_y = landmarks[_RIGHT_WRIST].y
            right_wrist_x = landmarks[_RIGHT_WRIST].x

            if left_shoulder_y - left_wrist_y > threshold:
                raised_wrist_x = left_wrist_x
            elif right_shoulder_y - right_wrist_y > threshold:
                raised_wrist_x = right_wrist_x

            if raised_wrist_x is not None:
                # Wrist is raised, track lateral movement
                if pid not in self.wrist_history:
                    self.wrist_history[pid] = deque(
                        maxlen=config.WAVE_OSCILLATION_WINDOW
                    )
                self.wrist_history[pid].append(raised_wrist_x)

                # Check for oscillation once we have enough samples
                history = self.wrist_history[pid]
                if len(history) >= config.WAVE_OSCILLATION_WINDOW:
                    direction_changes = self._count_direction_changes(history)
                    amplitude = max(history) - min(history)

                    if (direction_changes >= config.WAVE_MIN_DIRECTION_CHANGES
                            and amplitude >= config.WAVE_LATERAL_MOVEMENT_THRESHOLD):
                        # Check cooldown
                        last_wave = self.cooldowns.get(pid, 0)
                        if now_mono - last_wave >= config.WAVE_COOLDOWN_SECONDS:
                            wave_count += 1
                            self.total_waves += 1
                            self.cooldowns[pid] = now_mono
                            # Clear history after detection
                            self.wrist_history[pid].clear()
            else:
                # Wrist not raised, clear history
                self.wrist_history.pop(pid, None)

        # Clean up history and cooldowns for persons no longer visible
        stale_ids = set(self.wrist_history.keys()) - active_ids
        for sid in stale_ids:
            self.wrist_history.pop(sid, None)

        return {
            "wave_count": wave_count,
            "total_waves": self.total_waves,
        }

    def _match_centroids(self, person_boxes: list) -> list:
        """
        Assign temporary person IDs by matching current box centroids
        to previously known centroids using nearest-neighbor.

        Args:
            person_boxes: List of detection dicts with "bbox" key

        Returns:
            List of int person IDs, one per input box
        """
        current_centroids = []
        for det in person_boxes:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current_centroids.append((cx, cy))

        assigned_ids = []
        used_prev_ids = set()

        for cx, cy in current_centroids:
            best_id = None
            best_dist = float("inf")

            for pid, (px, py) in self._prev_centroids.items():
                if pid in used_prev_ids:
                    continue
                dist = math.hypot(cx - px, cy - py)
                if dist < best_dist and dist < config.PHOTO_STOP_MATCH_DISTANCE:
                    best_dist = dist
                    best_id = pid

            if best_id is not None:
                assigned_ids.append(best_id)
                used_prev_ids.add(best_id)
            else:
                assigned_ids.append(self._next_id)
                self._next_id += 1

        # Update stored centroids for next call
        self._prev_centroids = {}
        for pid, (cx, cy) in zip(assigned_ids, current_centroids):
            self._prev_centroids[pid] = (cx, cy)

        return assigned_ids

    @staticmethod
    def _count_direction_changes(positions: deque) -> int:
        """
        Count sign changes in consecutive differences of positions.

        Args:
            positions: Sequence of x-coordinate values

        Returns:
            Number of direction reversals
        """
        if len(positions) < 3:
            return 0

        pos_list = list(positions)
        changes = 0
        prev_diff = pos_list[1] - pos_list[0]

        for i in range(2, len(pos_list)):
            curr_diff = pos_list[i] - pos_list[i - 1]
            if prev_diff != 0 and curr_diff != 0:
                if (prev_diff > 0) != (curr_diff > 0):
                    changes += 1
            if curr_diff != 0:
                prev_diff = curr_diff

        return changes


class PhotoStopDetector:
    """
    Detects people who stop to take photos or engage with the camera view.

    Tracks person bounding box centroids across frames. A person is
    considered "stopped for a photo" when their centroid barely moves
    for several consecutive frames.
    """

    def __init__(self):
        # Tracked persons: id -> {centroid, stationary_count, last_stop_time}
        self.tracked: dict[int, dict] = {}
        self._next_id: int = 0
        self.total_photo_stops: int = 0

    def detect(self, person_detections: list, timestamp: datetime) -> dict:
        """
        Track person centroids and detect stationary behavior.

        Args:
            person_detections: List of detection dicts with "bbox": (x1, y1, x2, y2)
            timestamp: Current UTC datetime

        Returns:
            dict with photo_stop_count (currently stopped) and total_photo_stops (cumulative)
        """
        now_mono = time.monotonic()
        photo_stop_count = 0

        # Compute current centroids
        current_centroids = []
        for det in person_detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current_centroids.append((cx, cy))

        # Match to existing tracked persons
        matched_current = set()
        matched_tracked = set()
        assignments = []  # (current_idx, tracked_id)

        for i, (cx, cy) in enumerate(current_centroids):
            best_id = None
            best_dist = float("inf")

            for tid, tdata in self.tracked.items():
                if tid in matched_tracked:
                    continue
                px, py = tdata["centroid"]
                dist = math.hypot(cx - px, cy - py)
                if dist < best_dist and dist < config.PHOTO_STOP_MATCH_DISTANCE:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                assignments.append((i, best_id, best_dist))
                matched_current.add(i)
                matched_tracked.add(best_id)

        # Update matched tracks
        for idx, tid, dist in assignments:
            cx, cy = current_centroids[idx]
            track = self.tracked[tid]

            if dist < config.PHOTO_STOP_MOVEMENT_THRESHOLD:
                track["stationary_count"] += 1
            else:
                track["stationary_count"] = 0

            track["centroid"] = (cx, cy)

            # Check if person has been stationary long enough
            if track["stationary_count"] >= config.PHOTO_STOP_STATIONARY_FRAMES:
                last_stop = track.get("last_stop_time", 0)
                if now_mono - last_stop >= config.WAVE_COOLDOWN_SECONDS:
                    photo_stop_count += 1
                    self.total_photo_stops += 1
                    track["last_stop_time"] = now_mono
                    track["stationary_count"] = 0

        # Remove unmatched old tracks
        stale_ids = set(self.tracked.keys()) - matched_tracked
        for sid in stale_ids:
            del self.tracked[sid]

        # Add new tracks for unmatched current detections
        for i, (cx, cy) in enumerate(current_centroids):
            if i not in matched_current:
                self.tracked[self._next_id] = {
                    "centroid": (cx, cy),
                    "stationary_count": 0,
                    "last_stop_time": 0,
                }
                self._next_id += 1

        return {
            "photo_stop_count": photo_stop_count,
            "total_photo_stops": self.total_photo_stops,
        }


class FriendlinessIndex:
    """
    Computes a rolling Friendliness & Engagement score from wave
    and photo-stop events over a configurable time window.

    Score formula:
      wave_component = min(wave_count / MAX_WAVES, 1.0) * 100
      stop_component = min(stop_count / MAX_STOPS, 1.0) * 100
      score = wave_component * WAVE_WEIGHT + stop_component * STOP_WEIGHT

    Classification:
      >= 70: "very friendly"
      >= 40: "friendly"
      >= 15: "neutral"
      <  15: "quiet"
    """

    def __init__(self):
        self.wave_events: deque = deque()   # (datetime, count) tuples
        self.stop_events: deque = deque()   # (datetime, count) tuples
        self.window = timedelta(minutes=config.FRIENDLINESS_WINDOW_MINUTES)

    def record_waves(self, timestamp: datetime, count: int) -> None:
        """Record wave detections at a given timestamp."""
        if count > 0:
            self.wave_events.append((timestamp, count))

    def record_stops(self, timestamp: datetime, count: int) -> None:
        """Record photo-stop detections at a given timestamp."""
        if count > 0:
            self.stop_events.append((timestamp, count))

    def compute(self, now: datetime) -> dict:
        """
        Compute the current friendliness index.

        Args:
            now: Current UTC datetime

        Returns:
            dict with friendliness_index (float 0-100) and
            friendliness_level (string classification)
        """
        cutoff = now - self.window

        # Purge old entries
        while self.wave_events and self.wave_events[0][0] < cutoff:
            self.wave_events.popleft()
        while self.stop_events and self.stop_events[0][0] < cutoff:
            self.stop_events.popleft()

        # Sum counts in window
        wave_sum = sum(count for _, count in self.wave_events)
        stop_sum = sum(count for _, count in self.stop_events)

        # Compute normalized components
        wave_component = min(wave_sum / max(config.FRIENDLINESS_MAX_WAVES, 1), 1.0) * 100
        stop_component = min(stop_sum / max(config.FRIENDLINESS_MAX_STOPS, 1), 1.0) * 100

        # Weighted combined score
        score = (
            wave_component * config.FRIENDLINESS_WAVE_WEIGHT
            + stop_component * config.FRIENDLINESS_STOP_WEIGHT
        )
        score = round(min(score, 100.0), 1)

        # Classify
        if score >= 70:
            level = "very friendly"
        elif score >= 40:
            level = "friendly"
        elif score >= 15:
            level = "neutral"
        else:
            level = "quiet"

        return {
            "friendliness_index": score,
            "friendliness_level": level,
        }
