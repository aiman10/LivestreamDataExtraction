"""
wave_detector.py
Detects when a person waves at the camera using YOLOv8 pose estimation.

A wave is defined as:
  1. A wrist raised above shoulder level (with elbow also raised)
  2. The raised wrist oscillates horizontally with sufficient amplitude and frequency

Uses an IoU-based person tracker to maintain per-person wrist history across frames.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

import config
from detection_logger import DetectionLogger

# COCO-17 keypoint indices
_L_SHOULDER = 5
_R_SHOULDER = 6
_L_ELBOW    = 7
_R_ELBOW    = 8
_L_WRIST    = 9
_R_WRIST    = 10
_L_HIP      = 11
_R_HIP      = 12


@dataclass
class WristSample:
    frame_idx:      int
    left_wrist:     tuple | None    # (x, y) pixels; None if confidence too low
    right_wrist:    tuple | None
    left_conf:      float
    right_conf:     float
    left_elbow:     tuple | None
    right_elbow:    tuple | None
    shoulder_mid_y: float           # Y of shoulder midpoint (raise reference)
    hip_mid_y:      float | None    # Y of hip midpoint (body height normalisation)


@dataclass
class PersonTrack:
    track_id:           int
    box:                tuple                   # (x1, y1, x2, y2) in source pixels
    frames_missing:     int = 0
    wrist_history:      deque = field(default_factory=lambda: deque(maxlen=config.WAVE_HISTORY_LEN if hasattr(config, "WAVE_HISTORY_LEN") else 20))
    is_waving:          bool = False
    _confirm_count:     int = 0                 # consecutive frames conditions met
    wave_display_until: float = 0.0             # epoch time — hold indicator until here
    _logged:            bool = False            # suppress duplicate log entries per wave event


class WaveDetector:
    """
    Runs YOLOv8 pose inference and detects waving persons.

    Public API:
        detect_waves(frame, frame_idx) -> list[dict]
        draw_wave_indicators(display, wave_results) -> np.ndarray
    """

    _HISTORY_LEN = 20   # wrist deque length (pose inference frames)

    def __init__(self):
        self._model = YOLO(config.WAVE_MODEL)
        self._tracks: dict[int, PersonTrack] = {}
        self._next_id = 0
        self._logger = DetectionLogger(
            csv_file=config.WAVE_LOG_CSV,
            json_file=config.WAVE_LOG_JSON,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def detect_waves(self, frame: np.ndarray, frame_idx: int) -> list[dict]:
        """
        Run pose inference on frame, update tracks, evaluate wave conditions.

        Returns a list of dicts — one per tracked person:
            {
                "track_id":    int,
                "box":         (x1, y1, x2, y2),
                "is_waving":   bool,
                "wrist_raised": "left" | "right" | None,
                "keypoints":   dict   # visible keypoints for drawing
            }
        """
        raw = self._run_pose(frame)
        new_detections = self._parse_detections(raw, frame_idx)
        self._update_tracks(new_detections, frame_idx)

        results = []
        for track in self._tracks.values():
            waving, side = self._evaluate_wave(track)
            track.is_waving = waving

            # Log the start of each new wave event (suppress duplicates)
            if track.is_waving and not track._logged:
                self._logger.log(
                    event="waving",
                    frame_idx=frame_idx,
                    track_id=track.track_id,
                    confidence=1.0,
                    box=track.box,
                )
                track._logged = True
            elif not track.is_waving:
                track._logged = False

            # Collect visible keypoints for skeleton drawing
            kps = {}
            if track.wrist_history:
                last = track.wrist_history[-1]
                for name, val in [
                    ("left_wrist",  last.left_wrist),
                    ("right_wrist", last.right_wrist),
                    ("left_elbow",  last.left_elbow),
                    ("right_elbow", last.right_elbow),
                ]:
                    if val is not None:
                        kps[name] = val
                kps["shoulder_mid_y"] = last.shoulder_mid_y

            results.append({
                "track_id":    track.track_id,
                "box":         track.box,
                "is_waving":   track.is_waving,
                "wrist_raised": side,
                "keypoints":   kps,
            })

        return results

    def close(self) -> None:
        self._logger.close()

    def draw_wave_indicators(self, display: np.ndarray, wave_results: list[dict]) -> np.ndarray:
        """
        Draw wave indicators onto display (caller owns the frame — no .copy() here).
        Returns the annotated frame.
        """
        any_waving = any(r["is_waving"] for r in wave_results)

        for r in wave_results:
            x1, y1, x2, y2 = r["box"]

            if config.WAVE_SHOW_SKELETON:
                self._draw_skeleton(display, r["keypoints"], r["is_waving"])

            if r["is_waving"]:
                # Thick cyan bounding box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)

                # Yellow "WAVING!" badge above box
                label = "WAVING!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = 0.55, 2
                (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
                badge_y1 = max(0, y1 - th - baseline - 8)
                badge_y2 = y1
                badge_x2 = min(display.shape[1], x1 + tw + 8)
                cv2.rectangle(display, (x1, badge_y1), (badge_x2, badge_y2), (0, 220, 220), -1)
                cv2.putText(display, label, (x1 + 4, badge_y2 - baseline - 2),
                            font, scale, (0, 0, 0), thickness)

        if any_waving:
            self._draw_global_banner(display)

        return display

    # ------------------------------------------------------------------
    # Pose inference
    # ------------------------------------------------------------------

    def _run_pose(self, frame: np.ndarray):
        results = self._model(frame, conf=config.WAVE_CONFIDENCE, verbose=False)
        return results[0] if results else None

    def _parse_detections(self, raw, frame_idx: int) -> list[dict]:
        """Convert raw YOLO pose results into a list of per-person dicts."""
        detections = []
        if raw is None:
            return detections

        boxes = raw.boxes
        kp_xy  = raw.keypoints.xy.cpu().numpy()   # (N, 17, 2)
        kp_conf = raw.keypoints.conf.cpu().numpy() # (N, 17)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            sample = self._extract_keypoints(kp_xy[i], kp_conf[i], frame_idx)
            detections.append({
                "box":    (x1, y1, x2, y2),
                "sample": sample,
            })

        return detections

    def _extract_keypoints(self, xy: np.ndarray, conf: np.ndarray, frame_idx: int) -> WristSample:
        """Build a WristSample from one person's keypoint arrays."""

        def get(idx):
            """Return (x, y) if confidence passes threshold, else None."""
            c = float(conf[idx])
            if c >= config.WAVE_KEYPOINT_CONF:
                return (float(xy[idx][0]), float(xy[idx][1])), c
            return None, c

        lw, lw_c  = get(_L_WRIST)
        rw, rw_c  = get(_R_WRIST)
        le, _     = get(_L_ELBOW)
        re, _     = get(_R_ELBOW)
        ls, ls_c  = get(_L_SHOULDER)
        rs, rs_c  = get(_R_SHOULDER)
        lh, _     = get(_L_HIP)
        rh, _     = get(_R_HIP)

        # Shoulder midpoint Y (reference for raise condition)
        if ls and rs:
            shoulder_mid_y = (ls[1] + rs[1]) / 2.0
        elif ls:
            shoulder_mid_y = ls[1]
        elif rs:
            shoulder_mid_y = rs[1]
        else:
            shoulder_mid_y = float("inf")  # unknowable — raise condition will fail

        # Hip midpoint Y (for body-height normalisation)
        if lh and rh:
            hip_mid_y = (lh[1] + rh[1]) / 2.0
        elif lh:
            hip_mid_y = lh[1]
        elif rh:
            hip_mid_y = rh[1]
        else:
            hip_mid_y = None

        return WristSample(
            frame_idx=frame_idx,
            left_wrist=lw,   right_wrist=rw,
            left_conf=lw_c,  right_conf=rw_c,
            left_elbow=le,   right_elbow=re,
            shoulder_mid_y=shoulder_mid_y,
            hip_mid_y=hip_mid_y,
        )

    # ------------------------------------------------------------------
    # IoU person tracker
    # ------------------------------------------------------------------

    def _update_tracks(self, new_detections: list[dict], frame_idx: int):
        """Greedy IoU assignment. Creates/updates/expires PersonTrack objects."""
        # Increment missing counter for all existing tracks
        for t in self._tracks.values():
            t.frames_missing += 1

        assigned_track_ids = set()

        for det in new_detections:
            best_id, best_iou = None, config.WAVE_TRACKER_IOU
            for tid, track in self._tracks.items():
                iou = self._iou(det["box"], track.box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None and best_id not in assigned_track_ids:
                # Update existing track
                t = self._tracks[best_id]
                t.box = det["box"]
                t.frames_missing = 0
                t.wrist_history.append(det["sample"])
                assigned_track_ids.add(best_id)
            else:
                # Spawn new track
                tid = self._next_id
                self._next_id += 1
                h = deque(maxlen=self._HISTORY_LEN)
                h.append(det["sample"])
                self._tracks[tid] = PersonTrack(
                    track_id=tid,
                    box=det["box"],
                    wrist_history=h,
                )

        # Expire tracks that have been missing too long
        expired = [tid for tid, t in self._tracks.items()
                   if t.frames_missing > config.WAVE_TRACKER_MISSING]
        for tid in expired:
            del self._tracks[tid]

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    # ------------------------------------------------------------------
    # Wave evaluation
    # ------------------------------------------------------------------

    def _evaluate_wave(self, track: PersonTrack) -> tuple[bool, str | None]:
        """
        Check raise + oscillation conditions. Apply temporal hysteresis.
        Returns (is_waving, side) where side is "left", "right", or None.
        """
        now = time.time()

        # Collect raised-wrist X positions from history
        left_xs, right_xs = [], []
        for s in track.wrist_history:
            l_raised, l_x = self._check_raise(s, "left")
            r_raised, r_x = self._check_raise(s, "right")
            if l_raised and l_x is not None:
                left_xs.append(l_x)
            if r_raised and r_x is not None:
                right_xs.append(r_x)

        # Pick the arm with more raised samples
        if len(left_xs) >= len(right_xs):
            xs, side = left_xs, "left"
        else:
            xs, side = right_xs, "right"

        conditions_met = self._oscillation_ok(xs)
        side_result = side if conditions_met else None

        if conditions_met:
            track._confirm_count += 1
        else:
            track._confirm_count = 0

        if track._confirm_count >= config.WAVE_CONFIRM_FRAMES:
            track.wave_display_until = now + config.WAVE_HOLD_SECONDS

        waving = now < track.wave_display_until
        return waving, (side_result if waving else None)

    def _check_raise(self, s: WristSample, side: str) -> tuple[bool, float | None]:
        """
        Return (raised, wrist_x) for the given arm side.
        raised is True when wrist AND elbow are above the shoulder threshold.
        """
        if side == "left":
            wrist  = s.left_wrist
            elbow  = s.left_elbow
        else:
            wrist  = s.right_wrist
            elbow  = s.right_elbow

        if wrist is None:
            return False, None

        # Body-height-normalised raise threshold
        if s.hip_mid_y is not None and s.shoulder_mid_y != float("inf"):
            body_height = abs(s.hip_mid_y - s.shoulder_mid_y)
            raise_thresh = body_height * config.WAVE_RAISE_FRACTION
            elbow_slack  = body_height * 0.05
        else:
            raise_thresh = 30.0
            elbow_slack  = 5.0

        # Wrist must be above shoulder by raise_thresh (smaller Y = higher on screen)
        hand_raised = wrist[1] < s.shoulder_mid_y - raise_thresh

        # Elbow must also be at or above shoulder (with slack)
        if elbow is not None:
            elbow_raised = elbow[1] < s.shoulder_mid_y + elbow_slack
        else:
            elbow_raised = hand_raised  # can't check — defer to wrist alone

        if hand_raised and elbow_raised:
            return True, wrist[0]
        return False, None

    def _oscillation_ok(self, xs: list[float]) -> bool:
        """Return True when xs shows sufficient horizontal wave oscillation."""
        if len(xs) < config.WAVE_MIN_SAMPLES:
            return False
        if max(xs) - min(xs) < config.WAVE_MIN_AMPLITUDE:
            return False
        smoothed = self._smooth(xs)
        reversals = self._count_reversals(smoothed)
        return reversals >= config.WAVE_MIN_REVERSALS

    @staticmethod
    def _smooth(xs: list[float], window: int = 3) -> list[float]:
        """Simple centred moving average."""
        result = []
        half = window // 2
        for i in range(len(xs)):
            lo = max(0, i - half)
            hi = min(len(xs), i + half + 1)
            result.append(sum(xs[lo:hi]) / (hi - lo))
        return result

    @staticmethod
    def _count_reversals(xs: list[float]) -> int:
        """Count sign changes in the first differences (direction reversals)."""
        if len(xs) < 2:
            return 0
        diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        # Remove near-zero diffs to avoid counting jitter as reversals
        threshold = 1.0
        filtered = [d for d in diffs if abs(d) > threshold]
        if len(filtered) < 2:
            return 0
        count = 0
        for i in range(len(filtered) - 1):
            if filtered[i] * filtered[i + 1] < 0:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_skeleton(self, display: np.ndarray, kps: dict, is_waving: bool):
        """Draw shoulder→elbow→wrist arm lines."""
        color = (0, 255, 255) if is_waving else (200, 200, 200)
        thickness = 2 if is_waving else 1

        shoulder_y = kps.get("shoulder_mid_y")

        for side in ("left", "right"):
            elbow = kps.get(f"{side}_elbow")
            wrist = kps.get(f"{side}_wrist")

            # Draw elbow→wrist
            if elbow and wrist:
                cv2.line(display,
                         (int(elbow[0]), int(elbow[1])),
                         (int(wrist[0]), int(wrist[1])),
                         color, thickness)

    @staticmethod
    def _draw_global_banner(display: np.ndarray):
        """Semi-transparent red banner in top-right: 'WAVE DETECTED'."""
        h, w = display.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "WAVE DETECTED"
        scale, thickness = 0.65, 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        pad = 10
        bx1 = w - tw - pad * 2
        by1 = 8
        bx2 = w - 8
        by2 = by1 + th + baseline + pad * 2

        overlay = display.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        cv2.putText(display, text,
                    (bx1 + pad, by2 - pad - baseline),
                    font, scale, (255, 255, 255), thickness)
