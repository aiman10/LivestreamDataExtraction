"""
photo_detector.py
Detects when a person has stopped to take a photo using YOLOv8 pose estimation.

A person is considered to be taking a photo when ALL of the following hold
simultaneously for at least PHOTO_MIN_DURATION_SEC seconds:

  1. Stillness    — bounding-box centroid moves < PHOTO_STILL_PIXELS px between
                    consecutive pose frames over the rolling history window.
  2. Arm posture  — both wrists above their elbows AND both elbows above their
                    hips (classic phone-held-up stance).
  3. Head alignment — nose X is within PHOTO_HEAD_TOLERANCE px of the midpoint
                    between the two wrists (person is looking toward what they hold).

Confidence is the fraction of the rolling history window where all three
conditions held. The label fires when confidence >= PHOTO_DISPLAY_CONF.

Accepts the pre-loaded pose model from WaveDetector to avoid a second model
load; falls back to loading its own model if none is supplied.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from detection_logger import DetectionLogger

# COCO-17 keypoint indices used here
_NOSE    = 0
_L_ELBOW = 7
_R_ELBOW = 8
_L_WRIST = 9
_R_WRIST = 10
_L_HIP   = 11
_R_HIP   = 12


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhotoPoseSample:
    frame_idx:   int
    centroid:    tuple          # (cx, cy) of bounding box — for stillness
    nose:        tuple | None   # (x, y) if keypoint confidence passes threshold
    left_wrist:  tuple | None
    right_wrist: tuple | None
    left_elbow:  tuple | None
    right_elbow: tuple | None
    left_hip:    tuple | None
    right_hip:   tuple | None
    all_conds:   bool = False   # annotated during evaluation


@dataclass
class PhotoTrack:
    track_id:            int
    box:                 tuple           # (x1, y1, x2, y2) latest
    frames_missing:      int = 0
    history:             deque = field(
        default_factory=lambda: deque(maxlen=config.PHOTO_HISTORY_FRAMES)
    )
    is_taking_photo:     bool = False
    confidence:          float = 0.0
    _cond_start:         float | None = None  # epoch when conditions became continuously true
    photo_display_until: float = 0.0          # epoch — hold indicator until here
    _logged:             bool = False          # suppress duplicate log entries per event


# ---------------------------------------------------------------------------
# Module-level IoU helper (mirrors WaveDetector._iou)
# ---------------------------------------------------------------------------

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


# ---------------------------------------------------------------------------
# PhotoTakingDetector
# ---------------------------------------------------------------------------

class PhotoTakingDetector:
    """
    Detects photo-taking behaviour from pose keypoints.

    Public API:
        detect_photo_taking(frame, frame_idx) -> list[dict]
        draw_photo_indicators(display, results) -> np.ndarray
    """

    def __init__(self, pose_model=None):
        """
        pose_model — pass wave_detector._model to share weights and avoid a
                     second 26 MB load. If None, loads its own model.
        """
        if pose_model is not None:
            self._model = pose_model
        else:
            from ultralytics import YOLO
            self._model = YOLO(config.PHOTO_MODEL)

        self._tracks: dict[int, PhotoTrack] = {}
        self._next_id = 0
        self._logger = DetectionLogger()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect_photo_taking(self, frame: np.ndarray, frame_idx: int) -> list[dict]:
        """
        Run pose inference, update tracks, evaluate photo-taking conditions.

        Returns a list of dicts — one per tracked person:
            {
                "track_id":        int,
                "box":             (x1, y1, x2, y2),
                "is_taking_photo": bool,
                "confidence":      float   # 0-1
            }
        """
        raw = self._model(frame, conf=config.PHOTO_CONFIDENCE, verbose=False)
        raw0 = raw[0] if raw else None
        new_detections = self._parse_detections(raw0, frame_idx)
        self._update_tracks(new_detections)

        results = []
        for track in self._tracks.values():
            self._evaluate(track)

            # Log when a new photo-taking event is first confirmed
            if track.is_taking_photo and not track._logged:
                self._logger.log(
                    event="photo_taking",
                    frame_idx=frame_idx,
                    track_id=track.track_id,
                    confidence=track.confidence,
                    box=track.box,
                )
                track._logged = True
            elif not track.is_taking_photo:
                track._logged = False

            results.append({
                "track_id":        track.track_id,
                "box":             track.box,
                "is_taking_photo": track.is_taking_photo,
                "confidence":      track.confidence,
            })

        return results

    def draw_photo_indicators(self, display: np.ndarray, results: list[dict]) -> np.ndarray:
        """
        Draw photo-taking indicators (caller owns the frame, no .copy() here).
        Returns annotated frame.
        """
        any_photo = any(r["is_taking_photo"] for r in results)

        for r in results:
            if not r["is_taking_photo"]:
                continue

            x1, y1, x2, y2 = r["box"]
            conf = r["confidence"]

            # Magenta bounding box (distinct from wave cyan and YOLO green)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Badge label above box
            label = f"[CAM] PHOTO {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thickness = 0.52, 2
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            badge_y1 = max(0, y1 - th - baseline - 8)
            badge_x2 = min(display.shape[1], x1 + tw + 8)
            cv2.rectangle(display, (x1, badge_y1), (badge_x2, y1), (180, 0, 180), -1)
            cv2.putText(display, label,
                        (x1 + 4, y1 - baseline - 2),
                        font, scale, (255, 255, 255), thickness)

        if any_photo:
            self._draw_global_banner(display)

        return display

    def close(self) -> None:
        self._logger.close()

    # ------------------------------------------------------------------
    # Pose parsing
    # ------------------------------------------------------------------

    def _parse_detections(self, raw, frame_idx: int) -> list[dict]:
        if raw is None or raw.keypoints is None or raw.boxes is None:
            return []

        kp_xy   = raw.keypoints.xy.cpu().numpy()    # (N, 17, 2)
        kp_conf = raw.keypoints.conf.cpu().numpy()   # (N, 17)
        detections = []

        for i, box in enumerate(raw.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            sample = self._extract_keypoints(kp_xy[i], kp_conf[i], frame_idx, (cx, cy))
            detections.append({"box": (x1, y1, x2, y2), "sample": sample})

        return detections

    def _extract_keypoints(self, xy: np.ndarray, conf: np.ndarray,
                           frame_idx: int, centroid: tuple) -> PhotoPoseSample:
        def get(idx) -> tuple | None:
            if float(conf[idx]) >= config.PHOTO_KEYPOINT_CONF:
                return (float(xy[idx][0]), float(xy[idx][1]))
            return None

        return PhotoPoseSample(
            frame_idx=frame_idx,
            centroid=centroid,
            nose=get(_NOSE),
            left_wrist=get(_L_WRIST),  right_wrist=get(_R_WRIST),
            left_elbow=get(_L_ELBOW),  right_elbow=get(_R_ELBOW),
            left_hip=get(_L_HIP),      right_hip=get(_R_HIP),
        )

    # ------------------------------------------------------------------
    # IoU person tracker (same pattern as WaveDetector)
    # ------------------------------------------------------------------

    def _update_tracks(self, new_detections: list[dict]) -> None:
        for t in self._tracks.values():
            t.frames_missing += 1

        assigned: set[int] = set()

        for det in new_detections:
            best_id, best_iou = None, config.PHOTO_TRACKER_IOU
            for tid, track in self._tracks.items():
                v = _iou(det["box"], track.box)
                if v > best_iou:
                    best_iou = v
                    best_id = tid

            if best_id is not None and best_id not in assigned:
                t = self._tracks[best_id]
                t.box = det["box"]
                t.frames_missing = 0
                t.history.append(det["sample"])
                assigned.add(best_id)
            else:
                tid = self._next_id
                self._next_id += 1
                h: deque = deque(maxlen=config.PHOTO_HISTORY_FRAMES)
                h.append(det["sample"])
                self._tracks[tid] = PhotoTrack(track_id=tid, box=det["box"], history=h)

        expired = [tid for tid, t in self._tracks.items()
                   if t.frames_missing > config.PHOTO_TRACKER_MISSING]
        for tid in expired:
            del self._tracks[tid]

    # ------------------------------------------------------------------
    # Photo-taking evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, track: PhotoTrack) -> None:
        """Evaluate all three conditions and update track state."""
        now = time.time()
        history = list(track.history)

        if len(history) < 3:
            track.confidence = 0.0
            track._cond_start = None
            return

        # Stillness applies to the whole window (not per-sample)
        is_still = self._check_stillness(history)

        # Posture and alignment are evaluated per sample
        for s in history:
            s.all_conds = is_still and self._posture_ok(s) and self._alignment_ok(s)

        # Confidence = fraction of history where all conditions held
        track.confidence = sum(1 for s in history if s.all_conds) / len(history)

        # Duration gate: conditions must hold continuously
        all_conds_now = history[-1].all_conds
        if all_conds_now:
            if track._cond_start is None:
                track._cond_start = now
            elapsed = now - track._cond_start
            if (elapsed >= config.PHOTO_MIN_DURATION_SEC
                    and track.confidence >= config.PHOTO_DISPLAY_CONF):
                track.photo_display_until = now + config.PHOTO_HOLD_SECONDS
        else:
            track._cond_start = None  # reset — must be continuous

        track.is_taking_photo = now < track.photo_display_until

    # ------------------------------------------------------------------
    # Condition checks
    # ------------------------------------------------------------------

    def _check_stillness(self, history: list[PhotoPoseSample]) -> bool:
        """
        True when every consecutive centroid delta is below PHOTO_STILL_PIXELS.
        Reuses the centroid-delta logic pattern from WaveDetector's tracker.
        """
        centroids = [s.centroid for s in history]
        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            if (dx * dx + dy * dy) ** 0.5 > config.PHOTO_STILL_PIXELS:
                return False
        return True

    @staticmethod
    def _posture_ok(s: PhotoPoseSample) -> bool:
        """
        Both wrists above their elbows AND both elbows above their hips.
        All four arm keypoints must be visible. Hip check is skipped if
        hips are not detected (reduces false negatives for cropped persons).
        """
        if not (s.left_wrist and s.right_wrist
                and s.left_elbow and s.right_elbow):
            return False
        # Smaller Y = higher in image
        if not (s.left_wrist[1]  < s.left_elbow[1] and
                s.right_wrist[1] < s.right_elbow[1]):
            return False
        if s.left_hip and s.right_hip:
            if not (s.left_elbow[1]  < s.left_hip[1] and
                    s.right_elbow[1] < s.right_hip[1]):
                return False
        return True

    @staticmethod
    def _alignment_ok(s: PhotoPoseSample) -> bool:
        """
        Nose X within PHOTO_HEAD_TOLERANCE of the midpoint between both wrists.
        Requires nose and both wrists to be visible.
        """
        if not (s.nose and s.left_wrist and s.right_wrist):
            return False
        wrist_mid_x = (s.left_wrist[0] + s.right_wrist[0]) / 2.0
        return abs(s.nose[0] - wrist_mid_x) < config.PHOTO_HEAD_TOLERANCE

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_global_banner(display: np.ndarray) -> None:
        """Semi-transparent magenta banner positioned below the wave banner."""
        h, w = display.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "PHOTO DETECTED"
        scale, thickness = 0.65, 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

        pad = 10
        bx1 = w - tw - pad * 2
        by1 = 57          # sits just below the WAVE DETECTED banner (~53 px tall)
        bx2 = w - 8
        by2 = by1 + th + baseline + pad * 2

        overlay = display.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (160, 0, 160), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        cv2.putText(display, text,
                    (bx1 + pad, by2 - pad - baseline),
                    font, scale, (255, 255, 255), thickness)
