"""
object_detector.py
YOLOv8 wrapper for object detection and visualization.
Handles model loading, inference, and drawing bounding boxes with labels.
"""

import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

import config

# Colors for different object categories (BGR)
COLORS = {
    "person":     (0, 255, 0),     # Green
    "car":        (255, 128, 0),   # Blue-ish
    "bus":        (255, 0, 128),   # Purple
    "truck":      (200, 100, 0),   # Dark blue
    "motorcycle": (0, 200, 200),   # Yellow
    "bicycle":    (0, 255, 255),   # Yellow
    "umbrella":   (255, 0, 0),     # Blue (rain indicator!)
    "backpack":   (128, 0, 255),   # Pink
    "suitcase":   (0, 128, 255),   # Orange
    "dog":        (128, 255, 0),   # Light green
    "default":    (200, 200, 200), # Gray
}


class ObjectDetector:
    """
    YOLOv8 object detector.

    Usage:
        detector = ObjectDetector()
        detections = detector.detect(frame)
        annotated = detector.draw(frame, detections)
    """

    def __init__(
        self,
        model_name: str = None,
        confidence: float = None,
    ):
        self.confidence = confidence or config.YOLO_CONFIDENCE
        model_path = model_name or config.YOLO_MODEL

        print(f"[Detector] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print(f"[Detector] Model ready. Classes: {len(self.model.names)}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a frame with an additional background-person pass.

        Returns a list of dicts, each with:
            class:      str, e.g. "person"
            confidence: float, 0-1
            box:        (x1, y1, x2, y2) pixel coordinates
            is_background: bool, True for small/background person detections
        """
        frame_h, frame_w = frame.shape[:2]

        # --- Primary pass (full frame, normal confidence) ---
        results = self.model(frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes

        detections = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = self.model.names[cls_id]
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            detections.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "is_background": False,
            })

        # --- Background pass (top portion, relaxed confidence for persons) ---
        zone_h = int(frame_h * config.BACKGROUND_ZONE_RATIO)
        crop = frame[:zone_h, :]

        bg_results = self.model(
            crop,
            conf=config.SMALL_PERSON_CONF_THRESHOLD,
            verbose=False,
        )
        bg_boxes = bg_results[0].boxes

        height_threshold = frame_h * config.SMALL_PERSON_HEIGHT_RATIO

        bg_detections = []
        for i in range(len(bg_boxes)):
            cls_id = int(bg_boxes.cls[i])
            cls_name = self.model.names[cls_id]
            if cls_name != "person":
                continue

            conf = float(bg_boxes.conf[i])
            x1, y1, x2, y2 = bg_boxes.xyxy[i].tolist()
            box_height = y2 - y1

            # Only keep small detections from this pass
            if box_height >= height_threshold:
                continue

            bg_detections.append({
                "class": cls_name,
                "confidence": round(conf, 2),
                "box": (int(x1), int(y1), int(x2), int(y2)),
                "is_background": True,
            })

        # --- Deduplicate (suppress bg detections that overlap primary ones) ---
        merged = detections.copy()
        for bg_det in bg_detections:
            duplicate = False
            for det in detections:
                if det["class"] != "person":
                    continue
                if self._iou(bg_det["box"], det["box"]) > 0.3:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(bg_det)

        return merged

    @staticmethod
    def _iou(box_a, box_b) -> float:
        """Compute Intersection-over-Union between two (x1,y1,x2,y2) boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        Returns a copy with annotations.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls = det["class"]
            conf = det["confidence"]
            is_bg = det.get("is_background", False)
            color = COLORS.get(cls, COLORS["default"])

            if is_bg:
                # Small red filled dot at bottom-center of bounding box
                cx = (x1 + x2) // 2
                cv2.circle(annotated, (cx, y2), 6, (0, 0, 255), -1)
                continue

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background + text
            if config.SHOW_LABELS:
                label = f"{cls} {conf:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

                # Label sits above the box
                label_y = max(y1 - 6, th + 4)
                cv2.rectangle(
                    annotated,
                    (x1, label_y - th - 4),
                    (x1 + tw + 4, label_y + 2),
                    color, -1,
                )
                cv2.putText(
                    annotated, label,
                    (x1 + 2, label_y - 2),
                    font, font_scale, (0, 0, 0), thickness,
                )

        return annotated

    def summarize(self, detections: list[dict]) -> dict:
        """
        Summarize detections into counts by category.

        Returns:
            person_count, vehicle_count, bicycle_count,
            umbrella_count, backpack_count, total_objects
        """
        counts = Counter(d["class"] for d in detections)
        bg_person_count = sum(
            1 for d in detections
            if d["class"] == "person" and d.get("is_background", False)
        )

        person_count = counts.get("person", 0)
        vehicle_count = sum(counts.get(v, 0) for v in ["car", "bus", "truck", "motorcycle"])
        bicycle_count = counts.get("bicycle", 0)
        umbrella_count = counts.get("umbrella", 0)
        backpack_count = counts.get("backpack", 0)

        return {
            "person_count": person_count,
            "background_person_count": bg_person_count,
            "vehicle_count": vehicle_count,
            "bicycle_count": bicycle_count,
            "umbrella_count": umbrella_count,
            "backpack_count": backpack_count,
            "total_objects": len(detections),
        }
