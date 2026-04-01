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
        Run detection on a frame.

        Returns a list of dicts, each with:
            class:      str, e.g. "person"
            confidence: float, 0-1
            box:        (x1, y1, x2, y2) pixel coordinates
        """
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
            })

        return detections

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
            color = COLORS.get(cls, COLORS["default"])

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

        person_count = counts.get("person", 0)
        vehicle_count = sum(counts.get(v, 0) for v in ["car", "bus", "truck", "motorcycle"])
        bicycle_count = counts.get("bicycle", 0)
        umbrella_count = counts.get("umbrella", 0)
        backpack_count = counts.get("backpack", 0)

        return {
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "bicycle_count": bicycle_count,
            "umbrella_count": umbrella_count,
            "backpack_count": backpack_count,
            "total_objects": len(detections),
        }
