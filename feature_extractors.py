"""
feature_extractors.py
All computer vision analysis functions, organized into layers by cost:
  - Layer 1: Scene metrics (pure NumPy, very cheap)
  - Layer 2: Motion metrics (background subtraction, cheap)
  - Layer 3: Optical flow (medium cost)
  - Layer 4: Object detection with YOLO (expensive)
"""

import cv2
import numpy as np
from collections import Counter


# ============================================================
# LAYER 1: Scene-level metrics (pure OpenCV/NumPy, no ML)
# ============================================================

def extract_scene_metrics(frame):
    """
    Extract environmental/weather data from a single frame.
    
    A camera is effectively a multi-modal sensor: light meter,
    color temperature gauge, and seasonal vegetation tracker
    all in one.
    
    Args:
        frame: BGR image (numpy array)
    
    Returns:
        dict with brightness, contrast, saturation, color_temp,
        is_daytime, green_ratio
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    avg_b, avg_g, avg_r = np.mean(frame, axis=(0, 1))
    total_rgb = avg_r + avg_g + avg_b

    brightness = float(np.mean(v))
    contrast = float(np.std(v))

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "saturation": round(float(np.mean(s)), 2),
        "color_temp": round(float(avg_r - avg_b), 2),
        "is_daytime": brightness > 110,
        "green_ratio": round(float(avg_g / total_rgb), 4) if total_rgb > 0 else 0.0,
    }


# ============================================================
# LAYER 2: Motion and activity metrics (background subtraction)
# ============================================================

class MotionExtractor:
    """
    Uses MOG2 background subtraction to detect moving regions.
    
    Best for fixed cameras (like webcams). The background model
    adapts over time, so parked cars or static objects gradually
    become part of the background.
    """

    def __init__(self, history=500, var_threshold=16, min_contour_area=500):
        """
        Args:
            history: Number of frames used to build background model
            var_threshold: Pixel variance threshold for foreground detection
            min_contour_area: Minimum pixel area for a moving object contour
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )
        self.min_contour_area = min_contour_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def extract(self, frame):
        """
        Analyze motion in the current frame.
        
        Args:
            frame: BGR image
        
        Returns:
            dict with motion_pct, moving_object_count, activity_level
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadow pixels (MOG2 marks shadows as 127)
        fg_mask[fg_mask < 255] = 0

        # Morphological cleanup: remove noise, fill small holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        # Find contours of moving regions
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by minimum area
        moving_objects = [
            c for c in contours if cv2.contourArea(c) > self.min_contour_area
        ]

        # Calculate motion percentage (foreground pixels / total pixels)
        total_pixels = fg_mask.size
        fg_pixels = np.count_nonzero(fg_mask)
        motion_pct = (fg_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        # Classify activity level
        if motion_pct > 5:
            activity_level = "high"
        elif motion_pct > 1:
            activity_level = "medium"
        else:
            activity_level = "low"

        return {
            "motion_pct": round(motion_pct, 2),
            "moving_object_count": len(moving_objects),
            "activity_level": activity_level,
        }

    def get_foreground_mask(self, frame):
        """Return the cleaned foreground mask for visualization."""
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask[fg_mask < 255] = 0
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        return fg_mask


# ============================================================
# LAYER 3: Optical flow (pedestrian direction analysis)
# ============================================================

class FlowExtractor:
    """
    Uses Farneback dense optical flow to determine the dominant
    direction of movement in the scene.
    
    Useful for detecting pedestrian flow patterns (e.g., morning
    commute flows left-to-right, evening flows right-to-left).
    """

    def __init__(self):
        self.prev_gray = None

    def extract(self, frame):
        """
        Compute optical flow between current and previous frame.
        
        Args:
            frame: BGR image
        
        Returns:
            dict with dominant_flow_dir (degrees), avg_flow_speed,
            or None if this is the first frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        self.prev_gray = gray

        # Convert to polar coordinates (magnitude + angle)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Only look at significant motion (top 10% of magnitude)
        threshold = np.percentile(magnitude, 90)
        sig_mask = magnitude > threshold

        if np.any(sig_mask):
            dominant_dir = float(np.median(angle[sig_mask]) * 180 / np.pi)
            avg_speed = float(np.mean(magnitude[sig_mask]))
        else:
            dominant_dir = 0.0
            avg_speed = 0.0

        return {
            "dominant_flow_dir": round(dominant_dir, 1),
            "avg_flow_speed": round(avg_speed, 2),
        }


# ============================================================
# LAYER 4: Object detection with YOLOv8
# ============================================================

class ObjectExtractor:
    """
    Uses YOLOv8 for object detection and counting.
    
    The COCO pretrained model detects 80 classes. For street scenes,
    the most relevant are: person (0), bicycle (1), car (2), motorcycle (3),
    bus (5), truck (7), dog (16), umbrella (25), backpack (24), suitcase (28).
    
    Creative insight: umbrella detection serves as a rain proxy,
    suitcase/backpack detection helps estimate tourist vs local ratio.
    """

    def __init__(self, model_name="yolov8n.pt", confidence=0.4):
        """
        Args:
            model_name: YOLO model file (auto-downloads on first use)
            confidence: Minimum detection confidence threshold
        """
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.class_names = self.model.names
        print(f"[ObjectExtractor] Loaded {model_name} "
              f"({len(self.class_names)} classes)")

    def extract(self, frame):
        """
        Run YOLO detection on a frame.
        
        Args:
            frame: BGR image
        
        Returns:
            dict with person_count, vehicle_count, bicycle_count,
            umbrella_count, backpack_count, suitcase_count, dog_count,
            is_raining, pedestrian_vehicle_ratio, crowd_density,
            detections (list of dicts for visualization)
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        boxes = results[0].boxes

        # Count each detected class
        class_ids = [int(c) for c in boxes.cls] if len(boxes) > 0 else []
        counts = Counter(
            self.class_names[cid] for cid in class_ids
        )

        person_count = counts.get("person", 0)
        vehicle_count = sum(
            counts.get(v, 0) for v in ["car", "bus", "truck", "motorcycle"]
        )
        bicycle_count = counts.get("bicycle", 0)
        umbrella_count = counts.get("umbrella", 0)
        backpack_count = counts.get("backpack", 0)
        suitcase_count = counts.get("suitcase", 0)
        dog_count = counts.get("dog", 0)

        # Crowd density classification
        if person_count > 30:
            crowd_density = "crowded"
        elif person_count > 15:
            crowd_density = "medium"
        elif person_count > 5:
            crowd_density = "low"
        else:
            crowd_density = "empty"

        # Build detection list for visualization
        detections = []
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    "class": self.class_names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        return {
            "person_count": person_count,
            "vehicle_count": vehicle_count,
            "bicycle_count": bicycle_count,
            "umbrella_count": umbrella_count,
            "backpack_count": backpack_count,
            "suitcase_count": suitcase_count,
            "dog_count": dog_count,
            "is_raining": umbrella_count > 0,
            "pedestrian_vehicle_ratio": round(
                person_count / max(vehicle_count, 1), 2
            ),
            "crowd_density": crowd_density,
            "detections": detections,
        }

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame: BGR image (will be modified in-place)
            detections: List of detection dicts from extract()
        
        Returns:
            Annotated frame
        """
        colors = {
            "person": (0, 255, 0),
            "car": (255, 0, 0),
            "bus": (255, 0, 0),
            "truck": (255, 0, 0),
            "motorcycle": (255, 0, 0),
            "bicycle": (0, 255, 255),
            "umbrella": (255, 0, 255),
            "dog": (0, 165, 255),
        }
        default_color = (128, 128, 128)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class"]
            conf = det["confidence"]
            color = colors.get(label, default_color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.0%}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                frame,
                (x1, y1 - text_size[1] - 6),
                (x1 + text_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                frame, text, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        return frame


# ============================================================
# LAYER 5: Engagement / Friendliness metrics
# ============================================================

def extract_engagement_metrics(frame, person_detections, timestamp,
                                wave_detector, photo_detector,
                                friendliness_index):
    """
    Run engagement detection and compute friendliness score.

    Orchestrates the WaveDetector, PhotoStopDetector, and FriendlinessIndex
    classes to produce a single metrics dict.

    Args:
        frame: BGR image (numpy array)
        person_detections: List of detection dicts where class == "person",
                          each with "bbox": (x1, y1, x2, y2)
        timestamp: datetime (UTC)
        wave_detector: WaveDetector instance
        photo_detector: PhotoStopDetector instance
        friendliness_index: FriendlinessIndex instance

    Returns:
        dict with wave_count, total_waves, photo_stop_count,
        total_photo_stops, friendliness_index, friendliness_level
    """
    # Detect waves (requires frame for MediaPipe pose)
    wave_data = wave_detector.detect(frame, person_detections, timestamp)

    # Detect photo-stops (centroid tracking only, no frame needed)
    stop_data = photo_detector.detect(person_detections, timestamp)

    # Record events into the friendliness index
    friendliness_index.record_waves(timestamp, wave_data["wave_count"])
    friendliness_index.record_stops(timestamp, stop_data["photo_stop_count"])

    # Compute rolling friendliness score
    friendliness_data = friendliness_index.compute(timestamp)

    # Merge all results into a flat dict
    return {
        "wave_count": wave_data["wave_count"],
        "total_waves": wave_data["total_waves"],
        "photo_stop_count": stop_data["photo_stop_count"],
        "total_photo_stops": stop_data["total_photo_stops"],
        "friendliness_index": friendliness_data["friendliness_index"],
        "friendliness_level": friendliness_data["friendliness_level"],
    }
