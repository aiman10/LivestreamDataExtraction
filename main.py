"""
main.py
Phase 1: Connect to Dublin EarthCam livestream, run YOLO object detection,
and display the annotated feed in a live window.

Usage:
    python main.py                              # Default: Dublin EarthCam
    python main.py --youtube "OTHER_YT_URL"     # Different YouTube stream
    python main.py --url "rtsp://..."           # Direct stream URL

Controls:
    q - quit
    s - save screenshot to screenshots/
"""

import cv2
import time
import datetime
import json
import os
import argparse
import signal
import sys
from collections import Counter

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _DUBLIN_TZ = _ZoneInfo("Europe/Dublin")
except ImportError:
    import pytz as _pytz
    _DUBLIN_TZ = _pytz.timezone("Europe/Dublin")

import config
from stream_capture import LivestreamReader
from object_detector import ObjectDetector
from wave_detector import WaveDetector
from photo_detector import PhotoTakingDetector
from crowd_safety import CrowdSafetyAnalyzer
from data_logger import DataLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Livestream Analytics - Phase 1")
    parser.add_argument("--url", type=str, help="Direct stream URL (MJPEG, RTSP)")
    parser.add_argument("--youtube", type=str, help="YouTube livestream URL")
    return parser.parse_args()


# Dublin time helpers
def get_dublin_time() -> datetime.datetime:
    return datetime.datetime.now(tz=_DUBLIN_TZ)


def get_time_period(hour: int) -> str:
    if hour < 6:
        return "Night"
    if hour < 12:
        return "Morning"
    if hour < 17:
        return "Afternoon"
    if hour < 21:
        return "Evening"
    return "Late Night"


# BGR colours for each period label on the overlay
_PERIOD_COLORS = {
    "Night":      (139, 0,   0  ),  # dark blue
    "Morning":    (0,   140, 255),  # orange
    "Afternoon":  (0,   200, 0  ),  # green
    "Evening":    (128, 0,   128),  # purple
    "Late Night": (0,   0,   139),  # dark red
}

# Per-period crowd thresholds: (crowded_min, very_crowded_min)
_PERIOD_THRESHOLDS = {
    "Night":      (3,  8 ),
    "Morning":    (8,  15),
    "Afternoon":  (15, 25),
    "Evening":    (12, 20),
    "Late Night": (6,  12),
}


def crowd_level(people: int, period: str = "Afternoon") -> tuple[str, tuple]:
    """Return (label, BGR colour) based on person count and time-of-day period."""
    crowded_min, very_crowded_min = _PERIOD_THRESHOLDS.get(period, (15, 25))
    if people > very_crowded_min:
        return " VERY CROWDED", (0, 0, 255)
    if people >= crowded_min:
        return " CROWDED",      (0, 140, 255)
    return " NORMAL",           (0, 200, 0)


def draw_counts_overlay(frame, summary: dict, fps: float,
                        waving: int = 0, photo_taking: int = 0):
    """
    Draw a compact object-count overlay in the top-left corner.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent black background
    box_h = 210
    box_w = 300
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    white = (220, 220, 220)
    y = 30

    # Title
    cv2.putText(frame, "Dublin Livestream Analytics", (16, y), font, 0.5, green, 1)
    y += 24

    # Dublin time + period
    dt = get_dublin_time()
    period = get_time_period(dt.hour)
    time_str = dt.strftime("%H:%M %Z")
    period_color = _PERIOD_COLORS[period]
    cv2.putText(frame, f"Dublin: {time_str}", (16, y), font, 0.45, white, 1)
    cv2.putText(frame, f"  [{period}]", (140, y), font, 0.42, period_color, 1)
    y += 22

    # Object counts
    people = summary.get("person_count", 0)
    vehicles = summary.get("vehicle_count", 0)
    bikes = summary.get("bicycle_count", 0)
    umbrellas = summary.get("umbrella_count", 0)
    total = summary.get("total_objects", 0)
    bg_people = summary.get("background_person_count", 0)

    level_label, level_color = crowd_level(people, period)
    cv2.putText(frame, f"People: {people}   Vehicles: {vehicles}", (16, y), font, 0.45, white, 1)
    cv2.putText(frame, level_label, (190, y), font, 0.42, level_color, 1)
    y += 22
    bg_color = (0, 128, 255) if bg_people > 0 else white
    cv2.putText(frame, f"Background people: {bg_people}", (16, y), font, 0.45, bg_color, 1)
    y += 22
    cv2.putText(frame, f"Bikes: {bikes}   Umbrellas: {umbrellas}", (16, y), font, 0.45, white, 1)
    y += 22
    cv2.putText(frame, f"Total objects: {total}", (16, y), font, 0.45, white, 1)
    y += 22
    wave_color = (0, 255, 255) if waving > 0 else white
    cv2.putText(frame, f"Waving: {waving}", (16, y), font, 0.45, wave_color, 1)
    y += 22
    photo_color = (255, 100, 255) if photo_taking > 0 else white
    cv2.putText(frame, f"Taking photo: {photo_taking}", (16, y), font, 0.45, photo_color, 1)
    y += 22
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, y), font, 0.4, (0, 180, 0), 1)

    return frame


def main():
    args = parse_args()

    # Determine stream source
    is_youtube = False
    if args.youtube:
        stream_url = args.youtube
        is_youtube = True
    elif args.url:
        stream_url = args.url
    else:
        stream_url = config.STREAM_URL
        is_youtube = config.IS_YOUTUBE

    # -- Initialize ----------------------------------------------------------
    print("=" * 55)
    print("  DUBLIN LIVESTREAM - Phase 1: Detection Preview")
    print("=" * 55)
    print(f"  Stream:  {stream_url[:60]}...")
    print(f"  YouTube: {is_youtube}")
    print()

    print("[1/2] Connecting to livestream...")
    try:
        reader = LivestreamReader(
            source=stream_url,
            is_youtube=is_youtube,
            youtube_resolution=config.YOUTUBE_RESOLUTION,
        )
    except (ConnectionError, RuntimeError) as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("[2/2] Loading YOLO model...")
    detector = ObjectDetector()

    print("[3/4] Loading wave detection model...")
    wave_detector = WaveDetector() if config.WAVE_ENABLED else None

    print("[4/4] Loading photo-taking detection model...")
    photo_detector = (
        PhotoTakingDetector(pose_model=wave_detector._model if wave_detector else None)
        if config.PHOTO_ENABLED else None
    )

    crowd_analyzer = CrowdSafetyAnalyzer() if config.CROWD_SAFETY_ENABLED else None
    if crowd_analyzer:
        print("[5/5] Crowd safety analyzer ready.")

    data_logger = DataLogger() if config.DATA_LOG_ENABLED else None

    print()
    print("Live window open. Press 'q' to quit, 's' for screenshot.")
    print("-" * 55)

    # -- Main loop -----------------------------------------------------------
    frame_count = 0
    latest_detections = []
    latest_summary = {}
    latest_wave_results = []
    latest_photo_results = []
    latest_safety = None
    safety_cycle = 0
    running = True

    # FPS tracking
    fps_timer = time.time()
    fps_frames = 0
    display_fps = 0.0

    # Graceful Ctrl+C
    def on_signal(sig, f):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)

    while running and reader.is_running():
        ret, frame = reader.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1

        # FPS calculation
        fps_frames += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            display_fps = fps_frames / (now - fps_timer)
            fps_frames = 0
            fps_timer = now

        # -- Run YOLO every N frames -----------------------------------------
        if frame_count % config.YOLO_EVERY == 0:
            latest_detections = detector.detect(frame)
            latest_summary = detector.summarize(latest_detections)

            if wave_detector and frame_count % config.WAVE_EVERY == 0:
                latest_wave_results = wave_detector.detect_waves(frame, frame_count)

            if photo_detector and frame_count % config.PHOTO_EVERY == 0:
                latest_photo_results = photo_detector.detect_photo_taking(frame, frame_count)

            # Crowd safety analysis
            if crowd_analyzer:
                safety_cycle += 1
                if safety_cycle % config.CROWD_SAFETY_EVERY == 0:
                    latest_safety = crowd_analyzer.analyze(
                        latest_detections, frame.shape)
                    for alert in latest_safety.alerts:
                        print(f"  [{alert.severity}] {alert.message}")

            # -- Log all metrics to CSV ------------------------------------------
            if data_logger:
                dt = get_dublin_time()
                period = get_time_period(dt.hour)
                c_label, _ = crowd_level(
                    latest_summary.get("person_count", 0), period)
                waving = sum(1 for r in latest_wave_results if r["is_waving"])
                photos = sum(
                    1 for r in latest_photo_results if r["is_taking_photo"])

                # Crowd safety fields
                s_status = latest_safety.status if latest_safety else "NORMAL"
                s_choke = len(latest_safety.choke_cells) if latest_safety else 0
                s_baseline = latest_safety.baseline_count if latest_safety else 0.0
                s_alerts = ""
                s_grid_json = "[]"
                s_max_density = 0
                if latest_safety:
                    s_alerts = ";".join(
                        a.alert_type for a in latest_safety.alerts)
                    s_grid_json = json.dumps(latest_safety.grid_density)
                    s_max_density = max(
                        (cell for row in latest_safety.grid_density
                         for cell in row), default=0)

                data_logger.log(
                    timestamp=dt.isoformat(),
                    dublin_time=dt.strftime("%H:%M:%S"),
                    time_period=period,
                    frame_number=frame_count,
                    summary=latest_summary,
                    crowd_level=c_label.strip(),
                    waving_count=waving,
                    photo_taking_count=photos,
                    safety_status=s_status,
                    choke_point_count=s_choke,
                    grid_max_density=s_max_density,
                    baseline_count=s_baseline,
                    active_alerts=s_alerts,
                    grid_density_json=s_grid_json,
                    fps=display_fps,
                )

            # Print to terminal periodically
            if frame_count % (config.YOLO_EVERY * 15) == 0:
                p = latest_summary.get("person_count", 0)
                v = latest_summary.get("vehicle_count", 0)
                u = latest_summary.get("umbrella_count", 0)
                t = latest_summary.get("total_objects", 0)
                dt = get_dublin_time()
                period = get_time_period(dt.hour)
                time_str = dt.strftime("%H:%M %Z")
                c_label, _ = crowd_level(p, period)
                print(
                    f"  Frame {frame_count:>6} | "
                    f"{display_fps:.1f} fps | "
                    f"{time_str} [{period}] | "
                    f"People: {p} [{c_label.strip()}] | Vehicles: {v} | "
                    f"Umbrellas: {u} | Total: {t}"
                )

        # -- Draw annotations on frame --------------------------------------
        display = frame.copy()

        if latest_detections:
            display = detector.draw(display, latest_detections)

        if config.SHOW_COUNTS:
            waving_count = sum(1 for r in latest_wave_results if r["is_waving"])
            photo_count  = sum(1 for r in latest_photo_results if r["is_taking_photo"])
            display = draw_counts_overlay(display, latest_summary, display_fps,
                                          waving_count, photo_count)

        if wave_detector and latest_wave_results:
            display = wave_detector.draw_wave_indicators(display, latest_wave_results)

        if photo_detector and latest_photo_results:
            display = photo_detector.draw_photo_indicators(display, latest_photo_results)

        if crowd_analyzer and latest_safety:
            display = crowd_analyzer.draw_overlay(display, latest_safety)

        # Resize for display
        display = cv2.resize(display, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

        cv2.imshow("Dublin Livestream Analytics", display)

        # -- Keyboard controls -----------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n'q' pressed. Stopping...")
            break
        elif key == ord("s"):
            os.makedirs("screenshots", exist_ok=True)
            path = f"screenshots/frame_{frame_count}.jpg"
            cv2.imwrite(path, display)
            print(f"  Screenshot saved: {path}")

    # -- Cleanup -------------------------------------------------------------
    reader.release()
    cv2.destroyAllWindows()
    if photo_detector:
        photo_detector.close()
    if data_logger:
        data_logger.close()

    elapsed = time.time() - fps_timer
    print()
    print("=" * 55)
    print("  Stopped.")
    print(f"  Frames processed: {frame_count}")
    print("=" * 55)


if __name__ == "__main__":
    main()
