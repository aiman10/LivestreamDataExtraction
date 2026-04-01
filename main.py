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
import os
import argparse
import signal
import sys
from collections import Counter

import config
from stream_capture import LivestreamReader
from object_detector import ObjectDetector
from wave_detector import WaveDetector
from photo_detector import PhotoTakingDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Livestream Analytics - Phase 1")
    parser.add_argument("--url", type=str, help="Direct stream URL (MJPEG, RTSP)")
    parser.add_argument("--youtube", type=str, help="YouTube livestream URL")
    return parser.parse_args()


def draw_counts_overlay(frame, summary: dict, fps: float,
                        waving: int = 0, photo_taking: int = 0):
    """
    Draw a compact object-count overlay in the top-left corner.
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Semi-transparent black background
    box_h = 174
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

    # Object counts
    people = summary.get("person_count", 0)
    vehicles = summary.get("vehicle_count", 0)
    bikes = summary.get("bicycle_count", 0)
    umbrellas = summary.get("umbrella_count", 0)
    total = summary.get("total_objects", 0)

    cv2.putText(frame, f"People: {people}   Vehicles: {vehicles}", (16, y), font, 0.45, white, 1)
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

    print()
    print("Live window open. Press 'q' to quit, 's' for screenshot.")
    print("-" * 55)

    # -- Main loop -----------------------------------------------------------
    frame_count = 0
    latest_detections = []
    latest_summary = {}
    latest_wave_results = []
    latest_photo_results = []
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

            # Print to terminal periodically
            if frame_count % (config.YOLO_EVERY * 15) == 0:
                p = latest_summary.get("person_count", 0)
                v = latest_summary.get("vehicle_count", 0)
                u = latest_summary.get("umbrella_count", 0)
                t = latest_summary.get("total_objects", 0)
                print(
                    f"  Frame {frame_count:>6} | "
                    f"{display_fps:.1f} fps | "
                    f"People: {p} | Vehicles: {v} | "
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

    elapsed = time.time() - fps_timer
    print()
    print("=" * 55)
    print("  Stopped.")
    print(f"  Frames processed: {frame_count}")
    print("=" * 55)


if __name__ == "__main__":
    main()
