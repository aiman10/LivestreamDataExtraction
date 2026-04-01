"""
test_event_capture.py  —  TEMPORARY TEST SCRIPT
================================================
Saves annotated screenshots to event_captures/ whenever a wave or
photo-taking event is detected. Useful for offline review and research.

Usage:
    python test_event_capture.py                     # default Dublin stream
    python test_event_capture.py --url 0             # webcam
    python test_event_capture.py --youtube "URL"     # other YouTube stream

Output folder: event_captures/
  wave_<frame>_<timestamp>.jpg
  photo_<frame>_<timestamp>.jpg

Press 'q' to quit.
"""

import argparse
import os
import sys
import time

import cv2

import config
from stream_capture import LivestreamReader
from wave_detector import WaveDetector
from photo_detector import PhotoTakingDetector


SAVE_DIR = "event_captures"
# Minimum seconds between screenshots for the same track+event type,
# to avoid flooding the folder with near-identical images.
COOLDOWN_SEC = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="Event screenshot capture — test script")
    p.add_argument("--url",     type=str, help="Direct stream URL or webcam index (e.g. 0)")
    p.add_argument("--youtube", type=str, help="YouTube livestream URL")
    return p.parse_args()


def save_screenshot(display, event: str, frame_idx: int) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{event}_{frame_idx}_{ts}.jpg")
    cv2.imwrite(path, display)
    return path


def main():
    args = parse_args()

    if args.youtube:
        stream_url, is_youtube = args.youtube, True
    elif args.url:
        # Allow numeric webcam index
        stream_url = int(args.url) if args.url.isdigit() else args.url
        is_youtube = False
    else:
        stream_url, is_youtube = config.STREAM_URL, config.IS_YOUTUBE

    print("=" * 55)
    print("  EVENT CAPTURE — test script")
    print("=" * 55)
    print(f"  Output: {SAVE_DIR}/")
    print(f"  Cooldown per track: {COOLDOWN_SEC}s")
    print()

    print("[1/3] Connecting to stream...")
    try:
        reader = LivestreamReader(
            source=stream_url,
            is_youtube=is_youtube,
            youtube_resolution=config.YOUTUBE_RESOLUTION,
        )
    except (ConnectionError, RuntimeError) as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("[2/3] Loading wave detector...")
    wave_detector = WaveDetector()

    print("[3/3] Loading photo detector...")
    photo_detector = PhotoTakingDetector(pose_model=wave_detector._model)

    print()
    print("Running. Press 'q' to quit.")
    print("-" * 55)

    frame_count = 0
    # last_saved[event_type][track_id] = epoch time of last save
    last_saved: dict[str, dict[int, float]] = {"wave": {}, "photo": {}}

    while reader.is_running():
        ret, frame = reader.read()
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1

        if frame_count % config.WAVE_EVERY != 0:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Run both detectors on this frame
        wave_results  = wave_detector.detect_waves(frame, frame_count)
        photo_results = photo_detector.detect_photo_taking(frame, frame_count)

        # Build annotated display frame
        display = frame.copy()
        display = wave_detector.draw_wave_indicators(display, wave_results)
        display = photo_detector.draw_photo_indicators(display, photo_results)
        display = cv2.resize(display, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))

        now = time.time()

        # --- Wave screenshots ---
        for r in wave_results:
            if not r["is_waving"]:
                continue
            tid = r["track_id"]
            since = now - last_saved["wave"].get(tid, 0)
            if since >= COOLDOWN_SEC:
                path = save_screenshot(display, "wave", frame_count)
                last_saved["wave"][tid] = now
                print(f"  [WAVE]  track={tid}  frame={frame_count}  -> {path}")

        # --- Photo screenshots ---
        for r in photo_results:
            if not r["is_taking_photo"]:
                continue
            tid = r["track_id"]
            since = now - last_saved["photo"].get(tid, 0)
            if since >= COOLDOWN_SEC:
                path = save_screenshot(display, "photo", frame_count)
                last_saved["photo"][tid] = now
                print(f"  [PHOTO] track={tid}  conf={r['confidence']:.0%}  frame={frame_count}  -> {path}")

        cv2.imshow("Event Capture (test)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    photo_detector.close()
    cv2.destroyAllWindows()
    print()
    print(f"Done. Screenshots saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
