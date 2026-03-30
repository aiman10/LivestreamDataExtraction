"""
main.py
Master orchestrator for the Livestream Analytics Pipeline.

Pipeline: Livestream -> Frame Processing -> Feature Extraction ->
          Event Generation -> CSV Storage -> (Grafana Dashboard)

Usage:
    python main.py                     # Run with YouTube stream from config.py
    python main.py --no-display        # Headless mode (no video window)
    python main.py --url "URL"         # Override stream URL
    python main.py --youtube "YT_URL"  # Override YouTube livestream URL
"""

import cv2
import time
import argparse
import signal
import sys
from datetime import datetime, timezone

import config
from stream_capture import LivestreamReader
from feature_extractors import (
    extract_scene_metrics,
    extract_engagement_metrics,
    MotionExtractor,
    FlowExtractor,
    ObjectExtractor,
)
from event_engine import (
    ThresholdEventEngine,
    RollingAnomalyDetector,
    SceneStateMachine,
)
from data_writer import DataWriter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Livestream Analytics Pipeline"
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help="Direct stream URL (MJPEG, RTSP)"
    )
    parser.add_argument(
        "--youtube", type=str, default=None,
        help="YouTube livestream URL"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without video display (headless mode)"
    )
    parser.add_argument(
        "--no-yolo", action="store_true",
        help="Skip YOLO object detection (faster, less data)"
    )
    return parser.parse_args()


def draw_overlay(frame, metrics, scene_state, stats):
    """
    Draw a semi-transparent metrics overlay on the video frame.
    
    Args:
        frame: BGR image (modified in place)
        metrics: dict of current metrics
        scene_state: current state string
        stats: dict with metrics_rows and events_rows counts
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Background rectangle
    overlay_height = 180
    cv2.rectangle(overlay, (10, 10), (380, 10 + overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Text lines
    lines = [
        f"State: {scene_state.upper()}",
        f"People: {metrics.get('person_count', '?')} Motion: {metrics.get('motion_pct', '?')}%  "
        f"Activity: {metrics.get('activity_level', '?')}",
        f"Brightness: {metrics.get('brightness', '?')}  "
        f"Daytime: {metrics.get('is_daytime', '?')}",
        f"Umbrellas: {metrics.get('umbrella_count', '?')}  "
        f"Rain: {metrics.get('is_raining', '?')}",
        f"Logged: {stats['metrics_rows']} rows, "
        f"{stats['events_rows']} events",
        f"Friendliness: {metrics.get('friendliness_index', '?')}/100 "
        f"({metrics.get('friendliness_level', '?')})",
        f"Waves: {metrics.get('total_waves', 0)}  "
        f"Photo-stops: {metrics.get('total_photo_stops', 0)}",
    ]

    y = 35
    for line in lines:
        cv2.putText(
            frame, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
        )
        y += 24

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

    show_display = not args.no_display and config.SHOW_VIDEO
    use_yolo = not args.no_yolo

    # --------------------------------------------------------
    # INITIALIZE ALL COMPONENTS
    # --------------------------------------------------------
    print("=" * 60)
    print("  LIVESTREAM ANALYTICS PIPELINE")
    print("=" * 60)
    print(f"  Stream: {stream_url[:70]}...")
    print(f"  YouTube: {is_youtube}")
    print(f"  Display: {show_display}")
    print(f"  YOLO: {use_yolo}")
    print()

    # Phase 1: Stream capture
    print("[1/5] Connecting to livestream...")
    try:
        reader = LivestreamReader(
            source=stream_url,
            is_youtube=is_youtube,
            youtube_resolution=config.YOUTUBE_RESOLUTION,
        )
    except (ConnectionError, RuntimeError) as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # Phase 2: Feature extractors
    print("[2/5] Initializing feature extractors...")
    motion_extractor = MotionExtractor()
    flow_extractor = FlowExtractor()
    object_extractor = None
    if use_yolo:
        object_extractor = ObjectExtractor(
            model_name=config.YOLO_MODEL,
            confidence=config.YOLO_CONFIDENCE,
        )

    # Phase 2b: Engagement detectors (requires MediaPipe)
    wave_detector = None
    photo_detector = None
    friendliness = None
    if use_yolo:
        try:
            from engagement_detector import (
                WaveDetector, PhotoStopDetector, FriendlinessIndex,
            )
            wave_detector = WaveDetector()
            photo_detector = PhotoStopDetector()
            friendliness = FriendlinessIndex()
            print("  Engagement detectors initialized (MediaPipe loaded)")
        except ImportError:
            print("  MediaPipe not available, engagement detection disabled")

    # Phase 3: Event engine
    print("[3/5] Setting up event engine...")
    threshold_engine = ThresholdEventEngine()
    motion_anomaly = RollingAnomalyDetector()
    state_machine = SceneStateMachine()

    # Phase 4: Data writer
    print("[4/5] Opening data files...")
    writer = DataWriter()

    # Phase 5 note
    print("[5/5] Data will be saved to CSV. Import to InfluxDB/Grafana later.")
    print()
    print("Pipeline running. Press 'q' to stop.")
    print("-" * 60)

    # --------------------------------------------------------
    # MAIN PROCESSING LOOP
    # --------------------------------------------------------
    frame_count = 0
    current_metrics = {}       # Accumulated metrics for the current log interval
    last_detections = []       # YOLO detections for visualization
    running = True

    # Graceful shutdown on Ctrl+C
    def signal_handler(sig, frame_arg):
        nonlocal running
        print("\nShutdown signal received...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()

    while running and reader.is_running():
        ret, frame = reader.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        timestamp = datetime.now(timezone.utc)

        # --- LAYER 1: Motion (every frame, very cheap) ---
        if frame_count % config.MOTION_EVERY == 0:
            motion_data = motion_extractor.extract(frame)
            current_metrics.update(motion_data)

            # Feed motion to anomaly detector
            motion_anomaly.evaluate(motion_data["motion_pct"])

        # --- LAYER 2: Optical flow (every N frames) ---
        if frame_count % config.FLOW_EVERY == 0:
            flow_data = flow_extractor.extract(frame)
            if flow_data:
                current_metrics.update(flow_data)

        # --- LAYER 3: Scene metrics (every N frames) ---
        if frame_count % config.SCENE_EVERY == 0:
            scene_data = extract_scene_metrics(frame)
            current_metrics.update(scene_data)

        # --- LAYER 4: YOLO object detection (every N frames) ---
        if use_yolo and frame_count % config.YOLO_EVERY == 0:
            object_data = object_extractor.extract(frame)
            last_detections = object_data.pop("detections", [])
            current_metrics.update(object_data)

            # Update state machine with person count
            person_count = object_data.get("person_count", 0)
            transition = state_machine.update(person_count)
            if transition:
                print(f"  STATE CHANGE: {transition}")

        # --- LAYER 5: Engagement detection (every N frames) ---
        if (wave_detector is not None
                and frame_count % config.ENGAGEMENT_EVERY == 0
                and last_detections):
            person_dets = [d for d in last_detections if d["class"] == "person"]
            engagement_data = extract_engagement_metrics(
                frame, person_dets, timestamp,
                wave_detector, photo_detector, friendliness,
            )
            current_metrics.update(engagement_data)

        # --- Current state ---
        scene_state = state_machine.get_state()
        current_metrics["scene_state"] = scene_state

        # --- EVENT CHECKS (every log interval) ---
        if frame_count % config.LOG_EVERY == 0:
            # Threshold events
            events = threshold_engine.check(
                timestamp, current_metrics, scene_state
            )

            # Activity spike from anomaly detector
            rolling_avg = motion_anomaly.get_rolling_avg()
            motion_pct = current_metrics.get("motion_pct", 0)
            if rolling_avg > 0 and motion_pct > rolling_avg * config.ACTIVITY_SPIKE_MULTIPLIER:
                spike_event = {
                    "timestamp": timestamp.isoformat(),
                    "event_type": "activity_spike",
                    "severity": "WARNING",
                    "metric_value": round(motion_pct, 2),
                    "threshold": round(rolling_avg * config.ACTIVITY_SPIKE_MULTIPLIER, 2),
                    "description": (
                        f"Motion {motion_pct:.1f}% is "
                        f"{config.ACTIVITY_SPIKE_MULTIPLIER}x above "
                        f"rolling avg {rolling_avg:.1f}%"
                    ),
                    "scene_state": scene_state,
                }
                events.append(spike_event)
                print(f"  EVENT: [WARNING] {spike_event['description']}")

            # Write events
            if events:
                writer.write_events(events)

            # --- LOG METRICS ROW ---
            current_metrics["timestamp"] = timestamp.isoformat()
            writer.write_metric(current_metrics)

            # Print periodic status
            elapsed = time.time() - start_time
            stats = writer.get_stats()
            fps_actual = frame_count / elapsed if elapsed > 0 else 0

            if stats["metrics_rows"] % 10 == 0:
                print(
                    f"  [{timestamp.strftime('%H:%M:%S')}] "
                    f"Frame {frame_count} | "
                    f"{fps_actual:.1f} fps | "
                    f"People: {current_metrics.get('person_count', '?')} | "
                    f"Motion: {current_metrics.get('motion_pct', '?')}% | "
                    f"State: {scene_state} | "
                    f"Rows: {stats['metrics_rows']}"
                )

        # --- DISPLAY ---
        if show_display:
            display_frame = frame.copy()

            # Draw YOLO bounding boxes
            if use_yolo and last_detections:
                display_frame = object_extractor.draw_detections(
                    display_frame, last_detections
                )

            # Draw metrics overlay
            if config.OVERLAY_METRICS:
                stats = writer.get_stats()
                display_frame = draw_overlay(
                    display_frame, current_metrics, scene_state, stats
                )

            # Resize for display
            if config.DISPLAY_WIDTH and config.DISPLAY_HEIGHT:
                display_frame = cv2.resize(
                    display_frame,
                    (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
                )

            cv2.imshow("Livestream Analytics", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n'q' pressed, stopping...")
                break
            elif key == ord("s"):
                # Save a screenshot
                screenshot_path = f"data/screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"  Screenshot saved: {screenshot_path}")

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE STOPPED")
    print("=" * 60)

    elapsed = time.time() - start_time
    stats = writer.get_stats()
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"  Frames processed: {frame_count}")
    print(f"  Avg FPS: {frame_count / elapsed:.1f}" if elapsed > 0 else "")
    print(f"  Metrics logged: {stats['metrics_rows']}")
    print(f"  Events logged: {stats['events_rows']}")
    print(f"  Data files: {config.METRICS_CSV_PATH}, {config.EVENTS_CSV_PATH}")
    print()
    print("Next steps:")
    print("  1. docker-compose up -d     (start Grafana + InfluxDB)")
    print("  2. python influx_importer.py (load CSV into InfluxDB)")
    print("  3. Open http://localhost:3000 (Grafana dashboard)")

    reader.release()
    writer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
