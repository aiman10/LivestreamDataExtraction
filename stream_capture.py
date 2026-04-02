"""
stream_capture.py
Threaded livestream reader that always provides the most recent frame,
discarding stale buffered frames automatically.

Supports YouTube livestreams (via yt-dlp) and direct URLs (MJPEG, RTSP).
"""

import cv2
import subprocess
import time
import os
from threading import Thread, Lock, Event

import config


def get_youtube_url(youtube_url: str, resolution: str = "480p") -> str:
    """
    Extract a direct playable URL from a YouTube livestream using yt-dlp.
    These URLs expire after a few hours, so reconnection logic re-calls this.
    """
    height = resolution.replace("p", "")
    fmt = f"best[height<={height}][ext=mp4]/best[height<={height}]/best"

    cmd = ["yt-dlp", "-f", fmt, "-g", "--no-warnings", youtube_url]
    print(f"[Stream] Extracting URL via yt-dlp...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        raise ConnectionError(f"yt-dlp failed: {result.stderr.strip()}")

    url = result.stdout.strip().split("\n")[0]
    print(f"[Stream] Got direct URL ({len(url)} chars)")
    return url


class LivestreamReader:
    """
    Background-threaded video reader. Always returns the latest frame
    instead of reading from a growing buffer (which causes lag).

    Usage:
        reader = LivestreamReader("https://youtube.com/...", is_youtube=True)
        while reader.is_running():
            ret, frame = reader.read()
            if ret:
                cv2.imshow("Feed", frame)
        reader.release()
    """

    def __init__(
        self,
        source: str,
        is_youtube: bool = False,
        youtube_resolution: str = "480p",
    ):
        self.source = source
        self.is_youtube = is_youtube
        self.youtube_resolution = youtube_resolution

        self.lock = Lock()
        self.frame = None
        self.ret = False
        self._stop_event = Event()
        self.running = True
        self.cap = None
        self.fps = 30.0
        self._reconnect_attempts = 0
        self._max_reconnects = 8

        # Reduce internal ffmpeg buffering
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"

        self._connect()
        self._thread = Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _connect(self):
        """Open the video stream."""
        try:
            if self.is_youtube:
                direct_url = get_youtube_url(self.source, self.youtube_resolution)
            else:
                direct_url = self.source

            self.cap = cv2.VideoCapture(direct_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if not self.cap.isOpened():
                raise ConnectionError("VideoCapture failed to open")

            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"[Stream] Connected. FPS: {self.fps:.0f}")
            self._reconnect_attempts = 0

        except Exception as e:
            print(f"[Stream] Connection error: {e}")
            self._reconnect_attempts += 1

    def _reader_loop(self):
        """Background thread: continuously grab the newest frame."""
        fail_count = 0

        while not self._stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self._reconnect()
                continue

            ret, frame = self.cap.read()

            if self._stop_event.is_set():
                break

            if ret:
                fail_count = 0
                with self.lock:
                    self.ret = True
                    self.frame = frame
            else:
                fail_count += 1
                if fail_count > 60:
                    print("[Stream] Too many read failures, reconnecting...")
                    self._reconnect()
                    fail_count = 0
                time.sleep(0.01)

    def _reconnect(self):
        """Reconnect with exponential backoff."""
        if self._reconnect_attempts >= self._max_reconnects:
            print("[Stream] Max reconnect attempts reached. Stopping.")
            self._stop_event.set()
            self.running = False
            return

        wait = min(5 * (2 ** self._reconnect_attempts), 60)
        print(f"[Stream] Reconnecting in {wait}s (attempt {self._reconnect_attempts + 1})...")
        time.sleep(wait)

        if self.cap is not None:
            self.cap.release()
        self._connect()

    def read(self):
        """Return the most recent frame (thread-safe)."""
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
        return False, None

    def is_running(self) -> bool:
        return self.running and not self._stop_event.is_set()

    def release(self):
        """Signal the reader thread to stop, wait for it to exit, then release the cap."""
        self._stop_event.set()
        self.running = False
        self._thread.join(timeout=3.0)
        if self.cap is not None:
            self.cap.release()
        print("[Stream] Released.")
