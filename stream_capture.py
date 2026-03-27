"""
stream_capture.py
Handles connecting to livestreams (MJPEG, YouTube, RTSP) with threaded
frame reading to avoid buffer lag.
"""

import cv2
import time
import subprocess
import os
from threading import Thread, Lock


def get_youtube_stream_url(youtube_url, resolution="480p"):
    """
    Extract the direct stream URL from a YouTube livestream using yt-dlp.
    
    Args:
        youtube_url: Full YouTube URL (e.g. https://www.youtube.com/watch?v=...)
        resolution: Max resolution to request (e.g. "480p", "720p")
    
    Returns:
        Direct stream URL string
    
    Raises:
        RuntimeError: If yt-dlp fails or is not installed
    """
    height = resolution.replace("p", "")
    format_str = f"best[height<={height}][ext=mp4]/best[height<={height}]/best"

    cmd = ["yt-dlp", "-f", format_str, "-g", youtube_url]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with: pip install yt-dlp"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp timed out extracting stream URL")

    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    urls = result.stdout.strip().split("\n")
    return urls[0]


class LivestreamReader:
    """
    Threaded livestream reader that always provides the latest frame.
    
    The background thread continuously grabs frames, so when the main
    thread calls read(), it gets the most recent frame instead of a
    stale buffered one.
    
    Usage:
        reader = LivestreamReader("http://example.com/stream.mjpg")
        while True:
            ret, frame = reader.read()
            if ret:
                cv2.imshow("Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        reader.release()
    """

    def __init__(self, source, is_youtube=False, youtube_resolution="480p"):
        """
        Args:
            source: Stream URL (MJPEG, RTSP) or YouTube URL
            is_youtube: Set True if source is a YouTube URL
            youtube_resolution: Resolution cap for YouTube streams
        """
        self.source = source
        self.is_youtube = is_youtube
        self.youtube_resolution = youtube_resolution

        # Resolve the actual stream URL
        self._stream_url = self._resolve_url()

        # Set FFMPEG options for better stream handling
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"

        # Open the capture
        self.cap = cv2.VideoCapture(self._stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            raise ConnectionError(
                f"Could not open stream: {self._stream_url[:80]}..."
            )

        # Thread state
        self.lock = Lock()
        self.frame = None
        self.ret = False
        self.running = True
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.consecutive_failures = 0
        self.max_failures = 50

        # Start the background reader thread
        self._thread = Thread(target=self._update_loop, daemon=True)
        self._thread.start()

        print(f"[StreamCapture] Connected to stream (FPS: {self.fps:.1f})")

    def _resolve_url(self):
        """Get the direct stream URL, extracting from YouTube if needed."""
        if self.is_youtube:
            print("[StreamCapture] Extracting YouTube stream URL via yt-dlp...")
            return get_youtube_stream_url(self.source, self.youtube_resolution)
        return self.source

    def _update_loop(self):
        """Background thread: continuously grab the latest frame."""
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.ret = True
                    self.frame = frame
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1

                if self.consecutive_failures >= self.max_failures:
                    print("[StreamCapture] Too many failures, attempting reconnect...")
                    self._reconnect()

                time.sleep(0.05)

    def _reconnect(self):
        """Attempt to reconnect to the stream."""
        self.cap.release()
        self.consecutive_failures = 0

        # For YouTube, re-extract the URL (they expire after a few hours)
        if self.is_youtube:
            try:
                self._stream_url = self._resolve_url()
                print("[StreamCapture] Got fresh YouTube URL")
            except RuntimeError as e:
                print(f"[StreamCapture] YouTube URL refresh failed: {e}")
                time.sleep(5)
                return

        # Exponential backoff for reconnection attempts
        for attempt in range(5):
            wait_time = min(5 * (2 ** attempt), 60)
            print(f"[StreamCapture] Reconnect attempt {attempt + 1}/5 "
                  f"(waiting {wait_time}s)...")
            time.sleep(wait_time)

            self.cap = cv2.VideoCapture(self._stream_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            if self.cap.isOpened():
                print("[StreamCapture] Reconnected successfully")
                return

        print("[StreamCapture] All reconnect attempts failed")
        self.running = False

    def read(self):
        """
        Get the latest frame.
        
        Returns:
            Tuple of (success: bool, frame: numpy.ndarray or None)
        """
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def is_running(self):
        """Check if the reader is still active."""
        return self.running

    def get_fps(self):
        """Get the stream's reported FPS."""
        return self.fps

    def release(self):
        """Stop the background thread and release the capture."""
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=3)
        self.cap.release()
        print("[StreamCapture] Released")
