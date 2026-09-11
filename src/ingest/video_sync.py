"""
video_sync.py
=============
Video capture and high-speed telemetry synchronization engine.
Produces unified (Frame, Telemetry) packets for downstream vision and georeferencing.
"""

from dataclasses import dataclass
from typing import Optional, Generator, Tuple
import time
import logging
import cv2
import numpy as np

from .telemetry_parser import TelemetryParser, TelemetrySample

logger = logging.getLogger("Ingest.VideoSync")


@dataclass(slots=True)
class SynchronizedFrame:
    """Unified packet containing video image paired with exact drone geospatial telemetry."""
    frame_id: int
    timestamp_ms: float
    rgb_image: np.ndarray
    lat: float
    lon: float
    altitude_agl: float
    pitch: float
    roll: float
    yaw: float
    is_synced: bool
    sync_latency_ms: float = 0.0
    time_delta_ms: float = 0.0


class VideoSynchronizer:
    """
    Ingests video streams (MP4/RTSP/USB) alongside telemetry timelines and yields
    synchronized frames with guaranteed latency and tolerance checks.
    """

    def __init__(
        self,
        video_source: str,
        telemetry_parser: TelemetryParser,
        max_tolerance_ms: float = 100.0,
        drop_unsynced: bool = False,
        time_offset_ms: float = 0.0,
        is_live_stream: bool = False
    ):
        self.video_source = video_source
        self.telemetry_parser = telemetry_parser
        self.max_tolerance_ms = max_tolerance_ms
        self.drop_unsynced = drop_unsynced
        self.time_offset_ms = time_offset_ms
        self.is_live_stream = is_live_stream

        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        
        self._init_capture()

    def _init_capture(self):
        # Hardware-accelerated backends can be selected if available (e.g. CAP_FFMPEG, CAP_DSHOW)
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video source: '{self.video_source}'")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if (fps > 0 and not np.isnan(fps)) else 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Optimize buffer size for low latency on live streams
        if self.is_live_stream:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def stream_synchronized_frames(self) -> Generator[SynchronizedFrame, None, None]:
        """
        Sequentially reads video frames, queries the interpolated telemetry state,
        and yields SynchronizedFrame objects.
        """
        frame_idx = 0
        start_mono_time = time.monotonic() * 1000.0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    break

                sync_start = time.perf_counter()

                # Calculate frame presentation timestamp (PTS in ms)
                if self.is_live_stream:
                    frame_pts_ms = (time.monotonic() * 1000.0) - start_mono_time
                else:
                    # Video file: use CAP_PROP_POS_MSEC if valid, fallback to frame_idx / fps
                    pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    if pos_msec > 0:
                        frame_pts_ms = float(pos_msec)
                    else:
                        frame_pts_ms = (frame_idx / self.fps) * 1000.0

                effective_pts = frame_pts_ms + self.time_offset_ms

                # Query interpolated telemetry
                telemetry, delta_ms, is_synced = self.telemetry_parser.get_telemetry_at(
                    query_time_ms=effective_pts,
                    max_tolerance_ms=self.max_tolerance_ms
                )

                sync_elapsed_ms = (time.perf_counter() - sync_start) * 1000.0

                if not is_synced and self.drop_unsynced:
                    frame_idx += 1
                    continue

                packet = SynchronizedFrame(
                    frame_id=frame_idx,
                    timestamp_ms=round(frame_pts_ms, 2),
                    rgb_image=frame,
                    lat=telemetry.lat,
                    lon=telemetry.lon,
                    altitude_agl=telemetry.altitude_agl,
                    pitch=telemetry.pitch,
                    roll=telemetry.roll,
                    yaw=telemetry.yaw,
                    is_synced=is_synced,
                    sync_latency_ms=round(sync_elapsed_ms, 3),
                    time_delta_ms=round(delta_ms, 2)
                )

                yield packet
                frame_idx += 1

        finally:
            self.release()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
