"""
telemetry_parser.py
===================
High-performance telemetry ingestion, buffering, and timestamp interpolation.
Supports CSV logs and MAVLink (GLOBAL_POSITION_INT + ATTITUDE) telemetry streams.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import csv
import bisect
import math
import numpy as np


@dataclass(slots=True)
class TelemetrySample:
    """Represents a single aerial telemetry observation."""
    timestamp_ms: float
    lat: float
    lon: float
    altitude_agl: float
    pitch: float
    roll: float
    yaw: float


def interpolate_angle_deg(a1: float, a2: float, alpha: float) -> float:
    """
    Interpolates between two angles in degrees using the shortest angular path.
    Properly handles 0/360 wrap-around.
    """
    diff = (a2 - a1 + 180.0) % 360.0 - 180.0
    result = (a1 + alpha * diff) % 360.0
    if result < 0:
        result += 360.0
    return result


class TelemetryParser:
    """
    Stores telemetry in a fast, sorted timeline buffer and provides sub-millisecond
    interpolation for arbitrary query timestamps.
    """

    def __init__(self, max_buffer_size: int = 100000):
        self.max_buffer_size = max_buffer_size
        self._timestamps: List[float] = []
        self._samples: List[TelemetrySample] = []
        
        # Cached NumPy arrays for high-speed vectorized operations
        self._np_timestamps: Optional[np.ndarray] = None
        self._np_data: Optional[np.ndarray] = None  # [N, 6] -> lat, lon, alt, pitch, roll, yaw
        self._is_dirty: bool = True

    def clear(self):
        self._timestamps.clear()
        self._samples.clear()
        self._np_timestamps = None
        self._np_data = None
        self._is_dirty = True

    def add_sample(self, sample: TelemetrySample):
        """Adds a single sample into the sorted buffer (suitable for real-time live ingestion)."""
        idx = bisect.bisect_right(self._timestamps, sample.timestamp_ms)
        self._timestamps.insert(idx, sample.timestamp_ms)
        self._samples.insert(idx, sample)
        self._is_dirty = True
        
        if len(self._timestamps) > self.max_buffer_size:
            self._timestamps.pop(0)
            self._samples.pop(0)

    def load_csv(self, filepath: str):
        """
        Loads telemetry from a CSV file.
        Expected headers: timestamp_ms, lat, lon, altitude_agl, pitch, roll, yaw
        """
        self.clear()
        with open(filepath, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Normalize headers (strip whitespace and lower case)
            header_map = {k.strip().lower(): k for k in reader.fieldnames or []}
            
            req_cols = ["timestamp_ms", "lat", "lon", "altitude_agl", "pitch", "roll", "yaw"]
            for col in req_cols:
                if col not in header_map:
                    raise ValueError(f"Missing required telemetry header: '{col}' in {filepath}")
                    
            loaded_samples = []
            for row in reader:
                try:
                    s = TelemetrySample(
                        timestamp_ms=float(row[header_map["timestamp_ms"]]),
                        lat=float(row[header_map["lat"]]),
                        lon=float(row[header_map["lon"]]),
                        altitude_agl=float(row[header_map["altitude_agl"]]),
                        pitch=float(row[header_map["pitch"]]),
                        roll=float(row[header_map["roll"]]),
                        yaw=float(row[header_map["yaw"]])
                    )
                    loaded_samples.append(s)
                except (ValueError, TypeError):
                    continue
                    
        # Sort by timestamp
        loaded_samples.sort(key=lambda x: x.timestamp_ms)
        for s in loaded_samples:
            self._timestamps.append(s.timestamp_ms)
            self._samples.append(s)
            
        self._sync_numpy()

    def _sync_numpy(self):
        """Compiles timeline lists into contiguous NumPy arrays for accelerated querying."""
        if not self._timestamps:
            self._np_timestamps = np.empty(0, dtype=np.float64)
            self._np_data = np.empty((0, 6), dtype=np.float64)
        else:
            self._np_timestamps = np.array(self._timestamps, dtype=np.float64)
            data_matrix = np.zeros((len(self._samples), 6), dtype=np.float64)
            for i, s in enumerate(self._samples):
                data_matrix[i, :] = [s.lat, s.lon, s.altitude_agl, s.pitch, s.roll, s.yaw]
            self._np_data = data_matrix
        self._is_dirty = False

    def get_telemetry_at(
        self,
        query_time_ms: float,
        max_tolerance_ms: float = 100.0
    ) -> Tuple[TelemetrySample, float, bool]:
        """
        Interpolates drone position and orientation at an arbitrary query timestamp.
        
        Args:
            query_time_ms: Target frame presentation timestamp in milliseconds.
            max_tolerance_ms: Maximum allowable time delta before marking telemetry as degraded.
            
        Returns:
            Tuple[TelemetrySample, float, bool]:
                - interpolated TelemetrySample
                - time_delta_ms (absolute distance to closest telemetry observation)
                - is_synced (True if telemetry is within tolerance and valid)
        """
        if self._is_dirty:
            self._sync_numpy()
            
        n = len(self._timestamps)
        if n == 0:
            # Fallback zero sample
            return TelemetrySample(query_time_ms, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), float("inf"), False
            
        # Fast binary search
        idx = int(np.searchsorted(self._np_timestamps, query_time_ms))
        
        # Case 1: Query is before the first recorded telemetry sample
        if idx == 0:
            delta = abs(query_time_ms - self._timestamps[0])
            s = self._samples[0]
            is_synced = delta <= max_tolerance_ms
            return TelemetrySample(
                timestamp_ms=query_time_ms,
                lat=s.lat, lon=s.lon, altitude_agl=s.altitude_agl,
                pitch=s.pitch, roll=s.roll, yaw=s.yaw
            ), delta, is_synced
            
        # Case 2: Query is after the last recorded telemetry sample
        if idx >= n:
            delta = abs(query_time_ms - self._timestamps[-1])
            s = self._samples[-1]
            is_synced = delta <= max_tolerance_ms
            return TelemetrySample(
                timestamp_ms=query_time_ms,
                lat=s.lat, lon=s.lon, altitude_agl=s.altitude_agl,
                pitch=s.pitch, roll=s.roll, yaw=s.yaw
            ), delta, is_synced
            
        # Case 3: Query lies between two telemetry samples [idx - 1] and [idx]
        t1 = self._timestamps[idx - 1]
        t2 = self._timestamps[idx]
        s1 = self._samples[idx - 1]
        s2 = self._samples[idx]
        
        delta = min(abs(query_time_ms - t1), abs(query_time_ms - t2))
        
        # Telemetry gap check: If the gap between successive packets is too large (> 500ms),
        # do not interpolate blindly across missing telemetry
        gap_ms = t2 - t1
        if gap_ms > max_tolerance_ms * 5:
            is_synced = delta <= max_tolerance_ms
            # Fall back to nearest sample
            nearest = s1 if abs(query_time_ms - t1) <= abs(query_time_ms - t2) else s2
            return TelemetrySample(
                timestamp_ms=query_time_ms,
                lat=nearest.lat, lon=nearest.lon, altitude_agl=nearest.altitude_agl,
                pitch=nearest.pitch, roll=nearest.roll, yaw=nearest.yaw
            ), delta, is_synced
            
        alpha = (query_time_ms - t1) / gap_ms if gap_ms > 0 else 0.0
        alpha = max(0.0, min(1.0, alpha))
        
        # Linear interpolation for geospatial position
        interp_lat = s1.lat + alpha * (s2.lat - s1.lat)
        interp_lon = s1.lon + alpha * (s2.lon - s1.lon)
        interp_alt = s1.altitude_agl + alpha * (s2.altitude_agl - s1.altitude_agl)
        
        # Angular shortest-path interpolation for orientation
        interp_pitch = s1.pitch + alpha * (s2.pitch - s1.pitch)
        interp_roll = s1.roll + alpha * (s2.roll - s1.roll)
        interp_yaw = interpolate_angle_deg(s1.yaw, s2.yaw, alpha)
        
        is_synced = delta <= max_tolerance_ms
        
        return TelemetrySample(
            timestamp_ms=query_time_ms,
            lat=round(interp_lat, 7),
            lon=round(interp_lon, 7),
            altitude_agl=round(interp_alt, 2),
            pitch=round(interp_pitch, 2),
            roll=round(interp_roll, 2),
            yaw=round(interp_yaw, 2)
        ), round(delta, 2), is_synced


class MAVLinkTelemetryParser:
    """
    Parser and mock harness for MAVLink telemetry streams.
    Extracts GLOBAL_POSITION_INT and ATTITUDE packets and syncs to a TelemetryParser timeline.
    """

    def __init__(self, telemetry_parser: Optional[TelemetryParser] = None):
        self.parser = telemetry_parser or TelemetryParser()
        self._last_position: Optional[Dict[str, Any]] = None
        self._last_attitude: Optional[Dict[str, Any]] = None

    def process_mavlink_message(self, msg_type: str, msg_dict: Dict[str, Any], timestamp_ms: float):
        """
        Processes a decoded MAVLink message packet.
        
        GLOBAL_POSITION_INT expected fields:
            - lat: deg * 1e7
            - lon: deg * 1e7
            - relative_alt: mm (above ground / home)
        
        ATTITUDE expected fields:
            - roll: radians
            - pitch: radians
            - yaw: radians
        """
        if msg_type == "GLOBAL_POSITION_INT":
            self._last_position = {
                "timestamp_ms": timestamp_ms,
                "lat": float(msg_dict.get("lat", 0)) / 1e7,
                "lon": float(msg_dict.get("lon", 0)) / 1e7,
                "altitude_agl": float(msg_dict.get("relative_alt", 0)) / 1000.0
            }
        elif msg_type == "ATTITUDE":
            self._last_attitude = {
                "timestamp_ms": timestamp_ms,
                "roll": math.degrees(float(msg_dict.get("roll", 0.0))),
                "pitch": math.degrees(float(msg_dict.get("pitch", 0.0))),
                "yaw": (math.degrees(float(msg_dict.get("yaw", 0.0))) + 360.0) % 360.0
            }
            
        # If we have both recent position and attitude within 100ms, fuse into a TelemetrySample
        if self._last_position and self._last_attitude:
            time_diff = abs(self._last_position["timestamp_ms"] - self._last_attitude["timestamp_ms"])
            if time_diff <= 100.0:
                avg_time = (self._last_position["timestamp_ms"] + self._last_attitude["timestamp_ms"]) / 2.0
                sample = TelemetrySample(
                    timestamp_ms=avg_time,
                    lat=self._last_position["lat"],
                    lon=self._last_position["lon"],
                    altitude_agl=self._last_position["altitude_agl"],
                    pitch=self._last_attitude["pitch"],
                    roll=self._last_attitude["roll"],
                    yaw=self._last_attitude["yaw"]
                )
                self.parser.add_sample(sample)
