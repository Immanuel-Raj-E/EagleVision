"""
src/tracking/__init__.py
========================
Module 3: Multi-Object Tracking, ByteTrack Kalman Filter, and Geospatial Deduplication for Aerial SAR.
"""

from .kalman_filter import KalmanBoxTracker
from .tracker import ByteTracker, TrackedSurvivor
from .deduplicator import SpatialDeduplicator, GroundEntity, haversine_distance_m

__all__ = [
    "KalmanBoxTracker",
    "ByteTracker",
    "TrackedSurvivor",
    "SpatialDeduplicator",
    "GroundEntity",
    "haversine_distance_m",
]
