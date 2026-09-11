"""
src/tracking/__init__.py
========================
Module 3: Multi-Object Tracking and Temporal Deduplication for Aerial SAR Vision.
"""

from .kalman_filter import KalmanBoxTracker
from .tracker import ByteTracker, TrackedSurvivor

__all__ = [
    "KalmanBoxTracker",
    "ByteTracker",
    "TrackedSurvivor",
]
