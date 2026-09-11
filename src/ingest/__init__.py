"""
src/ingest/__init__.py
======================
Module 1: Ingest and Telemetry Synchronization for Aerial SAR Vision Systems.
"""

from .telemetry_parser import TelemetryParser, TelemetrySample, MAVLinkTelemetryParser
from .video_sync import VideoSynchronizer, SynchronizedFrame
from .synthetic_telemetry import generate_telemetry_for_video

__all__ = [
    "TelemetryParser",
    "TelemetrySample",
    "MAVLinkTelemetryParser",
    "VideoSynchronizer",
    "SynchronizedFrame",
    "generate_telemetry_for_video",
]
