"""
src/detection/__init__.py
========================
Module 2: Detection and Fusion for Aerial SAR Vision Systems.
"""

from .fusion_detector import FusionDetector, DetectionResult, FrameDetections

__all__ = [
    "FusionDetector",
    "DetectionResult",
    "FrameDetections",
]
