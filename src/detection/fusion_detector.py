"""
fusion_detector.py
==================
SAHI + YOLOv8 Sliced Inference with Multi-Modal RGB-T Thermal Fusion and Fallback.
Optimized for tiny survivor targets (human, animal) in full-resolution 4K aerial imagery.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import os
import time
import logging
import cv2
import numpy as np
import torch

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from src.ingest.video_sync import SynchronizedFrame

logger = logging.getLogger("Detection.Fusion")


@dataclass(slots=True)
class DetectionResult:
    """Individual target detection in full 4K frame space."""
    bbox_xyxy: list[float]       # [x1, y1, x2, y2] in original 4K coordinates
    confidence: float
    class_name: str              # "human" or "animal"
    class_id: int
    thermal_delta: Optional[float] = None  # Estimated thermal contrast delta if available


@dataclass(slots=True)
class FrameDetections:
    """Unified container for full-frame detections paired with telemetry metadata."""
    frame_id: int
    timestamp_ms: float
    detections: list[DetectionResult]
    is_synced: bool
    lat: float
    lon: float
    altitude_agl: float
    pitch: float
    roll: float
    yaw: float
    thermal_available: bool
    inference_time_ms: float


class FusionDetector:
    """
    Wraps YOLOv8 with SAHI for high-altitude 4K aerial detection.
    Fuses thermal heat signatures when available and gracefully falls back to RGB-only.
    """

    # COCO Class mappings for Aerial SAR
    # 0: person -> "human"
    # 14-23: animal classes -> "animal"
    ANIMAL_COCO_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    DEFAULT_SAR_MODEL = r"d:\SEC\runs\sar_finetune\yolov8s_sar\weights\best.pt"

    def __init__(
        self,
        model_path: str = DEFAULT_SAR_MODEL if os.path.exists(DEFAULT_SAR_MODEL) else ("yolov8s.pt" if os.path.exists("yolov8s.pt") else "yolov8n.pt"),
        confidence_threshold: float = 0.25,
        device: str = "cuda:0",
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        perform_standard_pred: bool = True
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.perform_standard_pred = perform_standard_pred
        
        self._warned_thermal_dropout = False

        logger.info(f"Initializing SAHI AutoDetectionModel with '{model_path}' on {self.device}...")
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=self.device
        )
        logger.info("SAHI YOLOv8 Detection Engine successfully initialized.")

    def _map_category(self, cat_id: int, cat_name: str) -> tuple[str, int]:
        """Maps model category ID / name to SAR classes ('human' or 'animal')."""
        if cat_id == 0 or cat_name.lower() in ("person", "human"):
            return "human", 0
        elif cat_id == 1 or cat_id in self.ANIMAL_COCO_IDS or any(a in cat_name.lower() for a in ("animal", "dog", "cat", "horse", "cow", "sheep", "bird")):
            return "animal", 1
        return cat_name.lower(), cat_id

    def _compute_thermal_delta(
        self,
        thermal_gray: np.ndarray,
        bbox: list[float]
    ) -> Optional[float]:
        """
        Computes the thermal contrast delta: mean temperature/intensity of bbox
        relative to the local surrounding background.
        """
        h, w = thermal_gray.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        
        # Clamp to bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = thermal_gray[y1:y2, x1:x2]
        roi_mean = float(np.mean(roi))
        
        # Expanded local neighborhood for background subtraction
        pad_x = max(10, (x2 - x1) // 2)
        pad_y = max(10, (y2 - y1) // 2)
        
        bx1, by1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        bx2, by2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        
        bg_region = thermal_gray[by1:by2, bx1:bx2]
        bg_mean = float(np.mean(bg_region))
        
        delta = roi_mean - bg_mean
        return round(delta, 2)

    def detect(
        self,
        frame_input: Union[SynchronizedFrame, np.ndarray],
        thermal_image: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
        is_synced: Optional[bool] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        altitude_agl: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        yaw: Optional[float] = None
    ) -> FrameDetections:
        """
        Executes SAHI sliced detection on RGB frames and evaluates thermal heat signatures.
        """
        start_time = time.perf_counter()
        
        # Extract RGB image and telemetry attributes
        if isinstance(frame_input, SynchronizedFrame):
            rgb_image = frame_input.rgb_image
            f_id = frame_id if frame_id is not None else frame_input.frame_id
            ts_ms = timestamp_ms if timestamp_ms is not None else frame_input.timestamp_ms
            synced = is_synced if is_synced is not None else frame_input.is_synced
            f_lat = lat if lat is not None else frame_input.lat
            f_lon = lon if lon is not None else frame_input.lon
            f_alt = altitude_agl if altitude_agl is not None else frame_input.altitude_agl
            f_pitch = pitch if pitch is not None else frame_input.pitch
            f_roll = roll if roll is not None else frame_input.roll
            f_yaw = yaw if yaw is not None else frame_input.yaw
        else:
            rgb_image = frame_input
            f_id = frame_id if frame_id is not None else 0
            ts_ms = timestamp_ms if timestamp_ms is not None else 0.0
            synced = is_synced if is_synced is not None else False
            f_lat = lat if lat is not None else 0.0
            f_lon = lon if lon is not None else 0.0
            f_alt = altitude_agl if altitude_agl is not None else 0.0
            f_pitch = pitch if pitch is not None else 0.0
            f_roll = roll if roll is not None else 0.0
            f_yaw = yaw if yaw is not None else 0.0

        # Validate thermal feed status
        has_valid_thermal = False
        thermal_gray = None
        if thermal_image is not None:
            if np.count_nonzero(thermal_image) > 0:
                has_valid_thermal = True
                if len(thermal_image.shape) == 3:
                    thermal_gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
                else:
                    thermal_gray = thermal_image
            else:
                if not self._warned_thermal_dropout:
                    logger.warning("Thermal stream contains all-zero frames (sensor dropout). Falling back to RGB-only.")
                    self._warned_thermal_dropout = True
        else:
            if not self._warned_thermal_dropout:
                logger.info("No thermal stream supplied. Operating in RGB-only detection mode.")
                self._warned_thermal_dropout = True

        # Run SAHI Sliced Inference
        sliced_result = get_sliced_prediction(
            image=rgb_image,
            detection_model=self.model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            perform_standard_pred=self.perform_standard_pred,
            verbose=0
        )

        detections: List[DetectionResult] = []
        for obj in sliced_result.object_prediction_list:
            cat_name, cat_id = self._map_category(obj.category.id, obj.category.name)
            
            # Filter for human / animal classes
            if cat_name not in ("human", "animal"):
                continue
                
            bbox_xyxy = [
                float(obj.bbox.minx),
                float(obj.bbox.miny),
                float(obj.bbox.maxx),
                float(obj.bbox.maxy)
            ]
            
            # Thermal fusion delta
            thermal_delta = None
            if has_valid_thermal and thermal_gray is not None:
                thermal_delta = self._compute_thermal_delta(thermal_gray, bbox_xyxy)
                
            detections.append(
                DetectionResult(
                    bbox_xyxy=bbox_xyxy,
                    confidence=round(float(obj.score.value), 3),
                    class_name=cat_name,
                    class_id=cat_id,
                    thermal_delta=thermal_delta
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return FrameDetections(
            frame_id=f_id,
            timestamp_ms=ts_ms,
            detections=detections,
            is_synced=synced,
            lat=f_lat,
            lon=f_lon,
            altitude_agl=f_alt,
            pitch=f_pitch,
            roll=f_roll,
            yaw=f_yaw,
            thermal_available=has_valid_thermal,
            inference_time_ms=round(elapsed_ms, 2)
        )
