"""
fusion_detector.py
==================
High-Throughput Batched Sliced Inference with Multi-Modal RGB-T Thermal Fusion.
Optimized for tiny survivor targets (human, animal) in full-resolution 4K aerial imagery.
Leverages Tensor Cores (FP16), optimized slice geometry, and batched GPU execution on NVIDIA RTX GPUs.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple
import os
import time
import logging
import cv2
import numpy as np
import torch
import torchvision.ops
from ultralytics import YOLO

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
    High-Performance GPU Sliced Detection Engine for Aerial SAR.
    Fuses thermal heat signatures when available and gracefully falls back to RGB-only.
    Optimized for sub-150ms 4K inference on NVIDIA RTX 40-series GPUs.
    """

    # COCO / Custom Class mappings for Aerial SAR
    # 0: person -> "human"
    # 1: animal (or COCO IDs 14..23) -> "animal"
    ANIMAL_COCO_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
    DEFAULT_SAR_MODEL = r"d:\SEC\runs\sar_finetune\yolov8s_sar\weights\best.pt"

    def __init__(
        self,
        model_path: str = DEFAULT_SAR_MODEL if os.path.exists(DEFAULT_SAR_MODEL) else ("yolov8s.pt" if os.path.exists("yolov8s.pt") else "yolov8n.pt"),
        confidence_threshold: float = 0.25,
        device: str = "cuda:0",
        slice_height: int = 960,
        slice_width: int = 1280,
        overlap_height_ratio: float = 0.12,
        overlap_width_ratio: float = 0.12,
        perform_standard_pred: bool = False,
        batch_size: int = 12,
        imgsz: int = 640,
        half: bool = True
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device if (torch.cuda.is_available() and "cuda" in device) else "cpu"
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.perform_standard_pred = perform_standard_pred
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.half = half and (self.device != "cpu")
        
        self._warned_thermal_dropout = False
        self._slice_cache: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}

        logger.info(f"Initializing Fast Sliced YOLOv8 with '{model_path}' on {self.device} (FP16={self.half})...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        if self.device != "cpu":
            try:
                self.model.fuse()
            except Exception:
                pass
            if self.half:
                try:
                    self.model.model.half()
                except Exception as e:
                    logger.warning(f"Could not convert model to half precision: {e}")
                    self.half = False

        logger.info("High-Throughput Sliced YOLOv8 Detection Engine successfully initialized.")

    def _get_slice_boxes(self, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        """Computes and caches overlapping slice window coordinates [x1, y1, x2, y2]."""
        cache_key = (img_w, img_h)
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        sw = min(self.slice_width, img_w)
        sh = min(self.slice_height, img_h)
        
        step_x = max(1, int(sw * (1.0 - self.overlap_width_ratio)))
        step_y = max(1, int(sh * (1.0 - self.overlap_height_ratio)))
        
        x_starts = list(range(0, img_w - sw + 1, step_x))
        if len(x_starts) == 0 or (x_starts[-1] + sw < img_w):
            x_starts.append(img_w - sw)
            
        y_starts = list(range(0, img_h - sh + 1, step_y))
        if len(y_starts) == 0 or (y_starts[-1] + sh < img_h):
            y_starts.append(img_h - sh)
            
        boxes = []
        for y in y_starts:
            for x in x_starts:
                boxes.append((x, y, min(x + sw, img_w), min(y + sh, img_h)))
                
        self._slice_cache[cache_key] = boxes
        return boxes

    def _map_category(self, cat_id: int, cat_name: str) -> tuple[str, int]:
        """Maps model category ID / name to SAR classes ('human' or 'animal')."""
        name_lower = str(cat_name).lower()
        if cat_id == 0 or name_lower in ("person", "human", "survivor"):
            return "human", 0
        elif cat_id == 1 or cat_id in self.ANIMAL_COCO_IDS or any(a in name_lower for a in ("animal", "dog", "cat", "horse", "cow", "sheep", "bird")):
            return "animal", 1
        return name_lower, cat_id

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
        Executes fast batched sliced detection on RGB frames and evaluates thermal heat signatures.
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

        img_h, img_w = rgb_image.shape[:2]

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

        # 1. Compute Slice Windows
        slice_boxes = self._get_slice_boxes(img_w, img_h)
        slices = [rgb_image[y1:y2, x1:x2] for (x1, y1, x2, y2) in slice_boxes]
        
        # 2. Batched Sliced Inference
        with torch.inference_mode():
            if self.device != "cpu":
                with torch.amp.autocast("cuda"):
                    batch_results = self.model(
                        slices,
                        conf=self.confidence_threshold,
                        verbose=False,
                        imgsz=self.imgsz,
                        batch=self.batch_size
                    )
            else:
                batch_results = self.model(
                    slices,
                    conf=self.confidence_threshold,
                    verbose=False,
                    imgsz=self.imgsz,
                    batch=self.batch_size
                )

        # 3. Coordinate Projection and Global NMS
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for idx, (x1, y1, x2, y2) in enumerate(slice_boxes):
            res = batch_results[idx]
            boxes = res.boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.clone()
                xyxy[:, [0, 2]] += x1
                xyxy[:, [1, 3]] += y1
                all_boxes.append(xyxy)
                all_scores.append(boxes.conf)
                all_classes.append(boxes.cls)

        # Standard full-frame downscaled prediction (if enabled)
        if self.perform_standard_pred:
            with torch.inference_mode():
                if self.device != "cpu":
                    with torch.amp.autocast("cuda"):
                        std_res = self.model(rgb_image, conf=self.confidence_threshold, verbose=False, imgsz=self.imgsz)
                else:
                    std_res = self.model(rgb_image, conf=self.confidence_threshold, verbose=False, imgsz=self.imgsz)
            std_boxes = std_res[0].boxes
            if len(std_boxes) > 0:
                all_boxes.append(std_boxes.xyxy.clone())
                all_scores.append(std_boxes.conf)
                all_classes.append(std_boxes.cls)

        detections: List[DetectionResult] = []
        names_dict = self.model.names if hasattr(self.model, "names") else {}

        if len(all_boxes) > 0:
            cat_boxes = torch.cat(all_boxes, dim=0)
            cat_scores = torch.cat(all_scores, dim=0)
            cat_classes = torch.cat(all_classes, dim=0)
            
            keep = torchvision.ops.batched_nms(cat_boxes, cat_scores, cat_classes, iou_threshold=0.45)
            final_boxes = cat_boxes[keep].cpu().numpy()
            final_scores = cat_scores[keep].cpu().numpy()
            final_classes = cat_classes[keep].cpu().numpy().astype(int)

            for b_idx in range(len(final_boxes)):
                c_id = int(final_classes[b_idx])
                raw_name = names_dict.get(c_id, str(c_id))
                cat_name, target_id = self._map_category(c_id, raw_name)
                
                # Filter for human / animal classes
                if cat_name not in ("human", "animal"):
                    continue
                    
                bbox_xyxy = [
                    float(np.clip(final_boxes[b_idx][0], 0, img_w)),
                    float(np.clip(final_boxes[b_idx][1], 0, img_h)),
                    float(np.clip(final_boxes[b_idx][2], 0, img_w)),
                    float(np.clip(final_boxes[b_idx][3], 0, img_h))
                ]
                
                # Compute thermal contrast delta
                thermal_delta = None
                if has_valid_thermal and thermal_gray is not None:
                    thermal_delta = self._compute_thermal_delta(thermal_gray, bbox_xyxy)

                detections.append(
                    DetectionResult(
                        bbox_xyxy=bbox_xyxy,
                        confidence=round(float(final_scores[b_idx]), 3),
                        class_name=cat_name,
                        class_id=target_id,
                        thermal_delta=thermal_delta
                    )
                )

        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.synchronize()
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
