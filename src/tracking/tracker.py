"""
tracker.py
==========
ByteTrack Multi-Object Tracker for Aerial SAR Survivor Tracking and Deduplication.
Uses two-stage Hungarian association (IoU + GIoU) with Kalman filter state estimation.
Guarantees persistent single-ID assignment across temporary occlusions and low-score intervals.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.fusion_detector import DetectionResult, FrameDetections
from .kalman_filter import KalmanBoxTracker


@dataclass
class TrackedSurvivor:
    """Represents a unique tracked survivor / animal across video time."""
    track_id: int
    class_name: str
    current_bbox: list[float]              # [x1, y1, x2, y2] in 4K frame coords
    confidence: float
    history_centers: list[tuple[float, float]] = field(default_factory=list)
    is_confirmed: bool = False
    is_stationary: bool = True
    thermal_delta: Optional[float] = None
    status: str = "TENTATIVE"              # "CONFIRMED", "LOST", "TENTATIVE"
    stationary_frames: int = 0             # Frames where center moved < threshold
    hits: int = 1
    lost_frames: int = 0
    age: int = 1


def compute_giou_distance_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Computes Generalized IoU (GIoU) cost matrix between two sets of bounding boxes.
    Cost = (1.0 - GIoU) / 2.0  -> normalized in [0.0, 1.0].
    Provides smooth distance gradient even when boxes have zero intersection during occlusions.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # 1. Intersection Box
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    intersection = w * h

    # 2. Areas & Standard Union
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    iou = np.where(union > 0, intersection / union, 0.0)

    # 3. Smallest Enclosing Convex Hull Box
    cx1 = np.minimum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    cy1 = np.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    cx2 = np.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    cy2 = np.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    c_area = np.maximum(1e-6, (cx2 - cx1) * (cy2 - cy1))

    # 4. GIoU = IoU - (C_area - Union) / C_area
    giou = iou - ((c_area - union) / c_area)
    # Map GIoU [-1.0, 1.0] to Cost [0.0, 1.0] where 0.0 is identical
    cost_matrix = (1.0 - giou) / 2.0
    return cost_matrix.astype(np.float32)


class SingleTrack:
    """Internal state machine and Kalman tracker for an individual target."""

    _next_id = 1

    @classmethod
    def reset_id_counter(cls, start_id: int = 1):
        cls._next_id = start_id

    def __init__(self, detection: DetectionResult):
        self.track_id = SingleTrack._next_id
        SingleTrack._next_id += 1

        self.kalman = KalmanBoxTracker(detection.bbox_xyxy)
        self.class_name = detection.class_name
        self.class_id = detection.class_id
        self.confidence = detection.confidence
        self.thermal_delta = detection.thermal_delta
        
        self.current_bbox = list(detection.bbox_xyxy)
        center_x = (detection.bbox_xyxy[0] + detection.bbox_xyxy[2]) / 2.0
        center_y = (detection.bbox_xyxy[1] + detection.bbox_xyxy[3]) / 2.0
        self.history_centers: list[tuple[float, float]] = [(center_x, center_y)]

        self.hits = 1
        self.lost_frames = 0
        self.age = 1
        self.stationary_frames = 1
        self.is_confirmed = False
        self.status = "TENTATIVE"

    def predict(self) -> list[float]:
        """Predicts new bounding box location with Kalman filter."""
        self.current_bbox = self.kalman.predict()
        self.age += 1
        if self.status != "LOST":
            self.lost_frames += 1
        return self.current_bbox

    def update(self, detection: DetectionResult, stationary_dist_thresh: float = 20.0):
        """Updates track state with an observed matched detection."""
        self.kalman.update(detection.bbox_xyxy)
        self.current_bbox = self.kalman.get_state()
        self.confidence = detection.confidence
        self.class_name = detection.class_name
        if detection.thermal_delta is not None:
            self.thermal_delta = detection.thermal_delta

        # Center tracking & stationary displacement test
        new_cx = (self.current_bbox[0] + self.current_bbox[2]) / 2.0
        new_cy = (self.current_bbox[1] + self.current_bbox[3]) / 2.0
        
        if self.history_centers:
            prev_cx, prev_cy = self.history_centers[-1]
            dist = math.hypot(new_cx - prev_cx, new_cy - prev_cy)
            if dist < stationary_dist_thresh:
                self.stationary_frames += 1
            else:
                self.stationary_frames = max(0, self.stationary_frames - 1)
                
        self.history_centers.append((round(new_cx, 1), round(new_cy, 1)))
        if len(self.history_centers) > 60:
            self.history_centers.pop(0)

        self.hits += 1
        self.lost_frames = 0
        if self.hits >= 3:
            self.status = "CONFIRMED"
            self.is_confirmed = True

    def mark_lost(self):
        """Marks track as lost during missing frames."""
        self.status = "LOST"

    def to_dataclass(self) -> TrackedSurvivor:
        """Exports to public TrackedSurvivor format."""
        is_stationary = self.stationary_frames >= 2 or (len(self.history_centers) <= 2)
        return TrackedSurvivor(
            track_id=self.track_id,
            class_name=self.class_name,
            current_bbox=list(self.current_bbox),
            confidence=round(self.confidence, 3),
            history_centers=list(self.history_centers),
            is_confirmed=self.is_confirmed,
            is_stationary=is_stationary,
            thermal_delta=self.thermal_delta,
            status=self.status,
            stationary_frames=self.stationary_frames,
            hits=self.hits,
            lost_frames=self.lost_frames,
            age=self.age
        )


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker for Aerial SAR.
    Executes two-stage association over high and low confidence detection pools.
    """

    def __init__(
        self,
        high_thresh: float = 0.40,
        low_thresh: float = 0.15,
        match_thresh_stage1: float = 0.60,  # Max cost for stage 1 (GIoU)
        match_thresh_stage2: float = 0.50,  # Max cost for stage 2 (low-confidence recovery)
        min_hits: int = 3,
        max_lost_frames: int = 30,
        stationary_dist_thresh: float = 20.0
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh_stage1 = match_thresh_stage1
        self.match_thresh_stage2 = match_thresh_stage2
        self.min_hits = min_hits
        self.max_lost_frames = max_lost_frames
        self.stationary_dist_thresh = stationary_dist_thresh

        self.tracks: List[SingleTrack] = []
        self.frame_count = 0

    def reset(self):
        """Resets tracker state."""
        self.tracks.clear()
        self.frame_count = 0
        SingleTrack.reset_id_counter(1)

    def update(
        self,
        detections_input: Union[List[DetectionResult], FrameDetections, list],
        frame_id: Optional[int] = None,
        timestamp_ms: Optional[float] = None
    ) -> List[TrackedSurvivor]:
        """
        Updates multi-object tracker with new frame detections.
        """
        self.frame_count += 1
        
        # Unpack detections
        if isinstance(detections_input, FrameDetections):
            raw_dets = detections_input.detections
        elif isinstance(detections_input, list):
            raw_dets = []
            for d in detections_input:
                if isinstance(d, DetectionResult):
                    raw_dets.append(d)
                elif isinstance(d, (list, tuple)) and len(d) >= 4:
                    conf = float(d[4]) if len(d) > 4 else 0.5
                    cls_name = str(d[5]) if len(d) > 5 else "human"
                    raw_dets.append(DetectionResult(
                        bbox_xyxy=[float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                        confidence=conf,
                        class_name="human" if cls_name in ("0", "person", "human") else "animal",
                        class_id=0
                    ))
        else:
            raw_dets = []

        # 1. Kalman Predict for all active tracks
        for t in self.tracks:
            t.predict()

        # 2. Separate detection pools (ByteTrack strategy)
        dets_high = [d for d in raw_dets if d.confidence >= self.high_thresh]
        dets_low = [d for d in raw_dets if self.low_thresh <= d.confidence < self.high_thresh]

        # -------------------------------------------------------------
        # STAGE 1: Match Active Tracks with High-Confidence Detections
        # -------------------------------------------------------------
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_high = self._associate(
            self.tracks,
            dets_high,
            cost_threshold=self.match_thresh_stage1
        )

        for t_idx, d_idx in matched_tracks_1:
            self.tracks[t_idx].update(dets_high[d_idx], self.stationary_dist_thresh)

        # -------------------------------------------------------------
        # STAGE 2: Match Remaining Tracks with Low-Confidence Detections
        # (Recovers occluded survivors in water, mud, shadows)
        # -------------------------------------------------------------
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks_1]
        
        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            remaining_tracks,
            dets_low,
            cost_threshold=self.match_thresh_stage2
        )

        for r_idx, d_idx in matched_tracks_2:
            remaining_tracks[r_idx].update(dets_low[d_idx], self.stationary_dist_thresh)

        # Unmatched tracks from Stage 2 are marked as lost
        for r_idx in unmatched_tracks_2:
            remaining_tracks[r_idx].mark_lost()

        # -------------------------------------------------------------
        # STAGE 3: Initialize New Tracks from Unmatched High-Conf Dets
        # -------------------------------------------------------------
        for d_idx in unmatched_dets_high:
            new_track = SingleTrack(dets_high[d_idx])
            self.tracks.append(new_track)

        # -------------------------------------------------------------
        # STAGE 4: Track Lifecycle Management & Pruning
        # -------------------------------------------------------------
        active_tracks: List[SingleTrack] = []
        for t in self.tracks:
            if t.lost_frames > self.max_lost_frames:
                continue  # Delete permanently (REMOVED)
            active_tracks.append(t)
        self.tracks = active_tracks

        return [t.to_dataclass() for t in self.tracks if t.status != "REMOVED"]

    @staticmethod
    def _associate(
        tracks: List[SingleTrack],
        detections: List[DetectionResult],
        cost_threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Matches tracks with detections using Hungarian linear sum assignment over GIoU distance.
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.current_bbox for t in tracks], dtype=np.float32)
        det_boxes = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)

        cost_matrix = compute_giou_distance_matrix(track_boxes, det_boxes)

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] <= cost_threshold:
                matched_pairs.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched_pairs, list(unmatched_tracks), list(unmatched_dets)
