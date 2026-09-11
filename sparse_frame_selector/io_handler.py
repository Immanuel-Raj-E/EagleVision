"""
io_handler.py
=============
Manages disk I/O, writing original full-resolution 4K frames (RGB and Thermal),
generating the detailed metadata CSV, and formatting executive summary reports.
"""

from typing import Optional, Dict, Any, List
import os
import csv
import time
import logging
import cv2
import numpy as np

logger = logging.getLogger("SparseFrameSelector.IO")


class FrameWriter:
    """
    Handles saving original full-resolution 4K frames without compression artifacts.
    """

    def __init__(
        self,
        output_dir: str,
        is_dual_stream: bool = False,
        jpeg_quality: int = 95
    ):
        self.output_dir = output_dir
        self.is_dual_stream = is_dual_stream
        self.jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        
        self.rgb_dir = os.path.join(output_dir, "rgb") if is_dual_stream else output_dir
        self.thermal_dir = os.path.join(output_dir, "thermal") if is_dual_stream else None
        
        os.makedirs(self.rgb_dir, exist_ok=True)
        if self.thermal_dir:
            os.makedirs(self.thermal_dir, exist_ok=True)

    def save_frame(
        self,
        frame_id: int,
        frame_rgb: np.ndarray,
        frame_thermal: Optional[np.ndarray] = None
    ) -> str:
        """
        Saves full-resolution original frames to disk.
        
        Returns:
            str: Relative filename of saved RGB frame.
        """
        filename = f"frame_{frame_id:06d}.jpg"
        rgb_path = os.path.join(self.rgb_dir, filename)
        
        # Save RGB
        cv2.imwrite(rgb_path, frame_rgb, self.jpeg_params)
        
        # Save Thermal if present
        if self.thermal_dir and frame_thermal is not None:
            thermal_path = os.path.join(self.thermal_dir, filename)
            cv2.imwrite(thermal_path, frame_thermal, self.jpeg_params)
            
        return filename


class MetadataLogger:
    """
    Logs comprehensive per-selected-frame metrics and audit data to a CSV file.
    """

    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        self.headers = [
            "frame_id",
            "original_frame_number",
            "timestamp_seconds",
            "fps",
            "sharpness_score",
            "brightness_score",
            "contrast_score",
            "motion_score",
            "similarity_score",
            "selection_score",
            "reason_for_selection"
        ]
        
        os.makedirs(os.path.dirname(os.path.abspath(csv_filepath)), exist_ok=True)
        with open(self.csv_filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_frame(
        self,
        frame_id: int,
        original_frame_number: int,
        timestamp_seconds: float,
        fps: float,
        metrics: Dict[str, Any],
        reason: str
    ):
        with open(self.csv_filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"frame_{frame_id:06d}",
                original_frame_number,
                f"{timestamp_seconds:.3f}",
                f"{fps:.2f}",
                f"{metrics.get('sharpness', 0.0):.2f}",
                f"{metrics.get('brightness_score', 0.0):.4f}",
                f"{metrics.get('contrast_score', 0.0):.4f}",
                f"{metrics.get('motion_score', 0.0):.4f}",
                f"{metrics.get('similarity_score', 0.0):.4f}",
                f"{metrics.get('selection_score', 0.0):.4f}",
                reason
            ])


class SummaryReporter:
    """
    Tracks and prints execution statistics and reduction benchmarks.
    """

    def __init__(
        self,
        input_resolution: str,
        input_fps: float,
        total_input_frames: int,
        video_duration: float
    ):
        self.input_resolution = input_resolution
        self.input_fps = input_fps
        self.total_input_frames = total_input_frames
        self.video_duration = video_duration
        self.start_time = time.time()
        self.candidate_count = 0
        self.selected_count = 0

    def increment_candidates(self):
        self.candidate_count += 1

    def increment_selected(self):
        self.selected_count += 1

    def generate_summary(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time
        reduction_pct = 0.0
        if self.total_input_frames > 0:
            reduction_pct = (1.0 - (self.selected_count / self.total_input_frames)) * 100.0
            
        avg_selected_fps = (
            self.selected_count / self.video_duration if self.video_duration > 0 else 0.0
        )
        proc_fps = (
            self.total_input_frames / total_time if total_time > 0 else 0.0
        )
        
        return {
            "input_resolution": self.input_resolution,
            "input_fps": round(self.input_fps, 2),
            "video_duration_sec": round(self.video_duration, 2),
            "total_input_frames": self.total_input_frames,
            "candidate_frames": self.candidate_count,
            "selected_frames": self.selected_count,
            "reduction_percentage": round(reduction_pct, 2),
            "average_selected_fps": round(avg_selected_fps, 2),
            "processing_time_sec": round(total_time, 2),
            "processing_speed_fps": round(proc_fps, 1),
            "realtime_speedup": round(proc_fps / max(self.input_fps, 1.0), 2)
        }

    def print_summary(self):
        stats = self.generate_summary()
        summary_str = f"""
================================================================================
                    SPARSE FRAME SELECTOR EXECUTION SUMMARY
================================================================================
  Input Video Resolution   : {stats['input_resolution']}
  Input FPS                : {stats['input_fps']} FPS
  Video Duration           : {stats['video_duration_sec']} seconds
--------------------------------------------------------------------------------
  Total Input Frames       : {stats['total_input_frames']}
  Candidate Frames Tested  : {stats['candidate_frames']}
  Selected High-Value Frames: {stats['selected_frames']}
  Reduction Rate           : {stats['reduction_percentage']}% (Load reduction for detector)
  Effective Output FPS     : {stats['average_selected_fps']} FPS
--------------------------------------------------------------------------------
  Total Processing Time    : {stats['processing_time_sec']} seconds
  Processing Throughput    : {stats['processing_speed_fps']} input FPS ({stats['realtime_speedup']}x Real-time)
================================================================================
"""
        print(summary_str)
        return stats
