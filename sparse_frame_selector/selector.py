"""
selector.py
===========
Adaptive, lightweight Sparse Frame Selector engine for 4K Drone SAR.
Selects high-value, crisp, novel frames and discards blur, vibration artifacts, and redundant hovers.
"""

from typing import Optional, Dict, Any, List
import logging
import cv2
import numpy as np
from tqdm import tqdm

from .metrics import compute_candidate_metrics
from .synchronizer import DualStreamReader
from .io_handler import FrameWriter, MetadataLogger, SummaryReporter

logger = logging.getLogger("SparseFrameSelector.Engine")


class SparseFrameSelector:
    """
    Processes high-resolution video streams sequentially, evaluating candidates on lightweight
    proxy frames while preserving original full-resolution 4K frames for downstream detection.
    """

    def __init__(
        self,
        rgb_video_path: str,
        thermal_video_path: Optional[str] = None,
        output_dir: str = "selected_frames",
        csv_path: str = "selected_frames.csv",
        sample_fps: float = 5.0,
        min_gap: float = 0.2,
        max_gap: float = 2.0,
        blur_threshold: float = 100.0,
        similarity_threshold: float = 0.90,
        motion_threshold: float = 0.20,
        proxy_width: int = 640,
        jpeg_quality: int = 95
    ):
        self.rgb_video_path = rgb_video_path
        self.thermal_video_path = thermal_video_path
        self.output_dir = output_dir
        self.csv_path = csv_path
        
        self.sample_fps = sample_fps
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.blur_threshold = blur_threshold
        self.similarity_threshold = similarity_threshold
        self.motion_threshold = motion_threshold
        self.proxy_width = proxy_width
        
        # Initialize subcomponents
        self.reader = DualStreamReader(rgb_video_path, thermal_video_path)
        self.writer = FrameWriter(
            output_dir=output_dir,
            is_dual_stream=(thermal_video_path is not None),
            jpeg_quality=jpeg_quality
        )
        self.logger = MetadataLogger(csv_path)
        
        res_str = f"{self.reader.rgb_info['width']}x{self.reader.rgb_info['height']}"
        self.reporter = SummaryReporter(
            input_resolution=res_str,
            input_fps=self.reader.rgb_info['fps'],
            total_input_frames=self.reader.rgb_info['total_frames'],
            video_duration=self.reader.rgb_info['duration']
        )
        
        # Calculate sample interval in frames
        input_fps = self.reader.rgb_info['fps']
        self.sample_interval = max(1, int(round(input_fps / sample_fps)))

    def _create_proxy(self, full_frame: np.ndarray) -> np.ndarray:
        """
        Generates a fast downscaled proxy image for metric calculation without touching full-res memory.
        """
        h, w = full_frame.shape[:2]
        proxy_height = int(round(h * (self.proxy_width / float(w))))
        proxy_bgr = cv2.resize(
            full_frame,
            (self.proxy_width, proxy_height),
            interpolation=cv2.INTER_AREA
        )
        return proxy_bgr

    def run(self) -> Dict[str, Any]:
        """
        Executes sequential streaming frame selection.
        """
        logger.info("Initiating sparse frame selection pipeline...")
        logger.info(
            f"Config: Sample FPS={self.sample_fps} (interval={self.sample_interval}), "
            f"Min Gap={self.min_gap}s, Blur Thresh={self.blur_threshold}, "
            f"Similarity Thresh={self.similarity_threshold}, Proxy Width={self.proxy_width}px"
        )
        
        last_selected_proxy_bgr: Optional[np.ndarray] = None
        prev_candidate_proxy_gray: Optional[np.ndarray] = None
        last_selected_timestamp: float = -999.0
        
        selected_id = 1
        
        # Best candidate buffer for max_gap keep-alive fallback
        buffered_best_candidate = None
        
        total_candidates_est = max(1, self.reader.rgb_info['total_frames'] // self.sample_interval)
        pbar = tqdm(
            total=total_candidates_est,
            desc="Selecting 4K Frames",
            unit="cand",
            ncols=90
        )
        
        try:
            for frame_idx, timestamp_sec, frame_rgb, frame_thermal in self.reader.generate_candidate_frames(self.sample_interval):
                self.reporter.increment_candidates()
                pbar.update(1)
                
                # Create proxy images for low-latency metric computation
                proxy_bgr_curr = self._create_proxy(frame_rgb)
                proxy_gray_curr = cv2.cvtColor(proxy_bgr_curr, cv2.COLOR_BGR2GRAY)
                
                # Compute metrics
                metrics = compute_candidate_metrics(
                    proxy_bgr_curr=proxy_bgr_curr,
                    proxy_gray_curr=proxy_gray_curr,
                    proxy_gray_prev=prev_candidate_proxy_gray,
                    proxy_bgr_last_selected=last_selected_proxy_bgr,
                    blur_threshold=self.blur_threshold
                )
                
                time_since_last_sec = timestamp_sec - last_selected_timestamp
                
                # Evaluate selection rules
                select_frame = False
                selection_reason = ""
                
                # 1. First frame of the video
                if last_selected_proxy_bgr is None:
                    select_frame = True
                    selection_reason = "initial_video_anchor"
                    
                # 2. Hard constraint: Minimum gap
                elif time_since_last_sec < self.min_gap:
                    select_frame = False
                    
                # 3. Blur rejection
                elif metrics["is_blurry"]:
                    # Frame is degraded by vibration/motion blur.
                    # Track best candidate in case all frames in this window are blurry
                    if buffered_best_candidate is None or metrics["quality_score"] > buffered_best_candidate["metrics"]["quality_score"]:
                        buffered_best_candidate = {
                            "frame_idx": frame_idx,
                            "timestamp_sec": timestamp_sec,
                            "frame_rgb": frame_rgb,
                            "frame_thermal": frame_thermal,
                            "proxy_bgr": proxy_bgr_curr,
                            "metrics": metrics
                        }
                    select_frame = False
                    
                # 4. Motion / Dynamic scene change trigger
                elif metrics["motion_score"] >= self.motion_threshold and metrics["similarity_score"] < self.similarity_threshold:
                    select_frame = True
                    selection_reason = "significant_motion_and_scene_change"
                    
                # 5. Visual novelty trigger (camera translation/panning)
                elif metrics["similarity_score"] < self.similarity_threshold:
                    select_frame = True
                    selection_reason = "novel_scene_content"
                    
                # 6. Keep-alive Fallback (prevents starvation during slow hover)
                elif time_since_last_sec >= self.max_gap:
                    select_frame = True
                    selection_reason = "periodic_coverage_keepalive"
                    
                else:
                    # Redundant frame (high similarity, low motion)
                    # Maintain buffer in case we hit max_gap soon
                    if buffered_best_candidate is None or metrics["quality_score"] > buffered_best_candidate["metrics"]["quality_score"]:
                        buffered_best_candidate = {
                            "frame_idx": frame_idx,
                            "timestamp_sec": timestamp_sec,
                            "frame_rgb": frame_rgb,
                            "frame_thermal": frame_thermal,
                            "proxy_bgr": proxy_bgr_curr,
                            "metrics": metrics
                        }
                    select_frame = False
                    
                # Execute Selection
                if select_frame:
                    self.writer.save_frame(
                        frame_id=selected_id,
                        frame_rgb=frame_rgb,
                        frame_thermal=frame_thermal
                    )
                    self.logger.log_frame(
                        frame_id=selected_id,
                        original_frame_number=frame_idx,
                        timestamp_seconds=timestamp_sec,
                        fps=self.reader.rgb_info['fps'],
                        metrics=metrics,
                        reason=selection_reason
                    )
                    
                    last_selected_proxy_bgr = proxy_bgr_curr.copy()
                    last_selected_timestamp = timestamp_sec
                    selected_id += 1
                    self.reporter.increment_selected()
                    buffered_best_candidate = None  # Reset fallback buffer
                    
                # Update rolling previous candidate
                prev_candidate_proxy_gray = proxy_gray_curr.copy()
                
        finally:
            pbar.close()
            self.reader.release()
            
        summary_stats = self.reporter.print_summary()
        return summary_stats
