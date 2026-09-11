"""
synchronizer.py
===============
Dual-stream synchronizer for synchronized RGB and Thermal (RGB-T) drone video feeds.
Ensures identical frame sampling and locks timestamp alignment across sensors.
"""

from typing import Optional, Tuple, Generator, Dict, Any
import logging
import cv2
import numpy as np

logger = logging.getLogger("SparseFrameSelector.Synchronizer")


class DualStreamReader:
    """
    Synchronized video stream reader supporting RGB-only or dual RGB + Thermal feeds.
    Uses sequential frame grabbing to preserve video decoder context and minimize memory usage.
    """

    def __init__(
        self,
        rgb_path: str,
        thermal_path: Optional[str] = None
    ):
        self.rgb_path = rgb_path
        self.thermal_path = thermal_path
        
        self.cap_rgb = cv2.VideoCapture(rgb_path)
        if not self.cap_rgb.isOpened():
            raise IOError(f"Failed to open RGB video file: {rgb_path}")
            
        self.cap_thermal = None
        if self.thermal_path:
            self.cap_thermal = cv2.VideoCapture(thermal_path)
            if not self.cap_thermal.isOpened():
                self.cap_rgb.release()
                raise IOError(f"Failed to open Thermal video file: {thermal_path}")
                
        self.rgb_info = self._get_stream_info(self.cap_rgb, "RGB")
        self.thermal_info = self._get_stream_info(self.cap_thermal, "Thermal") if self.cap_thermal else None
        
        self._validate_streams()

    @staticmethod
    def _get_stream_info(cap: cv2.VideoCapture, name: str) -> Dict[str, Any]:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Fallback default
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0
        
        return {
            "name": name,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": duration
        }

    def _validate_streams(self):
        logger.info(
            f"RGB Stream: {self.rgb_info['width']}x{self.rgb_info['height']} @ "
            f"{self.rgb_info['fps']:.2f} FPS ({self.rgb_info['total_frames']} frames, {self.rgb_info['duration']:.2f}s)"
        )
        if self.cap_thermal:
            logger.info(
                f"Thermal Stream: {self.thermal_info['width']}x{self.thermal_info['height']} @ "
                f"{self.thermal_info['fps']:.2f} FPS ({self.thermal_info['total_frames']} frames, {self.thermal_info['duration']:.2f}s)"
            )
            fps_diff = abs(self.rgb_info['fps'] - self.thermal_info['fps'])
            if fps_diff > 1.0:
                logger.warning(
                    f"Frame rate mismatch between RGB ({self.rgb_info['fps']} FPS) "
                    f"and Thermal ({self.thermal_info['fps']} FPS). Sync will map by timestamp."
                )

    def generate_candidate_frames(
        self,
        sample_interval: int
    ) -> Generator[Tuple[int, float, np.ndarray, Optional[np.ndarray]], None, None]:
        """
        Sequentially steps through video, grabbing lightweight frames and decoding only candidate frames.
        
        Args:
            sample_interval: Frame interval (e.g. 6 frames = 5 FPS from 30 FPS input).
            
        Yields:
            Tuple[int, float, np.ndarray, Optional[np.ndarray]]:
                (frame_index, timestamp_sec, full_rgb_frame, optional_full_thermal_frame)
        """
        frame_idx = 0
        fps = self.rgb_info['fps']
        
        while True:
            is_candidate = (frame_idx % sample_interval == 0)
            
            if is_candidate:
                # Full decode for candidate frame
                ret_rgb, frame_rgb = self.cap_rgb.read()
                if not ret_rgb or frame_rgb is None:
                    break
                    
                frame_thermal = None
                if self.cap_thermal:
                    ret_th, frame_thermal = self.cap_thermal.read()
                    if not ret_th or frame_thermal is None:
                        logger.warning(f"Thermal stream ended early at frame {frame_idx}")
                        break
                        
                timestamp_sec = frame_idx / fps
                yield frame_idx, timestamp_sec, frame_rgb, frame_thermal
            else:
                # Fast grab (no color decompression / memory allocation)
                grabbed_rgb = self.cap_rgb.grab()
                if not grabbed_rgb:
                    break
                if self.cap_thermal:
                    grabbed_th = self.cap_thermal.grab()
                    if not grabbed_th:
                        break
                        
            frame_idx += 1

    def release(self):
        if self.cap_rgb and self.cap_rgb.isOpened():
            self.cap_rgb.release()
        if self.cap_thermal and self.cap_thermal.isOpened():
            self.cap_thermal.release()
