"""
main.py
=======
Command-line interface for the 4K Drone Sparse Frame Selector.
"""

import os
import sys
import argparse
import logging

from sparse_frame_selector.selector import SparseFrameSelector


def setup_logger(log_level_str: str = "INFO") -> logging.Logger:
    """Configures structured console logging with timestamp and level."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("SparseFrameSelector")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight, adaptive Sparse Frame Selector for 4K Drone Search and Rescue (SAR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument(
        "-i", "--input", "--video",
        dest="input_video",
        type=str,
        required=True,
        help="Path to input 4K RGB drone video file (MP4, AVI, MOV, MKV)."
    )
    parser.add_argument(
        "-t", "--thermal", "--thermal-video",
        dest="thermal_video",
        type=str,
        default=None,
        help="Optional path to synchronized Thermal drone video file."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_dir",
        type=str,
        default="selected_frames",
        help="Output directory to store original full-resolution selected frames."
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=str,
        default="selected_frames.csv",
        help="Path to output CSV file containing per-frame selection metadata."
    )
    
    # Sampling & Timing Parameters
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
        help="Initial candidate sampling frame rate (FPS) from input video."
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.2,
        help="Minimum time interval (seconds) required between selected frames."
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=2.0,
        help="Maximum time interval (seconds) before triggering a forced coverage keepalive."
    )
    
    # Quality & Threshold Parameters
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Laplacian variance threshold below which candidate frames are rejected as blurry."
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Similarity upper threshold (0.0 - 1.0) against last selected frame before rejection as redundant."
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.20,
        help="Inter-frame motion score threshold (0.0 - 1.0) for detecting significant scene activity."
    )
    parser.add_argument(
        "--proxy-width",
        type=int,
        default=640,
        help="Width (pixels) of proxy image used for metric computation (does not affect saved frame resolution)."
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality level (1-100) for saving full-resolution output frames."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity level."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    logger = setup_logger(args.log_level)
    
    if not os.path.exists(args.input_video):
        logger.error(f"Input video file not found: {args.input_video}")
        sys.exit(1)
        
    if args.thermal_video and not os.path.exists(args.thermal_video):
        logger.error(f"Thermal video file not found: {args.thermal_video}")
        sys.exit(1)
        
    selector = SparseFrameSelector(
        rgb_video_path=args.input_video,
        thermal_video_path=args.thermal_video,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        sample_fps=args.sample_fps,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        blur_threshold=args.blur_threshold,
        similarity_threshold=args.similarity_threshold,
        motion_threshold=args.motion_threshold,
        proxy_width=args.proxy_width,
        jpeg_quality=args.jpeg_quality
    )
    
    try:
        selector.run()
        logger.info(f"Processing complete! Frames saved to '{args.output_dir}' and metadata to '{args.csv_path}'")
    except Exception as e:
        logger.exception(f"Fatal error during frame selection: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
