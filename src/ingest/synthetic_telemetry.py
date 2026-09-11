"""
synthetic_telemetry.py
======================
Generates realistic, synchronized aerial flight telemetry logs for any input drone video.
Simulates GPS coordinates, barometric/laser altitude AGL, and gimbal orientation.
"""

from typing import Optional
import os
import csv
import math
import argparse
import logging
import cv2
import numpy as np

logger = logging.getLogger("Ingest.SyntheticTelemetry")


def generate_telemetry_for_video(
    video_path: str,
    output_csv_path: str = "data/flight_telemetry.csv",
    sample_rate_hz: float = 10.0,
    base_lat: float = 27.7172,
    base_lon: float = 85.3240,
    base_alt_agl: float = 50.0,
    speed_mps: float = 8.0,
    heading_deg: float = 90.0,
    gimbal_pitch_deg: float = -75.0,
    add_jitter: bool = True
) -> dict:
    """
    Analyzes an input video's duration and frame rate to generate a matched
    10 Hz flight telemetry CSV log.
    
    Args:
        video_path: Path to target drone video file.
        output_csv_path: Destination path for the generated telemetry CSV.
        sample_rate_hz: Frequency of telemetry observations (default 10 Hz / 100ms).
        base_lat: Starting latitude in decimal degrees (e.g. Kathmandu flood zone).
        base_lon: Starting longitude in decimal degrees.
        base_alt_agl: Mean survey altitude Above Ground Level (meters).
        speed_mps: Drone ground speed in meters per second.
        heading_deg: Drone flight heading in degrees (0=North, 90=East).
        gimbal_pitch_deg: Gimbal look angle (-90=Nadir, -75=Near-Nadir).
        add_jitter: Whether to add realistic +/- 8ms packet transmission jitter.
        
    Returns:
        dict: Metadata summary of the generated telemetry log.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: '{video_path}'")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: '{video_path}'")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    # Calculate sampling timeline
    interval_ms = 1000.0 / sample_rate_hz
    # Add an extra 1.0 second buffer beyond video duration to prevent starvation at end of file
    total_samples = int(math.ceil((duration_sec + 1.0) * sample_rate_hz)) + 1
    
    # Earth radius approximation for meter-to-degree conversion
    meters_per_lat_deg = 111139.0
    meters_per_lon_deg = 111139.0 * math.cos(math.radians(base_lat))
    
    heading_rad = math.radians(heading_deg)
    velocity_north = speed_mps * math.cos(heading_rad)
    velocity_east = speed_mps * math.sin(heading_rad)

    np.random.seed(42)

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "lat", "lon", "altitude_agl", "pitch", "roll", "yaw"])
        
        for idx in range(total_samples):
            nominal_time_ms = idx * interval_ms
            
            # Simulated telemetry packet arrival jitter (except t=0)
            jitter_ms = np.random.uniform(-8.0, 8.0) if (add_jitter and idx > 0) else 0.0
            actual_time_ms = max(0.0, nominal_time_ms + jitter_ms)
            
            t_sec = actual_time_ms / 1000.0
            
            # Position along trajectory
            distance_north = velocity_north * t_sec
            distance_east = velocity_east * t_sec
            
            lat = base_lat + (distance_north / meters_per_lat_deg)
            lon = base_lon + (distance_east / meters_per_lon_deg)
            
            # Altitude with slight atmospheric/wind oscillation (+/- 1.5m)
            altitude = base_alt_agl + 1.2 * math.sin(0.4 * t_sec) + 0.3 * math.cos(1.1 * t_sec)
            
            # Gimbal orientation: Near-nadir (-75 deg) with slight stabilization tremor
            gimbal_pitch = gimbal_pitch_deg + 0.6 * math.sin(0.8 * t_sec)
            gimbal_roll = 0.0 + 0.4 * math.cos(0.9 * t_sec)
            gimbal_yaw = (heading_deg + 1.0 * math.sin(0.2 * t_sec)) % 360.0
            
            writer.writerow([
                f"{actual_time_ms:.2f}",
                f"{lat:.7f}",
                f"{lon:.7f}",
                f"{altitude:.2f}",
                f"{gimbal_pitch:.2f}",
                f"{gimbal_roll:.2f}",
                f"{gimbal_yaw:.2f}"
            ])

    logger.info(
        f"Generated {total_samples} telemetry samples for '{video_path}' "
        f"({duration_sec:.2f}s, {fps:.1f} FPS) -> '{output_csv_path}'"
    )

    return {
        "video_path": video_path,
        "video_resolution": f"{width}x{height}",
        "video_fps": round(fps, 2),
        "video_duration_sec": round(duration_sec, 2),
        "telemetry_samples": total_samples,
        "telemetry_csv_path": output_csv_path,
        "sample_rate_hz": sample_rate_hz,
        "base_coordinates": (base_lat, base_lon),
        "base_altitude_m": base_alt_agl
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic synchronized drone flight telemetry CSV for video feeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", dest="video_path", type=str, required=True, help="Input video file path.")
    parser.add_argument("-o", "--output", dest="output_csv", type=str, default="data/flight_telemetry.csv", help="Output CSV path.")
    parser.add_argument("--rate", type=float, default=10.0, help="Telemetry sample rate in Hz.")
    parser.add_argument("--alt", type=float, default=50.0, help="Base altitude AGL in meters.")
    parser.add_argument("--lat", type=float, default=27.7172, help="Starting latitude.")
    parser.add_argument("--lon", type=float, default=85.3240, help="Starting longitude.")
    parser.add_argument("--speed", type=float, default=8.0, help="Drone flight speed in m/s.")
    parser.add_argument("--pitch", type=float, default=-75.0, help="Gimbal pitch angle in degrees.")
    args = parser.parse_args()

    meta = generate_telemetry_for_video(
        video_path=args.video_path,
        output_csv_path=args.output_csv,
        sample_rate_hz=args.rate,
        base_lat=args.lat,
        base_lon=args.lon,
        base_alt_agl=args.alt,
        speed_mps=args.speed,
        gimbal_pitch_deg=args.pitch
    )

    print("\n" + "=" * 70)
    print("        SYNTHETIC FLIGHT TELEMETRY GENERATION REPORT")
    print("=" * 70)
    print(f"  Input Video File         : {meta['video_path']}")
    print(f"  Video Duration / FPS     : {meta['video_duration_sec']}s @ {meta['video_fps']} FPS ({meta['video_resolution']})")
    print(f"  Generated Telemetry CSV  : {meta['telemetry_csv_path']}")
    print(f"  Total Telemetry Samples  : {meta['telemetry_samples']} samples @ {meta['sample_rate_hz']} Hz")
    print(f"  Base Survey Coordinates  : Lat {meta['base_coordinates'][0]}, Lon {meta['base_coordinates'][1]}")
    print(f"  Survey Altitude AGL      : {meta['base_altitude_m']} meters")
    print("=" * 70)
    print(">>> SUCCESS: Telemetry CSV generated and ready for synchronization! <<<\n")


if __name__ == "__main__":
    main()
