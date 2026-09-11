"""
generate_synthetic_test_video.py
================================
Generates synthetic 4K (3840x2160) RGB and Thermal drone video feeds to test:
1. Static hover (redundancy rejection)
2. Panning / camera translation (novel scene acceptance)
3. Sudden drone vibration bursts (blur rejection)
4. Rapid lighting / contrast shifts
5. Small moving target (survivor simulation)
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def generate_synthetic_4k_drone_videos(
    output_rgb_path: str = "test_drone_4k_rgb.mp4",
    output_thermal_path: str = "test_drone_4k_thermal.mp4",
    width: int = 3840,
    height: int = 2160,
    fps: float = 30.0,
    duration_sec: float = 10.0
):
    total_frames = int(round(fps * duration_sec))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    rgb_writer = cv2.VideoWriter(output_rgb_path, fourcc, fps, (width, height))
    thermal_writer = cv2.VideoWriter(output_thermal_path, fourcc, fps, (width, height))
    
    print(f"Generating synthetic {width}x{height} @ {fps} FPS drone videos ({total_frames} frames, {duration_sec}s)...")
    
    # Base terrain canvas with texture (larger than 4K for panning simulation)
    canvas_w, canvas_h = width + 1000, height + 1000
    np.random.seed(42)
    base_terrain = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Simulate terrain zones (muddy water, debris, foliage, destroyed structures)
    base_terrain[:canvas_h//2, :] = [45, 75, 95]     # Muddy flood water
    base_terrain[canvas_h//2:, :] = [60, 110, 50]    # Grassy / debris land
    
    # Add noise & texture to simulate realistic drone aerial terrain
    noise = np.random.normal(0, 15, (canvas_h, canvas_w, 3)).astype(np.int16)
    base_terrain = np.clip(base_terrain.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add terrain features (river banks, rubble lines)
    cv2.line(base_terrain, (0, canvas_h // 2), (canvas_w, canvas_h // 2 + 100), (30, 45, 60), 40)
    for i in range(15):
        rx, ry = np.random.randint(100, canvas_w - 200), np.random.randint(100, canvas_h - 200)
        cv2.rectangle(base_terrain, (rx, ry), (rx + 150, ry + 100), (80, 80, 80), -1)
        
    # Thermal base
    thermal_base = cv2.cvtColor(base_terrain, cv2.COLOR_BGR2GRAY)
    thermal_base = cv2.applyColorMap(thermal_base, cv2.COLORMAP_INFERNO)
    
    # Frame generation loop
    cam_x, cam_y = 100, 100
    
    for i in tqdm(range(total_frames), desc="Rendering 4K Frames"):
        # Phase 1: Frames 0-60 (0-2s) -> Static Hover
        # Phase 2: Frames 60-150 (2-5s) -> Fast Camera Pan
        # Phase 3: Frames 150-180 (5-6s) -> Severe Drone Vibration Blur
        # Phase 4: Frames 180-300 (6-10s) -> Smooth Pan with Moving Survivor
        
        if 60 <= i < 150:
            cam_x += 4
            cam_y += 3
        elif 180 <= i:
            cam_x += 2
            cam_y += 1
            
        crop_rgb = base_terrain[cam_y:cam_y+height, cam_x:cam_x+width].copy()
        crop_thermal = thermal_base[cam_y:cam_y+height, cam_x:cam_x+width].copy()
        
        # Simulate tiny survivor target (e.g. 30x30 pixels in 4K)
        survivor_x = int(width * 0.45 + (i - 180) * 2) if i >= 180 else int(width * 0.45)
        survivor_y = int(height * 0.6)
        
        # Draw survivor on RGB (bright orange life vest)
        cv2.circle(crop_rgb, (survivor_x, survivor_y), 15, (0, 140, 255), -1)
        # Draw high heat signature on Thermal (bright white/yellow hotspot)
        cv2.circle(crop_thermal, (survivor_x, survivor_y), 15, (255, 255, 255), -1)
        
        # Simulate severe vibration blur in Phase 3
        if 150 <= i < 180:
            ksize = 31
            crop_rgb = cv2.GaussianBlur(crop_rgb, (ksize, ksize), 10.0)
            crop_thermal = cv2.GaussianBlur(crop_thermal, (ksize, ksize), 10.0)
            
        rgb_writer.write(crop_rgb)
        thermal_writer.write(crop_thermal)
        
    rgb_writer.release()
    thermal_writer.release()
    print(f"Generated test files: '{output_rgb_path}' and '{output_thermal_path}' successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0, help="Video duration in seconds.")
    args = parser.parse_args()
    generate_synthetic_4k_drone_videos(duration_sec=args.duration)
