"""
benchmark_latency_4k.py
=======================
Benchmark script for measuring 4K Aerial Video End-to-End Frame Processing Latency.
Tests 20 consecutive full-resolution 4K frames (3840 x 2160) on NVIDIA RTX 4060 GPU.
Audits slice generation time, batched FP16 inference, and NMS merging against the <= 300 ms limit.
"""

import sys
import time
import cv2
import numpy as np
import torch
import torchvision.ops

from src.detection.fusion_detector import FusionDetector
from src.ingest.video_sync import SynchronizedFrame


def create_synthetic_4k_test_frame() -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a realistic 4K (3840 x 2160) synthetic SAR scene with terrain texture,
    debris, human survivor targets, and thermal heatmap.
    """
    h, w = 2160, 3840
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:h//2, :] = [45, 75, 95]     # Flood water
    rgb[h//2:, :] = [60, 110, 50]    # Ground terrain
    
    # Texture noise
    noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
    rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    thermal = np.full((h, w), 55, dtype=np.uint8)
    
    # Survivor 1 (Human with high thermal contrast)
    x1, y1 = 1250, 750
    cv2.circle(rgb, (x1 + 30, y1 + 20), 18, (180, 150, 120), -1)
    cv2.rectangle(rgb, (x1 + 10, y1 + 40), (x1 + 50, y1 + 100), (0, 120, 255), -1)
    cv2.circle(thermal, (x1 + 30, y1 + 70), 45, 240, -1)
    
    # Survivor 2 (Human on rubble)
    x2, y2 = 2700, 1350
    cv2.circle(rgb, (x2 + 25, y2 + 20), 15, (170, 140, 110), -1)
    cv2.rectangle(rgb, (x2 + 10, y2 + 35), (x2 + 45, y2 + 85), (220, 220, 0), -1)
    cv2.circle(thermal, (x2 + 25, y2 + 55), 40, 230, -1)
    
    thermal_bgr = cv2.applyColorMap(thermal, cv2.COLORMAP_INFERNO)
    return rgb, thermal_bgr


def run_benchmark():
    print("=" * 88)
    print(" " * 18 + "EAGLEVISION 4K LATENCY BENCHMARK & AUDIT")
    print("=" * 88)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] Compute Device             : {device} ({device_name})")
    print(f"[*] PyTorch Version            : {torch.__version__}")
    print(f"[*] Target Resolution          : 3840 x 2160 (4K Aerial UHD)")
    print(f"[*] Competition Ceiling        : <= 300.0 ms per frame")
    print(f"[*] Optimization Target        : < 150.0 - 200.0 ms per frame")
    print("-" * 88)
    
    # 1. Initialize Optimized Detector
    print("[*] Initializing Optimized FusionDetector (FP16, 12-Slice Geometry)...")
    detector = FusionDetector(
        confidence_threshold=0.20,
        device=device,
        slice_height=960,
        slice_width=1280,
        overlap_height_ratio=0.12,
        overlap_width_ratio=0.12,
        perform_standard_pred=False,
        batch_size=12,
        imgsz=640,
        half=True
    )
    
    rgb_4k, thermal_4k = create_synthetic_4k_test_frame()
    slice_boxes = detector._get_slice_boxes(3840, 2160)
    print(f"[*] Total Slices per 4K Frame  : {len(slice_boxes)} slices (Down from 35+)")
    
    # 2. Warm-up passes
    print("[*] Running 5 CUDA Warm-up Passes on 4K imagery...")
    for w_i in range(5):
        _ = detector.detect(rgb_4k, thermal_image=thermal_4k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("    -> Warm-up complete.")
    
    # 3. Benchmark 20 consecutive full 4K frames with granular breakdown
    num_frames = 20
    print(f"\n[*] Commencing 20-Frame 4K Latency Benchmark...")
    
    slice_gen_times = []
    gpu_infer_times = []
    nms_merge_times = []
    total_frame_times = []
    detections_counts = []
    
    for f_idx in range(num_frames):
        # Synchronized Mock Frame
        mock_frame = SynchronizedFrame(
            frame_id=f_idx + 1,
            timestamp_ms=(f_idx * 33.33),
            rgb_image=rgb_4k,
            lat=27.71724 + (f_idx * 0.00001),
            lon=85.32402 + (f_idx * 0.00001),
            altitude_agl=60.0,
            pitch=-75.0,
            roll=0.0,
            yaw=90.0,
            is_synced=True
        )
        
        # Granular timing breakdown
        t_total_start = time.perf_counter()
        
        # A. Slice Generation
        t0 = time.perf_counter()
        slice_coords = detector._get_slice_boxes(3840, 2160)
        slices = [rgb_4k[y1:y2, x1:x2] for (x1, y1, x2, y2) in slice_coords]
        t1 = time.perf_counter()
        slice_ms = (t1 - t0) * 1000.0
        
        # B. Batched GPU Inference (FP16 Tensor Cores)
        with torch.inference_mode():
            if detector.device != "cpu":
                with torch.amp.autocast("cuda"):
                    batch_res = detector.model(
                        slices,
                        conf=detector.confidence_threshold,
                        verbose=False,
                        imgsz=detector.imgsz,
                        batch=detector.batch_size
                    )
            else:
                batch_res = detector.model(
                    slices,
                    conf=detector.confidence_threshold,
                    verbose=False,
                    imgsz=detector.imgsz,
                    batch=detector.batch_size
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        infer_ms = (t2 - t1) * 1000.0
        
        # C. Global NMS / Coordinate Merge
        all_boxes = []
        all_scores = []
        all_classes = []
        for s_i, (x1, y1, x2, y2) in enumerate(slice_coords):
            res = batch_res[s_i]
            boxes = res.boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy.clone()
                xyxy[:, [0, 2]] += x1
                xyxy[:, [1, 3]] += y1
                all_boxes.append(xyxy)
                all_scores.append(boxes.conf)
                all_classes.append(boxes.cls)
                
        if len(all_boxes) > 0:
            cat_b = torch.cat(all_boxes, dim=0)
            cat_s = torch.cat(all_scores, dim=0)
            cat_c = torch.cat(all_classes, dim=0)
            keep = torchvision.ops.batched_nms(cat_b, cat_s, cat_c, iou_threshold=0.45)
            final_b = cat_b[keep].cpu().numpy()
            det_count = len(final_b)
        else:
            det_count = 0
            
        t3 = time.perf_counter()
        merge_ms = (t3 - t2) * 1000.0
        
        # Full end-to-end call via detector.detect()
        res_full = detector.detect(mock_frame, thermal_image=thermal_4k)
        total_ms = res_full.inference_time_ms
        
        slice_gen_times.append(slice_ms)
        gpu_infer_times.append(infer_ms)
        nms_merge_times.append(merge_ms)
        total_frame_times.append(total_ms)
        detections_counts.append(len(res_full.detections))
        
        print(f"  * Frame #{f_idx+1:02d}: Total={total_ms:6.2f} ms | SliceGen={slice_ms:4.2f} ms | GPUInfer={infer_ms:5.2f} ms | NMSMerge={merge_ms:4.2f} ms | Detections={len(res_full.detections)}")

    # 4. Statistical Summary
    mean_slice = float(np.mean(slice_gen_times))
    mean_infer = float(np.mean(gpu_infer_times))
    mean_merge = float(np.mean(nms_merge_times))
    
    mean_total = float(np.mean(total_frame_times))
    min_total = float(np.min(total_frame_times))
    max_total = float(np.max(total_frame_times))
    std_total = float(np.std(total_frame_times))
    p95_total = float(np.percentile(total_frame_times, 95))
    fps_4k = 1000.0 / mean_total

    print("\n" + "=" * 88)
    print(" " * 22 + "4K LATENCY BENCHMARK BREAKDOWN TABLE")
    print("=" * 88)
    print(f"{'Pipeline Stage / Metric':<42} {'Measured Value':<24} {'Requirement / Threshold':<20}")
    print("-" * 88)
    print(f"{'Frame Resolution':<42} {'3840 x 2160 (4K UHD)':<24} {'4K Aerial Standard':<20}")
    print(f"{'Slices per Frame':<42} {f'{len(slice_boxes)} Slices (960x1280)':<24} {'<= 12 Slices':<20}")
    print(f"{'1. Slice Generation Latency':<42} {f'{mean_slice:.2f} ms':<24} {'< 5.0 ms':<20}")
    print(f"{'2. Batched FP16 GPU Inference':<42} {f'{mean_infer:.2f} ms':<24} {'< 150.0 ms':<20}")
    print(f"{'3. Coordinate Projection & NMS Merge':<42} {f'{mean_merge:.2f} ms':<24} {'< 5.0 ms':<20}")
    print("-" * 88)
    print(f"{'Mean End-to-End Latency':<42} {f'{mean_total:.2f} ms':<24} {'<= 200.0 ms (Target)':<20}")
    print(f"{'Min / Max Latency':<42} {f'{min_total:.2f} / {max_total:.2f} ms':<24} {'<= 300.0 ms (Ceiling)':<20}")
    print(f"{'95th Percentile Latency (P95)':<42} {f'{p95_total:.2f} ms':<24} {'<= 250.0 ms':<20}")
    print(f"{'Effective Full 4K Throughput':<42} {f'{fps_4k:.2f} FPS':<24} {'>= 5.0 FPS':<20}")
    print(f"{'Precision & Hardware Acceleration':<42} {'FP16 Tensor Cores':<24} {'CUDA 0':<20}")
    print("=" * 88)
    
    # Assertions
    assert mean_total <= 200.0, f"Mean latency ({mean_total:.2f} ms) exceeds 200 ms target!"
    assert max_total <= 300.0, f"Max latency ({max_total:.2f} ms) exceeds 300 ms competition threshold!"
    assert len(slice_boxes) <= 12, f"Slice count ({len(slice_boxes)}) exceeds 12 slices!"
    
    print("\n>>> ALL 4K LATENCY & THROUGHPUT ASSERTIONS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_benchmark()
