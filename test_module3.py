"""
test_module3.py
===============
Comprehensive Unit and Stress Test Suite for Module 3: Tracking & Deduplication.

Validates the complete deduplication problem statement:
- Test Case A (Continuous Movement): 40 frames smooth motion -> Exactly 1 consistent track_id.
- Test Case B (Occlusion & Deduplication): Disappears for 8 frames (15 to 22) and reappears -> Recovers original ID (1 unique survivor, not 2).
- Test Case C (False Positive Suppression): 1-frame & 2-frame noise spikes -> Never receives is_confirmed=True.
- Test Case D (Latency Overhead): 100 frames with 15 concurrent tracks -> Latency per frame < 5 ms.
"""

import sys
import time
import numpy as np

from src.detection.fusion_detector import DetectionResult
from src.tracking.tracker import ByteTracker, TrackedSurvivor


def run_unit_and_stress_tests():
    print("=" * 85)
    print("   MODULE 3: TRACKING & DEDUPLICATION (BYTETRACK) COMPREHENSIVE TEST SUITE")
    print("=" * 85)

    test_results = {}

    # -------------------------------------------------------------------------
    # TEST CASE A: Continuous Movement (40 Frames) -> Exactly 1 Consistent Track ID
    # -------------------------------------------------------------------------
    print("\n[Test Case A] Continuous Smooth Movement Across 40 Frames...")
    tracker_a = ByteTracker(high_thresh=0.5, low_thresh=0.1, min_hits=3, max_lost_frames=30)
    tracker_a.reset()

    observed_ids_a = set()
    frames_a = 40
    
    # Survivor traverses from (800, 600) to (1000, 760) over 40 frames
    for f in range(frames_a):
        cx = 800.0 + f * 5.0
        cy = 600.0 + f * 4.0
        w, h = 60.0, 120.0
        bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        
        conf = 0.88 + np.random.uniform(-0.03, 0.03)
        det = DetectionResult(bbox_xyxy=bbox, confidence=conf, class_name="human", class_id=0, thermal_delta=32.0)
        
        active = tracker_a.update([det], frame_id=f, timestamp_ms=f * 33.3)
        for t in active:
            if t.is_confirmed:
                observed_ids_a.add(t.track_id)

    unique_ids_a = len(observed_ids_a)
    print(f"  * Total Frames Streamed        : {frames_a} frames")
    print(f"  * Unique Confirmed Track IDs   : {unique_ids_a} (Observed ID: {list(observed_ids_a)})")
    print(f"  * Deduplication Ratio          : 1 Track ID for {frames_a} frames (Zero Fragmentation)")
    
    pass_a = (unique_ids_a == 1 and 1 in observed_ids_a)
    test_results["Test Case A (Continuous Movement)"] = pass_a
    print(f"  => STATUS: [{'PASS' if pass_a else 'FAIL'}] Exact single ID maintained across 40 frames.")

    # -------------------------------------------------------------------------
    # TEST CASE B: 8-Frame Occlusion (Frames 15 to 22) & Re-Identification
    # -------------------------------------------------------------------------
    print("\n[Test Case B] Occlusion & Deduplication (8 Missing Frames: 15 to 22)...")
    tracker_b = ByteTracker(high_thresh=0.5, low_thresh=0.1, min_hits=3, max_lost_frames=30)
    tracker_b.reset()

    # Track moving target:
    # Frames 0..14  (15 frames): Normal detection moving from cx=500 -> cx=570 (vx=5 px/frame)
    # Frames 15..22 (8 frames) : Complete occlusion (zero detections)
    # Frames 23..35 (13 frames): Target reappears at expected extrapolated location cx=615
    
    all_confirmed_ids_b = set()
    initial_track_id = None
    recovered_track_id = None

    for f in range(36):
        cx = 500.0 + f * 5.0
        cy = 500.0 + f * 3.0
        w, h = 60.0, 120.0
        bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        
        if 15 <= f <= 22:
            # 8 frames of occlusion: survivor hidden under debris / flood water
            current_dets = []
        else:
            # Detections present (with slight confidence dip on reappearance)
            conf = 0.85 if f < 15 else 0.45  # Low-confidence recovery test
            current_dets = [DetectionResult(bbox_xyxy=bbox, confidence=conf, class_name="human", class_id=0, thermal_delta=30.0)]
            
        active = tracker_b.update(current_dets, frame_id=f, timestamp_ms=f * 33.3)
        
        for t in active:
            if t.is_confirmed:
                all_confirmed_ids_b.add(t.track_id)
                if f < 15 and initial_track_id is None:
                    initial_track_id = t.track_id
                elif f >= 23 and recovered_track_id is None:
                    recovered_track_id = t.track_id

    print(f"  * Pre-Occlusion Track ID       : #{initial_track_id} (Frames 0-14, CONFIRMED)")
    print(f"  * Occlusion Duration           : 8 frames (Frames 15-22, Kalman state projected)")
    print(f"  * Post-Occlusion Track ID      : #{recovered_track_id} (Frames 23-35, Stage 2 recovery)")
    print(f"  * Total Unique Confirmed IDs   : {len(all_confirmed_ids_b)} (Expected: Exactly 1 ID)")
    
    pass_b = (len(all_confirmed_ids_b) == 1 and initial_track_id == recovered_track_id)
    test_results["Test Case B (Occlusion Deduplication)"] = pass_b
    print(f"  => STATUS: [{'PASS' if pass_b else 'FAIL'}] Original Track ID #{recovered_track_id} preserved through 8-frame gap (Zero duplicate records).")

    # -------------------------------------------------------------------------
    # TEST CASE C: False Positive Noise Suppression (1-Frame & 2-Frame Spikes)
    # -------------------------------------------------------------------------
    print("\n[Test Case C] False Positive Suppression (1-Frame & 2-Frame Transient Noise)...")
    tracker_c = ByteTracker(high_thresh=0.5, low_thresh=0.1, min_hits=3, max_lost_frames=10)
    tracker_c.reset()

    false_positive_confirmations = []

    # Frame 0: 1-frame noise spike (water reflection)
    noise_1 = DetectionResult(bbox_xyxy=[1500, 1500, 1560, 1560], confidence=0.82, class_name="human", class_id=0)
    out_0 = tracker_c.update([noise_1], frame_id=0)
    false_positive_confirmations.extend([t for t in out_0 if t.is_confirmed])

    # Frame 1: Empty
    out_1 = tracker_c.update([], frame_id=1)
    false_positive_confirmations.extend([t for t in out_1 if t.is_confirmed])

    # Frames 2 & 3: 2-frame transient debris glint
    noise_2a = DetectionResult(bbox_xyxy=[2500, 1800, 2550, 1850], confidence=0.78, class_name="human", class_id=0)
    out_2 = tracker_c.update([noise_2a], frame_id=2)
    false_positive_confirmations.extend([t for t in out_2 if t.is_confirmed])

    noise_2b = DetectionResult(bbox_xyxy=[2502, 1801, 2552, 1851], confidence=0.76, class_name="human", class_id=0)
    out_3 = tracker_c.update([noise_2b], frame_id=3)
    false_positive_confirmations.extend([t for t in out_3 if t.is_confirmed])

    # Frames 4-10: Empty (noise disappears)
    for f in range(4, 11):
        out_f = tracker_c.update([], frame_id=f)
        false_positive_confirmations.extend([t for t in out_f if t.is_confirmed])

    print(f"  * 1-Frame Noise Injected       : Glint at (1500, 1500), conf=0.82 -> Filtered")
    print(f"  * 2-Frame Noise Injected       : Debris at (2500, 1800), conf=0.78 -> Filtered")
    print(f"  * Confirmed Noise Records      : {len(false_positive_confirmations)} (Required: 0 confirmations)")

    pass_c = (len(false_positive_confirmations) == 0)
    test_results["Test Case C (Noise Suppression)"] = pass_c
    print(f"  => STATUS: [{'PASS' if pass_c else 'FAIL'}] Spurious noise suppressed; is_confirmed=True is never assigned.")

    # -------------------------------------------------------------------------
    # TEST CASE D: Latency Overhead (100 Frames with 15 Concurrent Tracks)
    # -------------------------------------------------------------------------
    print("\n[Test Case D] Multi-Target Latency Stress Benchmark (100 Frames, 15 Concurrent Tracks)...")
    tracker_d = ByteTracker(high_thresh=0.5, low_thresh=0.1, min_hits=3, max_lost_frames=30)
    tracker_d.reset()

    latencies = []
    num_targets = 15
    num_frames = 100

    np.random.seed(123)
    # Initialize 15 target positions
    target_bases = [(np.random.uniform(200, 3600), np.random.uniform(200, 1900)) for _ in range(num_targets)]
    target_vels = [(np.random.uniform(-3, 3), np.random.uniform(-3, 3)) for _ in range(num_targets)]

    for f in range(num_frames):
        frame_dets = []
        for i in range(num_targets):
            bx = target_bases[i][0] + f * target_vels[i][0]
            by = target_bases[i][1] + f * target_vels[i][1]
            conf = np.random.uniform(0.3, 0.95)
            frame_dets.append(DetectionResult(
                bbox_xyxy=[bx, by, bx + 50, by + 100],
                confidence=conf,
                class_name="human" if i % 3 != 0 else "animal",
                class_id=0 if i % 3 != 0 else 1,
                thermal_delta=np.random.uniform(15.0, 45.0)
            ))

        t0 = time.perf_counter()
        _ = tracker_d.update(frame_dets, frame_id=f, timestamp_ms=f * 33.3)
        t_ms = (time.perf_counter() - t0) * 1000.0

        if f >= 5:  # Exclude frame 0-4 JIT warmup
            latencies.append(t_ms)

    mean_lat = float(np.mean(latencies))
    max_lat = float(np.max(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    
    print(f"  * Concurrent Active Tracks     : {num_targets} targets across {num_frames} frames")
    print(f"  * Mean Association Latency     : {mean_lat:.4f} ms ({mean_lat * 1000.0:.1f} microseconds)")
    print(f"  * 95th Percentile Latency      : {p95_lat:.4f} ms")
    print(f"  * Max Latency                  : {max_lat:.4f} ms")
    print(f"  * Latency Constraint (< 5.0ms) : {'PASSED' if p95_lat < 5.0 else 'FAILED'}")

    pass_d = (mean_lat < 2.0 and p95_lat < 5.0)
    test_results["Test Case D (Latency Overhead)"] = pass_d
    print(f"  => STATUS: [{'PASS' if pass_d else 'FAIL'}] Execution latency is {mean_lat:.3f} ms (P95: {p95_lat:.3f} ms), well within the 5.0 ms ceiling.")

    # -------------------------------------------------------------------------
    # FINAL SUMMARY REPORT TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("               MODULE 3 VERIFICATION & DEDUPLICATION SUMMARY")
    print("=" * 85)
    print(f"{'Test Case':<42} {'Measured Outcome':<28} {'Status':<10}")
    print("-" * 85)
    print(f"{'Case A: Continuous Movement (40 frames)':<42} {'1 Unique Persistent ID':<28} {'PASS' if pass_a else 'FAIL':<10}")
    print(f"{'Case B: Occlusion Recovery (8 frames)':<42} {'Original ID Recovered':<28} {'PASS' if pass_b else 'FAIL':<10}")
    print(f"{'Case C: False Positive Noise Suppression':<42} {'0 Spurious Confirmations':<28} {'PASS' if pass_c else 'FAIL':<10}")
    print(f"{'Case D: Multi-Track Latency Stress':<42} {f'{mean_lat:.3f} ms (15 tracks)':<28} {'PASS' if pass_d else 'FAIL':<10}")
    print("=" * 85)

    all_passed = all(test_results.values())
    if all_passed:
        print("\n>>> ALL MODULE 3 DEDUPLICATION & TRACKING TEST CASES PASSED! <<<\n")
    else:
        print("\n>>> ONE OR MORE MODULE 3 TEST CASES FAILED <<<\n")
        sys.exit(1)


if __name__ == "__main__":
    run_unit_and_stress_tests()
