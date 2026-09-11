"""
test_module4.py
===============
Comprehensive Verification Suite for Module 4 (Geolocation & 3D Raycasting Engine).

Verifies:
1. Nadir Projection Accuracy (< 0.05m tolerance at image center)
2. Oblique Angle Ray-Ground Intersection (45° forward pitch)
3. Sensor Error Budget Scaling with Altitude
4. Georeferencing Latency Benchmark (< 1.0ms for 100 projections)
5. End-to-End Integration with Module 3 TrackedSurvivor Dataclass
"""

import sys
import time
import math
import numpy as np

from src.ingest.telemetry_parser import TelemetrySample
from src.tracking.tracker import TrackedSurvivor
from src.geoloc.raycaster import (
    CameraIntrinsics,
    GeoRaycaster,
    GeolocatedSurvivor,
    wgs84_distance_meters,
    ned_to_wgs84
)


def run_tests():
    print("=" * 80)
    print(" " * 20 + "MODULE 4 GEOLOCATION & RAYCASTING VERIFICATION")
    print("=" * 80)

    raycaster = GeoRaycaster()
    results = []

    # -------------------------------------------------------------
    # TEST CASE 1: Nadir Check (Drone directly overhead)
    # -------------------------------------------------------------
    drone_lat = 27.717245
    drone_lon = 85.324012
    altitude = 50.0  # 50m AGL
    pitch = -90.0   # Straight down
    roll = 0.0
    yaw = 0.0
    
    # Image center pixel for 4K
    center_u = 1920.0
    center_v = 1080.0

    target_lat, target_lon, err_r, g_dist = raycaster.raycast_pixel(
        u=center_u,
        v=center_v,
        drone_lat=drone_lat,
        drone_lon=drone_lon,
        altitude_agl=altitude,
        pitch=pitch,
        roll=roll,
        yaw=yaw
    )

    offset_meters = wgs84_distance_meters(drone_lat, drone_lon, target_lat, target_lon)
    passed_1 = offset_meters < 0.05
    results.append({
        "test": "1. Nadir Accuracy (Image Center)",
        "condition": f"Offset < 0.05m (Alt={altitude}m, Pitch={pitch} deg)",
        "measured": f"Offset: {offset_meters:.4f}m (Lat:{target_lat:.6f}, Lon:{target_lon:.6f})",
        "passed": passed_1
    })

    # -------------------------------------------------------------
    # TEST CASE 2: Oblique Projection (45 deg Forward Pitch)
    # -------------------------------------------------------------
    pitch_oblique = -45.0
    yaw_north = 0.0
    
    target_lat_2, target_lon_2, err_r_2, g_dist_2 = raycaster.raycast_pixel(
        u=center_u,
        v=center_v,
        drone_lat=drone_lat,
        drone_lon=drone_lon,
        altitude_agl=altitude,
        pitch=pitch_oblique,
        roll=roll,
        yaw=yaw_north
    )

    # At 45 deg pitch and 50m alt, forward ground displacement = 50m / tan(45 deg) = 50.0m North
    dist_measured = wgs84_distance_meters(drone_lat, drone_lon, target_lat_2, target_lon_2)
    expected_dist = 50.0
    dist_error = abs(dist_measured - expected_dist)
    passed_2 = dist_error < 0.1 and target_lat_2 > drone_lat  # Target is north of drone
    results.append({
        "test": "2. Oblique Projection (45 deg Forward)",
        "condition": "Displacement ~50.0m True North (+/-0.1m)",
        "measured": f"Dist: {dist_measured:.3f}m (dLat: +{target_lat_2 - drone_lat:.6f} deg)",
        "passed": passed_2
    })

    # -------------------------------------------------------------
    # TEST CASE 3: Error Budget Scaling
    # -------------------------------------------------------------
    err_30m = raycaster.compute_error_budget(altitude_agl=30.0, look_angle_deg=0.0)
    err_100m = raycaster.compute_error_budget(altitude_agl=100.0, look_angle_deg=0.0)
    
    # Expected: err(30m) ~ 2.02m, err(100m) ~ 3.25m
    passed_3 = (1.9 < err_30m < 2.2) and (3.1 < err_100m < 3.4) and (err_100m > err_30m)
    results.append({
        "test": "3. Error Budget Scaling",
        "condition": "Error radius scales linearly with altitude (AGL)",
        "measured": f"30m Alt: +/-{err_30m}m | 100m Alt: +/-{err_100m}m",
        "passed": passed_3
    })

    # -------------------------------------------------------------
    # TEST CASE 4: Latency & Throughput Benchmark
    # -------------------------------------------------------------
    num_queries = 100
    t0 = time.perf_counter()
    for i in range(num_queries):
        raycaster.raycast_pixel(
            u=1920.0 + (i % 200 - 100),
            v=1080.0 + (i % 200 - 100),
            drone_lat=drone_lat,
            drone_lon=drone_lon,
            altitude_agl=altitude,
            pitch=-60.0,
            roll=0.0,
            yaw=45.0
        )
    t1 = time.perf_counter()
    total_time_ms = (t1 - t0) * 1000.0
    avg_us_per_ray = (total_time_ms / num_queries) * 1000.0
    passed_4 = total_time_ms < 1.0  # 100 projections in < 1.0ms

    results.append({
        "test": f"4. Latency Benchmark ({num_queries} rays)",
        "condition": "Total time < 1.0ms (< 10us/ray)",
        "measured": f"Total: {total_time_ms:.3f}ms ({avg_us_per_ray:.2f} us/ray)",
        "passed": passed_4
    })

    # -------------------------------------------------------------
    # TEST CASE 5: Module 3 Integration with TrackedSurvivor
    # -------------------------------------------------------------
    mock_track = TrackedSurvivor(
        track_id=42,
        class_name="human",
        current_bbox=[1900.0, 1060.0, 1940.0, 1100.0],  # Center is (1920, 1080)
        confidence=0.88,
        history_centers=[(1920.0, 1080.0)],
        is_confirmed=True,
        is_stationary=True,
        thermal_delta=4.8,
        status="CONFIRMED"
    )

    mock_telemetry = TelemetrySample(
        timestamp_ms=10500.0,
        lat=drone_lat,
        lon=drone_lon,
        altitude_agl=50.0,
        pitch=-90.0,
        roll=0.0,
        yaw=0.0
    )

    geoloc_result = raycaster.geolocate_track(
        track=mock_track,
        telemetry=mock_telemetry,
        timestamp_ms=10500.0
    )

    passed_5 = (
        isinstance(geoloc_result, GeolocatedSurvivor) and
        geoloc_result.track_id == 42 and
        geoloc_result.class_name == "human" and
        geoloc_result.is_stationary is True and
        geoloc_result.thermal_delta == 4.8 and
        abs(geoloc_result.latitude - drone_lat) < 1e-5 and
        abs(geoloc_result.longitude - drone_lon) < 1e-5
    )

    results.append({
        "test": "5. Module 3 End-to-End Ingestion",
        "condition": "Valid GeolocatedSurvivor output dataclass",
        "measured": f"ID:{geoloc_result.track_id} ({geoloc_result.class_name}), Stationary:{geoloc_result.is_stationary}, GPS:({geoloc_result.latitude:.6f}, {geoloc_result.longitude:.6f})",
        "passed": passed_5
    })

    # -------------------------------------------------------------
    # Print Test Results Table
    # -------------------------------------------------------------
    print()
    print(f"{'Test Case':<32} | {'Requirement / Condition':<40} | {'Status':<8}")
    print("-" * 86)
    all_passed = True
    for r in results:
        status_str = "[PASS]" if r["passed"] else "[FAIL]"
        if not r["passed"]:
            all_passed = False
        print(f"{r['test']:<32} | {r['condition']:<40} | {status_str:<8}")
        print(f"   -> Result: {r['measured']}")
    print("-" * 86)
    
    if all_passed:
        print("\nALL MODULE 4 GEOLOCATION & RAYCASTING TESTS PASSED PERFECTLY!\n")
    else:
        print("\nSOME MODULE 4 TESTS FAILED. PLEASE REVIEW LOGS.\n")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
