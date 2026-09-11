"""
test_module5.py
===============
Comprehensive Verification Suite for Module 5 (Triage Priority Ranking, Dual Evidence Crops, GeoJSON/KML, & Interactive Map Viewer).

Verifies:
1. Multi-Criteria Triage Scoring Algorithm (Stationary vs Mobile priority)
2. Dual Evidence Thumbnail Extraction (Pristine Raw vs Annotated Bounding Box)
3. Standard RFC 7946 GeoJSON FeatureCollection Export & Schema Validation
4. ATAK/Google Earth Compatible KML Export
5. Standalone Interactive Leaflet HTML Map with UI Toggle Switch
"""

import sys
import os
import json
import numpy as np
import cv2

from src.geoloc.raycaster import GeolocatedSurvivor
from src.triage.triage_engine import TriageEngine, TriageRecord


def run_tests():
    print("=" * 80)
    print(" " * 18 + "MODULE 5 TRIAGE & RESCUE DISPATCH VERIFICATION")
    print("=" * 80)

    engine = TriageEngine(
        output_dir="output",
        evidence_dir="evidence"
    )

    results = []

    # -------------------------------------------------------------
    # TEST CASE 1: Multi-Criteria Triage Scoring
    # -------------------------------------------------------------
    # Case 1A: Critical Stationary Victim with Body Heat
    score_crit, urg_crit, col_crit = engine.compute_triage_score(
        confidence=0.90,
        is_stationary=True,
        thermal_delta=6.5,
        hits=10
    )

    # Case 1B: Mobile Animal / Non-Urgent
    score_low, urg_low, col_low = engine.compute_triage_score(
        confidence=0.60,
        is_stationary=False,
        thermal_delta=0.0,
        hits=2
    )

    # Stationary victim score must be significantly higher
    passed_1 = (
        score_crit >= 0.75 and
        urg_crit == "CRITICAL_RESCUE" and
        score_low < 0.50 and
        urg_low == "LOW_PRIORITY" and
        score_crit > score_low
    )

    results.append({
        "test": "1. Multi-Criteria Urgency Scoring",
        "condition": "Stationary+Thermal >= 0.75 (CRITICAL) vs Mobile < 0.50",
        "measured": f"Crit Score: {score_crit:.3f} [{urg_crit}] | Low Score: {score_low:.3f} [{urg_low}]",
        "passed": passed_1
    })

    # -------------------------------------------------------------
    # TEST CASE 2: Dual Evidence Thumbnail Generation
    # -------------------------------------------------------------
    # Create mock 4K synthetic frame with a survivor object
    mock_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    mock_frame[:] = (60, 80, 70)  # Terrain background
    # Draw mock survivor at (1900, 1050) -> (1950, 1120)
    cv2.rectangle(mock_frame, (1900, 1050), (1950, 1120), (180, 190, 200), -1)

    mock_survivor_1 = GeolocatedSurvivor(
        track_id=101,
        class_name="human",
        confidence=0.92,
        latitude=27.717245,
        longitude=85.324012,
        error_radius_m=2.1,
        is_stationary=True,
        thermal_delta=7.2,
        crop_bbox=[1900.0, 1050.0, 1950.0, 1120.0],
        timestamp_ms=3000.0
    )

    mock_survivor_2 = GeolocatedSurvivor(
        track_id=102,
        class_name="animal",
        confidence=0.65,
        latitude=27.717800,
        longitude=85.324500,
        error_radius_m=2.3,
        is_stationary=False,
        thermal_delta=1.0,
        crop_bbox=[2200.0, 1400.0, 2260.0, 1480.0],
        timestamp_ms=3000.0
    )

    rec_1 = engine.process_survivor(mock_survivor_1, frame_rgb=mock_frame, hits=12)
    rec_2 = engine.process_survivor(mock_survivor_2, frame_rgb=mock_frame, hits=4)

    raw_exists = os.path.exists(rec_1.raw_crop_path) and os.path.getsize(rec_1.raw_crop_path) > 500
    ann_exists = os.path.exists(rec_1.annotated_crop_path) and os.path.getsize(rec_1.annotated_crop_path) > 500

    raw_img = cv2.imread(rec_1.raw_crop_path)
    ann_img = cv2.imread(rec_1.annotated_crop_path)
    dims_valid = (raw_img.shape == (256, 256, 3)) and (ann_img.shape == (256, 256, 3))

    passed_2 = raw_exists and ann_exists and dims_valid
    results.append({
        "test": "2. Dual Evidence Thumbnails",
        "condition": "Generate 256x256 pristine raw & annotated crops",
        "measured": f"Raw: {os.path.basename(rec_1.raw_crop_path)} ({raw_img.shape}) | Annotated: {os.path.basename(rec_1.annotated_crop_path)} ({ann_img.shape})",
        "passed": passed_2
    })

    # -------------------------------------------------------------
    # TEST CASE 3: Standard RFC 7946 GeoJSON Export & Validation
    # -------------------------------------------------------------
    records = [rec_1, rec_2]
    geojson_path = engine.export_geojson(records)

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    is_feature_collection = geojson_data.get("type") == "FeatureCollection"
    num_features = len(geojson_data.get("features", []))
    first_coords = geojson_data["features"][0]["geometry"]["coordinates"]
    # Coordinates in GeoJSON RFC 7946 must be [longitude, latitude]
    coords_valid = (first_coords[0] == mock_survivor_1.longitude and first_coords[1] == mock_survivor_1.latitude)
    props_valid = "triage_score" in geojson_data["features"][0]["properties"]

    passed_3 = is_feature_collection and num_features == 2 and coords_valid and props_valid
    results.append({
        "test": "3. RFC 7946 GeoJSON Export",
        "condition": "Valid FeatureCollection with [lon, lat] & metadata",
        "measured": f"{num_features} Features exported to {os.path.basename(geojson_path)} (Point: {first_coords})",
        "passed": passed_3
    })

    # -------------------------------------------------------------
    # TEST CASE 4: Standard ATAK / KML Export
    # -------------------------------------------------------------
    kml_path = engine.export_kml(records)
    with open(kml_path, "r", encoding="utf-8") as f:
        kml_content = f.read()

    kml_valid = (
        "<kml" in kml_content and
        "<Placemark>" in kml_content and
        f"{mock_survivor_1.longitude},{mock_survivor_1.latitude},0" in kml_content and
        "CRITICAL_RESCUE" in kml_content
    )

    passed_4 = kml_valid and os.path.exists(kml_path)
    results.append({
        "test": "4. ATAK/Google Earth KML Export",
        "condition": "Valid KML placemarks with styled urgency icons",
        "measured": f"Exported {os.path.basename(kml_path)} ({os.path.getsize(kml_path)} bytes)",
        "passed": passed_4
    })

    # -------------------------------------------------------------
    # TEST CASE 5: Interactive Leaflet Mission Map Generation
    # -------------------------------------------------------------
    html_path = engine.generate_interactive_map(records)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    html_valid = (
        "leaflet.js" in html_content and
        "toggleCropView" in html_content and
        "CRITICAL_RESCUE" in html_content and
        "btn-raw" in html_content and
        "btn-ann" in html_content
    )

    passed_5 = html_valid and os.path.exists(html_path)
    results.append({
        "test": "5. Interactive Leaflet Map Viewer",
        "condition": "Self-contained HTML map with UI toggle switch",
        "measured": f"Generated {os.path.basename(html_path)} ({os.path.getsize(html_path)} bytes, toggle switch enabled)",
        "passed": passed_5
    })

    # -------------------------------------------------------------
    # Print Test Results Table
    # -------------------------------------------------------------
    print()
    print(f"{'Test Case':<32} | {'Requirement / Condition':<42} | {'Status':<8}")
    print("-" * 88)
    all_passed = True
    for r in results:
        status_str = "[PASS]" if r["passed"] else "[FAIL]"
        if not r["passed"]:
            all_passed = False
        print(f"{r['test']:<32} | {r['condition']:<42} | {status_str:<8}")
        print(f"   -> Result: {r['measured']}")
    print("-" * 88)

    if all_passed:
        print("\nALL MODULE 5 TRIAGE & RESCUE DISPATCH TESTS PASSED PERFECTLY!\n")
    else:
        print("\nSOME MODULE 5 TESTS FAILED. PLEASE REVIEW LOGS.\n")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
