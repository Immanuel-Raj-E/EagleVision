"""
triage_engine.py
================
Triage & Priority Ranking Engine for Aerial SAR Operations.
Ingests geolocated survivor tracks, calculates multi-criteria medical/tactical urgency scores,
extracts dual evidence thumbnails (pristine raw vs. annotated bounding box), and exports
standardized GeoJSON, KML, and interactive Leaflet mission maps.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import os
import json
import base64
import cv2
import numpy as np

from src.geoloc.raycaster import GeolocatedSurvivor


@dataclass
class TriageRecord:
    """Standardized record for an actioned disaster survivor sighting."""
    track_id: int
    class_name: str
    latitude: float
    longitude: float
    triage_score: float
    urgency_level: str              # "CRITICAL_RESCUE", "MODERATE_SEARCH", "LOW_PRIORITY"
    urgency_color: str              # "#e63946" (Red), "#f4a261" (Orange), "#e9c46a" (Yellow)
    confidence: float
    is_stationary: bool
    status: str                     # "STATIONARY" vs "MOBILE"
    thermal_delta: Optional[float]
    error_radius_m: float
    raw_crop_path: str
    annotated_crop_path: str
    crop_bbox: list[float] = field(default_factory=list)
    full_frame_path: str = ""
    crop_base64_raw: Optional[str] = None
    crop_base64_annotated: Optional[str] = None
    first_detected_ts: float = 0.0
    last_detected_ts: float = 0.0
    hits: int = 1


class TriageEngine:
    """
    Multi-criteria triage ranking and geospatial report generation engine.
    """

    def __init__(
        self,
        output_dir: str = "output",
        evidence_dir: str = "evidence",
        w_conf: float = 0.30,
        w_stat: float = 0.35,
        w_therm: float = 0.20,
        w_pers: float = 0.15
    ):
        self.output_dir = output_dir
        self.evidence_dir = evidence_dir
        self.raw_evidence_dir = os.path.join(evidence_dir, "raw")
        self.annotated_evidence_dir = os.path.join(evidence_dir, "annotated")
        self.full_frames_dir = os.path.join(evidence_dir, "full_frames")

        self.w_conf = w_conf
        self.w_stat = w_stat
        self.w_therm = w_therm
        self.w_pers = w_pers

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.raw_evidence_dir, exist_ok=True)
        os.makedirs(self.annotated_evidence_dir, exist_ok=True)
        os.makedirs(self.full_frames_dir, exist_ok=True)

    def clear_evidence(self):
        """Purges old data frames and evidence crops from the survivor priority queue / evidence store."""
        for d in [self.raw_evidence_dir, self.annotated_evidence_dir, self.full_frames_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    fp = os.path.join(d, f)
                    try:
                        if os.path.isfile(fp):
                            os.remove(fp)
                    except Exception:
                        pass

    def compute_triage_score(
        self,
        confidence: float,
        is_stationary: bool,
        thermal_delta: Optional[float] = None,
        hits: int = 3
    ) -> Tuple[float, str, str]:
        """
        Calculates normalized urgency score S in [0.0, 1.0].
        
        Formula:
            S = w1 * conf + w2 * I_stationary + w3 * min(1.0, thermal_delta / 10.0) + w4 * persistence
        """
        stat_factor = 1.0 if is_stationary else 0.0
        
        thermal_factor = 0.0
        if thermal_delta is not None and thermal_delta > 0.0:
            thermal_factor = min(1.0, thermal_delta / 10.0)
            
        persistence_factor = min(1.0, max(0.1, hits / 10.0))

        score = (
            self.w_conf * confidence +
            self.w_stat * stat_factor +
            self.w_therm * thermal_factor +
            self.w_pers * persistence_factor
        )
        score = max(0.0, min(1.0, score))
        score = round(score, 3)

        if score >= 0.75:
            urgency = "CRITICAL_RESCUE"
            color = "#e63946"  # Red
        elif score >= 0.50:
            urgency = "MODERATE_SEARCH"
            color = "#f4a261"  # Orange
        else:
            urgency = "LOW_PRIORITY"
            color = "#e9c46a"  # Yellow

        return score, urgency, color

    def extract_dual_evidence_crops(
        self,
        frame_rgb: Optional[np.ndarray],
        crop_bbox: list[float],
        track_id: int,
        urgency_level: str,
        confidence: float,
        class_name: str,
        padding_pct: float = 0.25
    ) -> Tuple[str, str, str, str]:
        """
        Extracts pristine raw crop and annotated crop with bounding box & triage header.
        Returns: (raw_path, annotated_path, raw_b64, annotated_b64)
        """
        raw_filename = f"track_{track_id}_raw.jpg"
        annotated_filename = f"track_{track_id}_annotated.jpg"

        raw_path = os.path.join(self.raw_evidence_dir, raw_filename)
        annotated_path = os.path.join(self.annotated_evidence_dir, annotated_filename)

        if frame_rgb is None:
            # Generate placeholder 256x256 image if no frame passed
            placeholder = np.zeros((256, 256, 3), dtype=np.uint8)
            placeholder[:] = (40, 44, 52)
            cv2.putText(placeholder, f"ID #{track_id} ({class_name})", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(placeholder, urgency_level, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(raw_path, placeholder)
            cv2.imwrite(annotated_path, placeholder)
            _, buf = cv2.imencode(".jpg", placeholder)
            b64_str = base64.b64encode(buf).decode("utf-8")
            return raw_path, annotated_path, b64_str, b64_str

        h_img, w_img = frame_rgb.shape[:2]
        x1, y1, x2, y2 = crop_bbox
        w_box = max(10, x2 - x1)
        h_box = max(10, y2 - y1)

        # Add context padding
        pad_x = w_box * padding_pct
        pad_y = h_box * padding_pct

        crop_x1 = int(max(0, x1 - pad_x))
        crop_y1 = int(max(0, y1 - pad_y))
        crop_x2 = int(min(w_img, x2 + pad_x))
        crop_y2 = int(min(h_img, y2 + pad_y))

        raw_crop = frame_rgb[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        if raw_crop.size == 0:
            raw_crop = np.zeros((256, 256, 3), dtype=np.uint8)

        # Standardize thumbnail dimensions
        raw_crop_resized = cv2.resize(raw_crop, (256, 256), interpolation=cv2.INTER_AREA)

        # Build Annotated Crop
        annotated_crop = raw_crop.copy()
        rel_x1 = int(x1 - crop_x1)
        rel_y1 = int(y1 - crop_y1)
        rel_x2 = int(x2 - crop_x1)
        rel_y2 = int(y2 - crop_y1)

        # Color mapping for box
        if urgency_level == "CRITICAL_RESCUE":
            box_color = (0, 0, 230)      # BGR Red
        elif urgency_level == "MODERATE_SEARCH":
            box_color = (0, 140, 255)    # BGR Orange
        else:
            box_color = (0, 220, 220)    # BGR Yellow

        # Draw bounding box
        cv2.rectangle(annotated_crop, (rel_x1, rel_y1), (rel_x2, rel_y2), box_color, 2)
        
        # Header banner
        label = f"#{track_id} {class_name} {int(confidence * 100)}%"
        cv2.rectangle(annotated_crop, (rel_x1, max(0, rel_y1 - 20)), (rel_x1 + 140, max(20, rel_y1)), box_color, -1)
        cv2.putText(annotated_crop, label, (rel_x1 + 2, max(14, rel_y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        annotated_crop_resized = cv2.resize(annotated_crop, (256, 256), interpolation=cv2.INTER_AREA)

        # Save to disk
        cv2.imwrite(raw_path, raw_crop_resized)
        cv2.imwrite(annotated_path, annotated_crop_resized)

        # Encode base64 data strings for standalone web maps
        _, raw_buf = cv2.imencode(".jpg", raw_crop_resized)
        _, ann_buf = cv2.imencode(".jpg", annotated_crop_resized)
        raw_b64 = base64.b64encode(raw_buf).decode("utf-8")
        ann_b64 = base64.b64encode(ann_buf).decode("utf-8")

        return raw_path, annotated_path, raw_b64, ann_b64

    def process_survivor(
        self,
        survivor: GeolocatedSurvivor,
        frame_rgb: Optional[np.ndarray] = None,
        hits: int = 3,
        first_detected_ts: Optional[float] = None
    ) -> TriageRecord:
        """Processes a single geolocated survivor into a triage record."""
        score, urgency, color = self.compute_triage_score(
            confidence=survivor.confidence,
            is_stationary=survivor.is_stationary,
            thermal_delta=survivor.thermal_delta,
            hits=hits
        )

        raw_path, ann_path, raw_b64, ann_b64 = self.extract_dual_evidence_crops(
            frame_rgb=frame_rgb,
            crop_bbox=survivor.crop_bbox,
            track_id=survivor.track_id,
            urgency_level=urgency,
            confidence=survivor.confidence,
            class_name=survivor.class_name
        )

        full_frame_path = ""
        if frame_rgb is not None:
            full_frame_path = os.path.join(self.full_frames_dir, f"track_{survivor.track_id}_full.jpg")
            cv2.imwrite(full_frame_path, frame_rgb)

        return TriageRecord(
            track_id=survivor.track_id,
            class_name=survivor.class_name,
            latitude=survivor.latitude,
            longitude=survivor.longitude,
            triage_score=score,
            urgency_level=urgency,
            urgency_color=color,
            confidence=survivor.confidence,
            is_stationary=survivor.is_stationary,
            status="STATIONARY" if survivor.is_stationary else "MOBILE",
            thermal_delta=survivor.thermal_delta,
            error_radius_m=survivor.error_radius_m,
            raw_crop_path=raw_path,
            annotated_crop_path=ann_path,
            crop_bbox=list(survivor.crop_bbox),
            full_frame_path=full_frame_path,
            crop_base64_raw=raw_b64,
            crop_base64_annotated=ann_b64,
            first_detected_ts=first_detected_ts if first_detected_ts is not None else survivor.timestamp_ms,
            last_detected_ts=survivor.timestamp_ms,
            hits=hits
        )

    def process_ground_entity(
        self,
        entity: Any,
        first_detected_ts: Optional[float] = None
    ) -> TriageRecord:
        """Processes a deduplicated GroundEntity into a final TriageRecord."""
        score, urgency, color = self.compute_triage_score(
            confidence=entity.best_confidence,
            is_stationary=entity.is_stationary,
            thermal_delta=entity.thermal_delta,
            hits=entity.total_hits
        )

        raw_path, ann_path, raw_b64, ann_b64 = self.extract_dual_evidence_crops(
            frame_rgb=entity.best_frame,
            crop_bbox=entity.best_crop_bbox,
            track_id=entity.entity_id,
            urgency_level=urgency,
            confidence=entity.best_confidence,
            class_name=entity.class_name
        )

        full_frame_path = ""
        if entity.best_frame is not None:
            full_frame_path = os.path.join(self.full_frames_dir, f"track_{entity.entity_id}_full.jpg")
            cv2.imwrite(full_frame_path, entity.best_frame)

        return TriageRecord(
            track_id=entity.entity_id,
            class_name=entity.class_name,
            latitude=entity.latitude,
            longitude=entity.longitude,
            triage_score=score,
            urgency_level=urgency,
            urgency_color=color,
            confidence=entity.best_confidence,
            is_stationary=entity.is_stationary,
            status=entity.status,
            thermal_delta=entity.thermal_delta,
            error_radius_m=entity.error_radius_m,
            raw_crop_path=raw_path,
            annotated_crop_path=ann_path,
            crop_bbox=list(entity.best_crop_bbox),
            full_frame_path=full_frame_path,
            crop_base64_raw=raw_b64,
            crop_base64_annotated=ann_b64,
            first_detected_ts=entity.first_detected_ms,
            last_detected_ts=entity.last_detected_ms,
            hits=entity.total_hits
        )

    def deduplicate_records(
        self,
        records: List[TriageRecord],
        match_radius_m: float = 8.0
    ) -> List[TriageRecord]:
        """Performs post-hoc spatial-temporal deduplication on a list of TriageRecords."""
        from src.tracking.deduplicator import haversine_distance_m
        deduped: List[TriageRecord] = []
        for rec in records:
            matched = False
            for existing in deduped:
                if existing.class_name != rec.class_name:
                    continue
                dist = haversine_distance_m(rec.latitude, rec.longitude, existing.latitude, existing.longitude)
                if dist <= match_radius_m:
                    matched = True
                    # Merge into existing: update hit count, last detected, higher confidence
                    existing.hits += rec.hits
                    existing.last_detected_ts = max(existing.last_detected_ts, rec.last_detected_ts)
                    if rec.confidence > existing.confidence:
                        existing.confidence = rec.confidence
                        existing.triage_score = max(existing.triage_score, rec.triage_score)
                        existing.crop_bbox = rec.crop_bbox
                        existing.raw_crop_path = rec.raw_crop_path
                        existing.annotated_crop_path = rec.annotated_crop_path
                        existing.crop_base64_raw = rec.crop_base64_raw
                        existing.crop_base64_annotated = rec.crop_base64_annotated
                    break
            if not matched:
                deduped.append(rec)
        return deduped

    def export_geojson(
        self,
        records: List[TriageRecord],
        output_path: Optional[str] = None
    ) -> str:
        """
        Exports triage records to standard RFC 7946 GeoJSON FeatureCollection.
        """
        path = output_path or os.path.join(self.output_dir, "triage_survivors.geojson")
        features = []

        # Sort descending by priority score
        sorted_records = sorted(records, key=lambda x: x.triage_score, reverse=True)

        for rec in sorted_records:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [rec.longitude, rec.latitude]
                },
                "properties": {
                    "track_id": rec.track_id,
                    "class_name": rec.class_name,
                    "triage_score": rec.triage_score,
                    "urgency_level": rec.urgency_level,
                    "urgency_color": rec.urgency_color,
                    "confidence": rec.confidence,
                    "status": rec.status,
                    "is_stationary": rec.is_stationary,
                    "thermal_delta_c": rec.thermal_delta,
                    "error_radius_m": rec.error_radius_m,
                    "first_detected_ms": rec.first_detected_ts,
                    "last_detected_ms": rec.last_detected_ts,
                    "hits_count": rec.hits,
                    "crop_bbox": rec.crop_bbox,
                    "full_frame_path": rec.full_frame_path.replace("\\", "/"),
                    "raw_crop_path": rec.raw_crop_path.replace("\\", "/"),
                    "annotated_crop_path": rec.annotated_crop_path.replace("\\", "/")
                }
            }
            features.append(feature)

        geojson_doc = {
            "type": "FeatureCollection",
            "name": "SAR_Triage_Survivors",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(geojson_doc, f, indent=2)

        return path

    def export_kml(
        self,
        records: List[TriageRecord],
        output_path: Optional[str] = None
    ) -> str:
        """
        Exports triage records to standard Google Earth / ATAK compatible KML.
        """
        path = output_path or os.path.join(self.output_dir, "triage_survivors.kml")
        
        kml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '  <Document>',
            '    <name>SAR Mission Survivor Triage Map</name>',
            '    <description>Confidence-ranked aerial SAR survivor geolocations with dual evidence imagery</description>',
            '    <!-- Style Definitions -->',
            '    <Style id="criticalStyle">',
            '      <IconStyle>',
            '        <color>ff4639e6</color>',  # KML aabbggrr
            '        <scale>1.3</scale>',
            '        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/target.png</href></Icon>',
            '      </IconStyle>',
            '    </Style>',
            '    <Style id="moderateStyle">',
            '      <IconStyle>',
            '        <color>ff61a2f4</color>',
            '        <scale>1.1</scale>',
            '        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/caution.png</href></Icon>',
            '      </IconStyle>',
            '    </Style>',
            '    <Style id="lowStyle">',
            '      <IconStyle>',
            '        <color>ff6ac4e9</color>',
            '        <scale>0.9</scale>',
            '        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>',
            '      </IconStyle>',
            '    </Style>'
        ]

        sorted_records = sorted(records, key=lambda x: x.triage_score, reverse=True)

        for rec in sorted_records:
            style_id = (
                "#criticalStyle" if rec.urgency_level == "CRITICAL_RESCUE"
                else "#moderateStyle" if rec.urgency_level == "MODERATE_SEARCH"
                else "#lowStyle"
            )

            desc = (
                f"<![CDATA["
                f"<h3>Track #{rec.track_id} - {rec.urgency_level}</h3>"
                f"<p><b>Class:</b> {rec.class_name}<br/>"
                f"<b>Triage Score:</b> {rec.triage_score}<br/>"
                f"<b>Confidence:</b> {int(rec.confidence * 100)}%<br/>"
                f"<b>Movement Status:</b> {rec.status}<br/>"
                f"<b>Thermal Delta:</b> {f'+{rec.thermal_delta}°C' if rec.thermal_delta else 'N/A'}<br/>"
                f"<b>GPS Uncertainty:</b> ±{rec.error_radius_m}m<br/>"
                f"<b>Coordinates:</b> {rec.latitude:.6f}, {rec.longitude:.6f}</p>"
                f"]]>"
            )

            kml_lines.extend([
                '    <Placemark>',
                f'      <name>Survivor #{rec.track_id} [{rec.urgency_level}]</name>',
                f'      <description>{desc}</description>',
                f'      <styleUrl>{style_id}</styleUrl>',
                '      <Point>',
                f'        <coordinates>{rec.longitude},{rec.latitude},0</coordinates>',
                '      </Point>',
                '    </Placemark>'
            ])

        kml_lines.extend([
            '  </Document>',
            '</kml>'
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(kml_lines))

        return path

    def generate_interactive_map(
        self,
        records: List[TriageRecord],
        output_path: Optional[str] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None
    ) -> str:
        """
        Generates a self-contained, interactive Leaflet HTML dashboard with dual raw/annotated crop toggles.
        """
        path = output_path or os.path.join(self.output_dir, "mission_triage_map.html")

        if records:
            c_lat = center_lat if center_lat is not None else sum(r.latitude for r in records) / len(records)
            c_lon = center_lon if center_lon is not None else sum(r.longitude for r in records) / len(records)
        else:
            c_lat, c_lon = 27.717245, 85.324012

        # Prepare JSON payload for map pins
        markers_data = []
        for r in records:
            raw_src = f"data:image/jpeg;base64,{r.crop_base64_raw}" if r.crop_base64_raw else r.raw_crop_path
            ann_src = f"data:image/jpeg;base64,{r.crop_base64_annotated}" if r.crop_base64_annotated else r.annotated_crop_path

            markers_data.append({
                "track_id": r.track_id,
                "class_name": r.class_name,
                "lat": r.latitude,
                "lon": r.longitude,
                "score": r.triage_score,
                "urgency": r.urgency_level,
                "color": r.urgency_color,
                "confidence": int(r.confidence * 100),
                "status": r.status,
                "thermal": f"+{r.thermal_delta}°C" if r.thermal_delta else "N/A",
                "error_m": r.error_radius_m,
                "raw_img": raw_src,
                "ann_img": ann_src
            })

        markers_json = json.dumps(markers_data)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Drone SAR Survivor Mission Triage Dashboard</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{ background: #0f172a; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
    header {{ background: #1e293b; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; }}
    .title {{ font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 8px; color: #38bdf8; }}
    .stats-bar {{ display: flex; gap: 16px; font-size: 13px; }}
    .stat-pill {{ padding: 4px 12px; border-radius: 9999px; font-weight: 600; display: flex; align-items: center; gap: 6px; }}
    .critical {{ background: rgba(230, 57, 70, 0.2); color: #e63946; border: 1px solid #e63946; }}
    .moderate {{ background: rgba(244, 162, 97, 0.2); color: #f4a261; border: 1px solid #f4a261; }}
    .low {{ background: rgba(233, 196, 106, 0.2); color: #e9c46a; border: 1px solid #e9c46a; }}
    #map-container {{ flex: 1; position: relative; }}
    #map {{ width: 100%; height: 100%; background: #0b0f19; }}
    
    /* Popup Card Styling */
    .leaflet-popup-content-wrapper {{ background: #1e293b; color: #f8fafc; border-radius: 12px; border: 1px solid #475569; padding: 0; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }}
    .leaflet-popup-content {{ margin: 0; width: 280px !important; }}
    .popup-header {{ padding: 12px 16px; border-bottom: 1px solid #334155; display: flex; justify-content: space-between; align-items: center; }}
    .popup-title {{ font-size: 15px; font-weight: 700; }}
    .badge {{ font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; }}
    .popup-body {{ padding: 12px 16px; }}
    
    /* Dual Image Viewer with Toggle */
    .img-container {{ position: relative; width: 100%; height: 200px; border-radius: 8px; overflow: hidden; background: #000; border: 1px solid #334155; margin-bottom: 10px; }}
    .img-container img {{ width: 100%; height: 100%; object-fit: cover; }}
    .toggle-bar {{ display: flex; background: #0f172a; border-radius: 6px; padding: 2px; margin-bottom: 10px; border: 1px solid #334155; }}
    .toggle-btn {{ flex: 1; padding: 6px 0; font-size: 11px; font-weight: 600; text-align: center; border-radius: 4px; cursor: pointer; color: #94a3b8; transition: all 0.2s; }}
    .toggle-btn.active {{ background: #38bdf8; color: #0f172a; }}
    
    /* Metadata Grid */
    .meta-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 12px; margin-bottom: 12px; }}
    .meta-item {{ display: flex; flex-direction: column; background: #0f172a; padding: 6px 8px; border-radius: 6px; }}
    .meta-label {{ font-size: 10px; color: #94a3b8; text-transform: uppercase; }}
    .meta-value {{ font-weight: 600; color: #f1f5f9; }}
    .dispatch-btn {{ width: 100%; padding: 8px 0; background: #2563eb; color: #fff; font-weight: 700; font-size: 12px; border: none; border-radius: 6px; cursor: pointer; text-align: center; transition: background 0.2s; }}
    .dispatch-btn:hover {{ background: #1d4ed8; }}
  </style>
</head>
<body>
  <header>
    <div class="title">
      <span>🚁</span> Drone SAR Mission Triage Intelligence Map
    </div>
    <div class="stats-bar">
      <div class="stat-pill critical">Critical Rescue: <span id="crit-count">0</span></div>
      <div class="stat-pill moderate">Moderate Search: <span id="mod-count">0</span></div>
      <div class="stat-pill low">Low Priority: <span id="low-count">0</span></div>
    </div>
  </header>

  <div id="map-container">
    <div id="map"></div>
  </div>

  <script>
    const survivors = {markers_json};

    // Calculate Summary Counts
    let crit = 0, mod = 0, low = 0;
    survivors.forEach(s => {{
      if (s.urgency === 'CRITICAL_RESCUE') crit++;
      else if (s.urgency === 'MODERATE_SEARCH') mod++;
      else low++;
    }});
    document.getElementById('crit-count').innerText = crit;
    document.getElementById('mod-count').innerText = mod;
    document.getElementById('low-count').innerText = low;

    // Initialize Leaflet Map with CartoDB Dark Matter tiles
    const map = L.map('map').setView([{c_lat}, {c_lon}], 17);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 21
    }}).addTo(map);

    // Interactive Image Toggle Function
    window.toggleCropView = function(trackId, viewType) {{
      const imgElem = document.getElementById('crop-img-' + trackId);
      const rawBtn = document.getElementById('btn-raw-' + trackId);
      const annBtn = document.getElementById('btn-ann-' + trackId);
      const s = survivors.find(item => item.track_id === trackId);
      if (!imgElem || !s) return;

      if (viewType === 'raw') {{
        imgElem.src = s.raw_img;
        rawBtn.classList.add('active');
        annBtn.classList.remove('active');
      }} else {{
        imgElem.src = s.ann_img;
        annBtn.classList.add('active');
        rawBtn.classList.remove('active');
      }}
    }};

    // Plot Markers with pulsing pins
    survivors.forEach(s => {{
      const customIcon = L.divIcon({{
        className: 'custom-pin',
        html: `<div style="background:${{s.color}}; width: 22px; height: 22px; border-radius: 50%; border: 3px solid #ffffff; box-shadow: 0 0 10px ${{s.color}};"></div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11]
      }});

      const marker = L.marker([s.lat, s.lon], {{ icon: customIcon }}).addTo(map);

      // Uncertainty Circle
      L.circle([s.lat, s.lon], {{
        radius: s.error_m,
        color: s.color,
        fillColor: s.color,
        fillOpacity: 0.15,
        weight: 1
      }}).addTo(map);

      const popupHtml = `
        <div class="popup-header">
          <div class="popup-title">Track #${{s.track_id}} (${{s.class_name}})</div>
          <div class="badge" style="background:${{s.color}}; color:#fff;">${{s.urgency.replace('_', ' ')}}</div>
        </div>
        <div class="popup-body">
          <div class="toggle-bar">
            <div id="btn-raw-${{s.track_id}}" class="toggle-btn" onclick="toggleCropView(${{s.track_id}}, 'raw')">Raw Image</div>
            <div id="btn-ann-${{s.track_id}}" class="toggle-btn active" onclick="toggleCropView(${{s.track_id}}, 'annotated')">Show Box</div>
          </div>
          <div class="img-container">
            <img id="crop-img-${{s.track_id}}" src="${{s.ann_img}}" alt="Evidence Crop" />
          </div>
          <div class="meta-grid">
            <div class="meta-item"><span class="meta-label">Triage Score</span><span class="meta-value">${{s.score}}</span></div>
            <div class="meta-item"><span class="meta-label">Confidence</span><span class="meta-value">${{s.confidence}}%</span></div>
            <div class="meta-item"><span class="meta-label">Status</span><span class="meta-value">${{s.status}}</span></div>
            <div class="meta-item"><span class="meta-label">Thermal ΔT</span><span class="meta-value">${{s.thermal}}</span></div>
            <div class="meta-item"><span class="meta-label">GPS Lat/Lon</span><span class="meta-value">${{s.lat.toFixed(6)}}, ${{s.lon.toFixed(6)}}</span></div>
            <div class="meta-item"><span class="meta-label">Error Budget</span><span class="meta-value">±${{s.error_m}}m</span></div>
          </div>
          <button class="dispatch-btn" onclick="alert('Dispatching SAR Rescue Drone to Lat: ' + ${{s.lat}} + ', Lon: ' + ${{s.lon}})">🚀 DISPATCH RESCUE TEAM</button>
        </div>
      `;

      marker.bindPopup(popupHtml);
    }});
  </script>
</body>
</html>
"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return path
