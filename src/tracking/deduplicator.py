"""
deduplicator.py
===============
Geospatial and Temporal Deduplication Engine for Drone SAR Intelligence.
Merges fragmented tracklets and recurrent observations of the same ground survivor
using geodesic (Haversine) distance clustering, appearance confidence, and coordinate refinement.
Guarantees a 1-to-1 mapping between real-world ground survivors and mission triage records.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import math
import time
import numpy as np


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes geodesic distance in meters between two WGS84 coordinates.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c


@dataclass
class GroundEntity:
    """Persistent ground survivor / animal entity after spatial-temporal deduplication."""
    entity_id: int
    class_name: str
    latitude: float
    longitude: float
    best_confidence: float
    is_stationary: bool
    status: str
    thermal_delta: Optional[float]
    error_radius_m: float
    first_detected_ms: float
    last_detected_ms: float
    total_hits: int
    associated_track_ids: List[int] = field(default_factory=list)
    best_crop_bbox: List[float] = field(default_factory=list)
    best_frame: Optional[np.ndarray] = None
    observation_count: int = 1
    trajectory_coords: List[Tuple[float, float]] = field(default_factory=list)


class SpatialDeduplicator:
    """
    Multi-stage spatial-temporal deduplication engine.
    Clusters track detections on the ground reference plane using WGS84 distance metrics.
    """

    def __init__(
        self,
        match_radius_m: float = 8.0,
        min_hits_to_confirm: int = 3,
        stationary_disp_thresh_m: float = 3.0
    ):
        self.match_radius_m = match_radius_m
        self.min_hits_to_confirm = min_hits_to_confirm
        self.stationary_disp_thresh_m = stationary_disp_thresh_m

        self._next_entity_id = 1
        self.entities: Dict[int, GroundEntity] = {}
        self.total_raw_sightings: int = 0
        self.total_duplicate_suppressions: int = 0

    def reset(self):
        """Resets deduplicator state for a new mission."""
        self._next_entity_id = 1
        self.entities.clear()
        self.total_raw_sightings = 0
        self.total_duplicate_suppressions = 0

    def update_entity(
        self,
        track_id: int,
        class_name: str,
        lat: float,
        lon: float,
        confidence: float,
        is_stationary: bool,
        thermal_delta: Optional[float],
        error_radius_m: float,
        crop_bbox: List[float],
        frame_rgb: Optional[np.ndarray],
        timestamp_ms: float,
        hits: int = 1
    ) -> Tuple[GroundEntity, bool]:
        """
        Processes an incoming geolocated track sighting.
        Associates with existing ground entity if within match_radius_m, otherwise creates new entity.
        Returns: (matched_or_new_entity, is_new_entity)
        """
        self.total_raw_sightings += 1

        # Search for closest existing ground entity of matching class
        best_match_id: Optional[int] = None
        min_dist = float("inf")

        for e_id, entity in self.entities.items():
            if entity.class_name != class_name:
                continue
            dist = haversine_distance_m(lat, lon, entity.latitude, entity.longitude)
            if dist <= self.match_radius_m and dist < min_dist:
                min_dist = dist
                best_match_id = e_id

        # 1. Merge into existing entity
        if best_match_id is not None:
            self.total_duplicate_suppressions += 1
            entity = self.entities[best_match_id]
            
            # Update trajectory & observations
            entity.observation_count += 1
            entity.total_hits += hits
            entity.last_detected_ms = max(entity.last_detected_ms, timestamp_ms)
            if track_id not in entity.associated_track_ids:
                entity.associated_track_ids.append(track_id)
            entity.trajectory_coords.append((lat, lon))

            # Refine coordinates using confidence-weighted averaging
            total_weight = entity.best_confidence + confidence
            w_old = entity.best_confidence / total_weight
            w_new = confidence / total_weight
            entity.latitude = (w_old * entity.latitude) + (w_new * lat)
            entity.longitude = (w_old * entity.longitude) + (w_new * lon)

            # Update best visual crop if higher confidence
            if confidence > entity.best_confidence:
                entity.best_confidence = confidence
                entity.best_crop_bbox = list(crop_bbox)
                if frame_rgb is not None:
                    entity.best_frame = frame_rgb.copy()

            # Update thermal signature
            if thermal_delta is not None:
                if entity.thermal_delta is None or thermal_delta > entity.thermal_delta:
                    entity.thermal_delta = thermal_delta

            # Re-evaluate stationary status based on total spatial displacement
            if len(entity.trajectory_coords) >= 2:
                init_lat, init_lon = entity.trajectory_coords[0]
                curr_lat, curr_lon = entity.trajectory_coords[-1]
                net_disp = haversine_distance_m(init_lat, init_lon, curr_lat, curr_lon)
                entity.is_stationary = (net_disp < self.stationary_disp_thresh_m)
                entity.status = "STATIONARY" if entity.is_stationary else "MOBILE"

            return entity, False

        # 2. Register as a new ground entity
        new_entity = GroundEntity(
            entity_id=self._next_entity_id,
            class_name=class_name,
            latitude=lat,
            longitude=lon,
            best_confidence=confidence,
            is_stationary=is_stationary,
            status="STATIONARY" if is_stationary else "MOBILE",
            thermal_delta=thermal_delta,
            error_radius_m=error_radius_m,
            first_detected_ms=timestamp_ms,
            last_detected_ms=timestamp_ms,
            total_hits=hits,
            associated_track_ids=[track_id],
            best_crop_bbox=list(crop_bbox),
            best_frame=frame_rgb.copy() if frame_rgb is not None else None,
            observation_count=1,
            trajectory_coords=[(lat, lon)]
        )
        self.entities[self._next_entity_id] = new_entity
        self._next_entity_id += 1
        return new_entity, True

    def get_all_entities(self, confirmed_only: bool = True) -> List[GroundEntity]:
        """Returns sorted list of all deduplicated ground entities."""
        results = []
        for e in self.entities.values():
            if confirmed_only and e.total_hits < self.min_hits_to_confirm:
                continue
            results.append(e)
        # Sort by confidence / hit count descending
        results.sort(key=lambda x: (x.total_hits, x.best_confidence), reverse=True)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Returns deduplication performance statistics."""
        confirmed_count = len(self.get_all_entities(confirmed_only=True))
        all_count = len(self.entities)
        dup_pct = 0.0
        if self.total_raw_sightings > 0:
            dup_pct = (self.total_duplicate_suppressions / self.total_raw_sightings) * 100.0

        return {
            "total_raw_sightings": self.total_raw_sightings,
            "total_duplicate_suppressions": self.total_duplicate_suppressions,
            "deduplication_ratio_pct": round(dup_pct, 2),
            "total_ground_entities": all_count,
            "confirmed_unique_survivors": confirmed_count,
            "match_radius_m": self.match_radius_m
        }
