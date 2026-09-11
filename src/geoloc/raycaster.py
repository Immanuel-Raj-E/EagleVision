"""
raycaster.py
============
High-precision 3D Raycasting and Geolocation Engine for Aerial SAR.
Transforms 2D image pixel coordinates (u, v) into WGS84 Geodetic coordinates (Lat, Lon)
using synchronized UAV telemetry (altitude AGL, gimbal pitch, roll, yaw) and camera intrinsics.
Includes rigorous error budget estimation for sensor uncertainty.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import numpy as np

from src.ingest.telemetry_parser import TelemetrySample
from src.tracking.tracker import TrackedSurvivor


# WGS84 Ellipsoid Constants (EPSG:4326)
WGS84_A = 6378137.0           # Semi-major axis in meters
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # First eccentricity squared (~0.00669438)


@dataclass
class CameraIntrinsics:
    """
    Pinhole camera intrinsic matrix parameters for 4K aerial sensors.
    Default config: 4K (3840x2160) sensor with ~84° horizontal FOV (~24mm equivalent).
    """
    width: int = 3840
    height: int = 2160
    fx: float = 2132.395  # Focal length X in pixels: (width / 2) / tan(fov_h / 2)
    fy: float = 2132.395  # Focal length Y in pixels (square pixels)
    cx: float = 1920.0    # Principal point X (image center)
    cy: float = 1080.0    # Principal point Y (image center)
    fov_h_deg: float = 84.0

    @classmethod
    def from_fov(cls, width: int = 3840, height: int = 2160, fov_h_deg: float = 84.0) -> "CameraIntrinsics":
        """Calculates focal lengths automatically from horizontal Field of View (FOV)."""
        fov_rad = math.radians(fov_h_deg)
        fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        fy = fx  # Square sensor assumption
        cx = width / 2.0
        cy = height / 2.0
        return cls(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, fov_h_deg=fov_h_deg)

    def unproject(self, u: float, v: float) -> np.ndarray:
        """
        Unprojects a 2D pixel (u, v) to a 3D unit ray in camera optical coordinates:
        X_cam: Right, Y_cam: Down, Z_cam: Forward along optical axis.
        """
        x_c = (u - self.cx) / self.fx
        y_c = (v - self.cy) / self.fy
        z_c = 1.0
        norm = math.sqrt(x_c * x_c + y_c * y_c + z_c * z_c)
        return np.array([x_c / norm, y_c / norm, z_c / norm], dtype=np.float64)


@dataclass
class GeolocatedSurvivor:
    """
    Final geolocated survivor record actioned for SAR rescue dispatch.
    Contains confirmed persistent track ID, high-precision WGS84 coordinates,
    confidence, movement risk status, and sensory error bounds.
    """
    track_id: int
    class_name: str
    confidence: float
    latitude: float
    longitude: float
    error_radius_m: float
    is_stationary: bool
    thermal_delta: Optional[float]
    crop_bbox: list[float]
    timestamp_ms: float
    ground_distance_m: float = 0.0


def ned_to_wgs84(lat0: float, lon0: float, delta_north_m: float, delta_east_m: float) -> Tuple[float, float]:
    """
    Converts local North-East metric displacements into WGS84 Geodetic Coordinates (Lat, Lon)
    using ellipsoidal radii of curvature.
    """
    lat0_rad = math.radians(lat0)
    sin_lat = math.sin(lat0_rad)
    sin2_lat = sin_lat * sin_lat

    # Meridian radius of curvature (North-South)
    m = (WGS84_A * (1.0 - WGS84_E2)) / math.pow(1.0 - WGS84_E2 * sin2_lat, 1.5)
    # Prime vertical radius of curvature (East-West)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin2_lat)

    delta_lat_rad = delta_north_m / m
    delta_lon_rad = delta_east_m / (n * math.cos(lat0_rad)) if math.cos(lat0_rad) != 0 else 0.0

    target_lat = lat0 + math.degrees(delta_lat_rad)
    target_lon = lon0 + math.degrees(delta_lon_rad)

    return round(target_lat, 7), round(target_lon, 7)


def wgs84_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes accurate geodetic distance in meters between two WGS84 coordinates
    using local ellipsoidal radii of curvature.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lam1, lam2 = math.radians(lon1), math.radians(lon2)
    mean_lat = (phi1 + phi2) / 2.0
    sin_lat = math.sin(mean_lat)
    sin2_lat = sin_lat * sin_lat

    # Meridian and prime-vertical radii of curvature
    m = (WGS84_A * (1.0 - WGS84_E2)) / math.pow(1.0 - WGS84_E2 * sin2_lat, 1.5)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin2_lat)

    delta_north = m * (phi2 - phi1)
    delta_east = n * math.cos(mean_lat) * (lam2 - lam1)

    return math.hypot(delta_north, delta_east)


class GeoRaycaster:
    """
    High-speed 3D Raycaster that maps image coordinates to Earth ground plane.
    Supports arbitrary gimbal attitudes, flight elevations, and intrinsics.
    Includes zero-allocation caching for multi-detection frame processing.
    """

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        gimbal_err_deg: float = 1.0,
        gps_cep_err_m: float = 1.5,
        alt_err_m: float = 0.5
    ):
        self.intrinsics = intrinsics or CameraIntrinsics.from_fov(3840, 2160, 84.0)
        self.gimbal_err_deg = gimbal_err_deg
        self.gps_cep_err_m = gps_cep_err_m
        self.alt_err_m = alt_err_m

        # Pre-cached attitude and geodetic curvature states
        self._cached_ypr: Optional[Tuple[float, float, float]] = None
        self._cached_R: Optional[np.ndarray] = None
        
        self._cached_lat0: Optional[float] = None
        self._cached_m: float = 0.0
        self._cached_n_cos: float = 0.0

    def compute_rotation_matrix(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        """
        Constructs the 3D rotation matrix from Camera Frame to local NED World Frame.
        Caches result if attitude is unchanged.
        """
        ypr_key = (yaw_deg, pitch_deg, roll_deg)
        if self._cached_ypr == ypr_key and self._cached_R is not None:
            return self._cached_R

        psi = math.radians(yaw_deg)
        theta = math.radians(pitch_deg)
        phi = math.radians(roll_deg)

        c_psi, s_psi = math.cos(psi), math.sin(psi)
        c_th, s_th = math.cos(theta), math.sin(theta)
        c_ph, s_ph = math.cos(phi), math.sin(phi)

        # Standard ZYX Euler (Yaw * Pitch * Roll) rotation matrix entries:
        r00 = c_psi * c_th
        r01 = c_psi * s_th * s_ph - s_psi * c_ph
        r02 = c_psi * s_th * c_ph + s_psi * s_ph

        r10 = s_psi * c_th
        r11 = s_psi * s_th * s_ph + c_psi * c_ph
        r12 = s_psi * s_th * c_ph - c_psi * s_ph

        r20 = -s_th
        r21 = c_th * s_ph
        r22 = c_th * c_ph

        # Post-multiplied by R_cam_to_body:
        # Col 0 (Cam Right)   -> Col 1 of R_ypr
        # Col 1 (Cam Down)    -> Col 2 of R_ypr
        # Col 2 (Cam Forward) -> Col 0 of R_ypr
        R = np.array([
            [r01, r02, r00],
            [r11, r12, r10],
            [r21, r22, r20]
        ], dtype=np.float64)

        self._cached_ypr = ypr_key
        self._cached_R = R
        return R

    def compute_error_budget(self, altitude_agl: float, look_angle_deg: float = 0.0) -> float:
        """
        Calculates theoretical geospatial error radius in meters given sensor uncertainties.
        Error = altitude_agl * tan(theta_err) + GPS_err + alt_err * tan(look_angle)
        """
        theta_err_rad = math.radians(self.gimbal_err_deg)
        look_angle_rad = math.radians(abs(look_angle_deg))
        
        angular_drift = altitude_agl * math.tan(theta_err_rad)
        altitude_drift = self.alt_err_m * math.tan(look_angle_rad)
        total_error_m = angular_drift + self.gps_cep_err_m + altitude_drift
        return round(total_error_m, 2)

    def _get_geodetic_curvature(self, lat0: float) -> Tuple[float, float]:
        """Returns cached meridian (m) and parallel (n*cos) radii of curvature."""
        if self._cached_lat0 == lat0:
            return self._cached_m, self._cached_n_cos

        lat0_rad = math.radians(lat0)
        sin_lat = math.sin(lat0_rad)
        sin2_lat = sin_lat * sin_lat

        m = (WGS84_A * (1.0 - WGS84_E2)) / math.pow(1.0 - WGS84_E2 * sin2_lat, 1.5)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin2_lat)
        n_cos = n * math.cos(lat0_rad) if math.cos(lat0_rad) != 0 else 1.0

        self._cached_lat0 = lat0
        self._cached_m = m
        self._cached_n_cos = n_cos
        return m, n_cos

    def raycast_pixel(
        self,
        u: float,
        v: float,
        drone_lat: float,
        drone_lon: float,
        altitude_agl: float,
        pitch: float = -90.0,
        roll: float = 0.0,
        yaw: float = 0.0,
        pitch_deg: Optional[float] = None,
        roll_deg: Optional[float] = None,
        yaw_deg: Optional[float] = None,
        ground_elevation_m: float = 0.0,
        rotation_matrix: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float, float]:
        """
        Raycasts a single 2D pixel (u, v) to ground WGS84 geocoordinates.
        
        Returns:
            Tuple[float, float, float, float]: (target_lat, target_lon, error_radius_m, ground_dist_m)
        """
        pitch_val = pitch_deg if pitch_deg is not None else pitch
        roll_val = roll_deg if roll_deg is not None else roll
        yaw_val = yaw_deg if yaw_deg is not None else yaw

        # 1. Unproject pixel to normalized camera ray
        x_c = (u - self.intrinsics.cx) / self.intrinsics.fx
        y_c = (v - self.intrinsics.cy) / self.intrinsics.fy
        z_c = 1.0

        # 2. Rotate ray into World NED Frame
        R = rotation_matrix if rotation_matrix is not None else self.compute_rotation_matrix(yaw_val, pitch_val, roll_val)
        
        v_north = R[0, 0] * x_c + R[0, 1] * y_c + R[0, 2] * z_c
        v_east  = R[1, 0] * x_c + R[1, 1] * y_c + R[1, 2] * z_c
        v_down  = R[2, 0] * x_c + R[2, 1] * y_c + R[2, 2] * z_c

        # Effective vertical height to ground plane
        effective_alt = max(1.0, altitude_agl - ground_elevation_m)

        # 3. Ray-Ground Intersection
        if v_down <= 1e-6:
            v_down = 1e-6

        scale_lambda = effective_alt / v_down
        delta_north = scale_lambda * v_north
        delta_east = scale_lambda * v_east

        ground_dist = math.hypot(delta_north, delta_east)

        # 4. Project relative NED displacement to WGS84 (Lat, Lon) using cached curvature
        m, n_cos = self._get_geodetic_curvature(drone_lat)
        target_lat = drone_lat + math.degrees(delta_north / m)
        target_lon = drone_lon + math.degrees(delta_east / n_cos)

        # 5. Compute Error Budget
        look_angle_deg = math.degrees(math.atan2(ground_dist, effective_alt))
        error_radius_m = self.compute_error_budget(altitude_agl, look_angle_deg)

        return round(target_lat, 7), round(target_lon, 7), error_radius_m, round(ground_dist, 2)

    def geolocate_track(
        self,
        track: TrackedSurvivor,
        telemetry: Union[TelemetrySample, dict],
        timestamp_ms: float,
        ground_elevation_m: float = 0.0
    ) -> GeolocatedSurvivor:
        """
        Geolocates a tracked survivor bounding box center to Earth GPS coordinates.
        """
        bbox = track.current_bbox
        center_u = (bbox[0] + bbox[2]) / 2.0
        center_v = (bbox[1] + bbox[3]) / 2.0

        if isinstance(telemetry, TelemetrySample):
            lat = telemetry.lat
            lon = telemetry.lon
            alt = telemetry.altitude_agl
            pitch = telemetry.pitch
            roll = telemetry.roll
            yaw = telemetry.yaw
        elif isinstance(telemetry, dict):
            lat = float(telemetry.get("lat", 0.0))
            lon = float(telemetry.get("lon", 0.0))
            alt = float(telemetry.get("altitude_agl", 50.0))
            pitch = float(telemetry.get("pitch", -90.0))
            roll = float(telemetry.get("roll", 0.0))
            yaw = float(telemetry.get("yaw", 0.0))
        else:
            raise ValueError("Unsupported telemetry format")

        target_lat, target_lon, error_radius_m, ground_dist = self.raycast_pixel(
            u=center_u,
            v=center_v,
            drone_lat=lat,
            drone_lon=lon,
            altitude_agl=alt,
            pitch=pitch,
            roll=roll,
            yaw=yaw,
            ground_elevation_m=ground_elevation_m
        )

        return GeolocatedSurvivor(
            track_id=track.track_id,
            class_name=track.class_name,
            confidence=track.confidence,
            latitude=target_lat,
            longitude=target_lon,
            error_radius_m=error_radius_m,
            is_stationary=track.is_stationary,
            thermal_delta=track.thermal_delta,
            crop_bbox=list(bbox),
            timestamp_ms=timestamp_ms,
            ground_distance_m=ground_dist
        )

    def geolocate_tracks(
        self,
        tracks: List[TrackedSurvivor],
        telemetry: Union[TelemetrySample, dict],
        timestamp_ms: float,
        confirmed_only: bool = False
    ) -> List[GeolocatedSurvivor]:
        """
        Batch geolocates multiple tracked survivors in a video frame.
        """
        results: List[GeolocatedSurvivor] = []
        for t in tracks:
            if confirmed_only and not t.is_confirmed:
                continue
            if t.status == "LOST":
                continue
            geoloc = self.geolocate_track(t, telemetry, timestamp_ms)
            results.append(geoloc)
        return results
