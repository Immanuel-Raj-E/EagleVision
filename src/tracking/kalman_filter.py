"""
kalman_filter.py
================
High-performance 2D Kalman Filter optimized for bounding box tracking in SAR vision.
State vector: [x, y, a, h, vx, vy, va, vh]^T
"""

import numpy as np


class KalmanBoxTracker:
    """
    Optimized constant-velocity Kalman filter with closed-form state updates.
    """

    def __init__(self, bbox_xyxy: list[float]):
        self.dim_x = 8
        self.dim_z = 4

        # State vector [x, y, a, h, vx, vy, va, vh]^T
        self.x = np.zeros((8, 1), dtype=np.float32)
        w = max(1.0, float(bbox_xyxy[2] - bbox_xyxy[0]))
        h = max(1.0, float(bbox_xyxy[3] - bbox_xyxy[1]))
        cx = float(bbox_xyxy[0] + w / 2.0)
        cy = float(bbox_xyxy[1] + h / 2.0)
        self.x[0, 0] = cx
        self.x[1, 0] = cy
        self.x[2, 0] = w / h
        self.x[3, 0] = h

        # Covariance matrices
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0

        self.R = np.eye(4, dtype=np.float32)
        self.R[0:2, 0:2] *= 1.0
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 1.0

        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.Q[4:, 4:] *= 0.01

    def predict(self) -> list[float]:
        """Advances state vector: x_pos += v_vel."""
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0

        # Fast analytical constant velocity transition
        self.x[0:4] += self.x[4:8]

        # P = F * P * F^T + Q
        # Analytical block multiply
        self.P[0:4, 0:4] += self.P[0:4, 4:8] + self.P[4:8, 0:4] + self.P[4:8, 4:8]
        self.P[0:4, 4:8] += self.P[4:8, 4:8]
        self.P[4:8, 0:4] += self.P[4:8, 4:8]
        self.P += self.Q

        return self.get_state()

    def update(self, bbox_xyxy: list[float]):
        """Updates state vector with observed bounding box measurement."""
        w = max(1.0, float(bbox_xyxy[2] - bbox_xyxy[0]))
        h = max(1.0, float(bbox_xyxy[3] - bbox_xyxy[1]))
        z = np.array([
            [float(bbox_xyxy[0] + w / 2.0)],
            [float(bbox_xyxy[1] + h / 2.0)],
            [w / h],
            [h]
        ], dtype=np.float32)

        # Innovation: y = z - Hx (where Hx is simply x[0:4])
        y = z - self.x[0:4]

        # Innovation covariance: S = H * P * H^T + R = P[0:4, 0:4] + R
        S = self.P[0:4, 0:4] + self.R

        # Kalman gain: K = P * H^T * inv(S) = P[:, 0:4] * inv(S)
        # Using fast solve
        try:
            K = np.linalg.solve(S.T, self.P[:, 0:4].T).T
        except np.linalg.LinAlgError:
            K = np.dot(self.P[:, 0:4], np.linalg.pinv(S))

        # State and covariance update
        self.x += np.dot(K, y)
        self.P -= np.dot(K, self.P[0:4, :])

    def get_state(self) -> list[float]:
        """Returns [x1, y1, x2, y2] bounding box."""
        w = float(self.x[2, 0] * self.x[3, 0])
        h = float(self.x[3, 0])
        x1 = float(self.x[0, 0] - w / 2.0)
        y1 = float(self.x[1, 0] - h / 2.0)
        return [round(x1, 2), round(y1, 2), round(x1 + w, 2), round(y1 + h, 2)]
