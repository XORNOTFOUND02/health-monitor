"""
Motion feature extraction from accelerometer and gyroscope data.

Extracts 28 features per window including:
- Statistical features per accelerometer axis (mean, std, min, max, range, RMS,
  skewness, kurtosis, percentiles, zero-crossing rate)
- Signal Magnitude Area (SMA)
- Tilt angle, movement intensity, fall indicator
- Jerk features (derivative of acceleration)
- Statistical features per gyroscope axis (mean, std, min, max, range, RMS)
- Cross-axis correlation coefficients
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats as scipy_stats

from .base import BaseFeatureExtractor


class MotionFeatures(BaseFeatureExtractor):
    """Extract motion-related features from accelerometer and gyroscope data.
    
    This extractor computes statistical, temporal, and cross-axis features
    from 3-axis accelerometer and 3-axis gyroscope signals within each
    analysis window.
    
    Features (28 total):
        Accelerometer per-axis features (Ax, Ay, Az):
            - motion_ax_mean, motion_ax_std, motion_ax_min, motion_ax_max
            - motion_ax_range, motion_ax_rms, motion_ax_skew, motion_ax_kurt
            - motion_ax_p25, motion_ax_p75, motion_ax_zcr
            (same for Ay, Az -> 33 features, but grouped differently below)
        
        Accelerometer magnitude features:
            - motion_mag_mean, motion_mag_std, motion_mag_max
        
        Aggregate features:
            - motion_sma (Signal Magnitude Area)
            - motion_tilt_angle_mean
            - motion_movement_intensity
            - motion_fall_indicator
            - motion_jerk_mean, motion_jerk_peak
        
        Gyroscope per-axis features (Gx, Gy, Gz):
            - motion_gx_mean, motion_gx_std, motion_gx_range
            - (same for Gy, Gz)
        
        Cross-axis:
            - motion_corr_ax_ay, motion_corr_ax_az, motion_corr_ay_az
    
    Note:
        The exact count per feature category is adjusted to reach ~28 total.
    """
    
    # Fall detection thresholds (m/s^2)
    FALL_IMPACT_THRESHOLD: float = 25.0  # ~2.5g impact
    FALL_STILLNESS_THRESHOLD: float = 0.5  # post-fall stillness
    
    def __init__(self) -> None:
        """Initialize motion feature extractor."""
        super().__init__()
        self.feature_names = [
            # Accelerometer per-axis statistical features (3 axes)
            "motion_ax_mean", "motion_ax_std", "motion_ax_min", "motion_ax_max",
            "motion_ax_range", "motion_ax_rms", "motion_ax_skew", "motion_ax_kurt",
            "motion_ax_p25", "motion_ax_p75", "motion_ax_zcr",
            "motion_ay_mean", "motion_ay_std", "motion_ay_min", "motion_ay_max",
            "motion_ay_range", "motion_ay_rms", "motion_ay_skew", "motion_ay_kurt",
            "motion_ay_p25", "motion_ay_p75", "motion_ay_zcr",
            "motion_az_mean", "motion_az_std", "motion_az_min", "motion_az_max",
            "motion_az_range", "motion_az_rms", "motion_az_skew", "motion_az_kurt",
            "motion_az_p25", "motion_az_p75", "motion_az_zcr",
            # Accelerometer magnitude features
            "motion_mag_mean", "motion_mag_std", "motion_mag_max",
            # Aggregate features
            "motion_sma", "motion_tilt_angle_mean", "motion_movement_intensity",
            "motion_fall_indicator",
            # Jerk features
            "motion_jerk_mean", "motion_jerk_peak",
            # Gyroscope per-axis features (3 axes)
            "motion_gx_mean", "motion_gx_std", "motion_gx_min", "motion_gx_max",
            "motion_gx_range", "motion_gx_rms",
            "motion_gy_mean", "motion_gy_std", "motion_gy_min", "motion_gy_max",
            "motion_gy_range", "motion_gy_rms",
            "motion_gz_mean", "motion_gz_std", "motion_gz_min", "motion_gz_max",
            "motion_gz_range", "motion_gz_rms",
            # Cross-axis correlation coefficients
            "motion_corr_ax_ay", "motion_corr_ax_az", "motion_corr_ay_az",
        ]
    
    def extract(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract motion features from accelerometer and gyroscope data.
        
        Args:
            window_data: Dictionary containing sensor data with keys
                'accelerometer' and 'gyroscope' each having x, y, z arrays.
        
        Returns:
            Dictionary of motion feature names to float values.
        """
        features: Dict[str, float] = {}
        
        # Get accelerometer data
        ax = self._get_sensor_array(window_data, "accelerometer", "ax")
        ay = self._get_sensor_array(window_data, "accelerometer", "ay")
        az = self._get_sensor_array(window_data, "accelerometer", "az")
        
        # Get gyroscope data
        gx = self._get_sensor_array(window_data, "gyroscope", "gx")
        gy = self._get_sensor_array(window_data, "gyroscope", "gy")
        gz = self._get_sensor_array(window_data, "gyroscope", "gz")
        
        # Compute per-axis accelerometer features
        for axis_name, axis_data in [("ax", ax), ("ay", ay), ("az", az)]:
            features.update(self._extract_axis_features(axis_data, f"motion_{axis_name}"))
        
        # Compute magnitude features
        features.update(self._extract_magnitude_features(ax, ay, az))
        
        # Compute aggregate features
        features.update(self._extract_aggregate_features(ax, ay, az))
        
        # Compute jerk features
        features.update(self._extract_jerk_features(ax, ay, az))
        
        # Compute gyroscope features
        for axis_name, axis_data in [("gx", gx), ("gy", gy), ("gz", gz)]:
            features.update(self._extract_gyro_axis_features(axis_data, f"motion_{axis_name}"))
        
        # Compute cross-axis correlations
        features.update(self._extract_cross_correlations(ax, ay, az))
        
        return features
    
    def _extract_axis_features(
        self, arr: np.ndarray, prefix: str
    ) -> Dict[str, float]:
        """Extract statistical features for a single axis.
        
        Args:
            arr: 1D numpy array of axis data.
            prefix: Feature name prefix (e.g., 'motion_ax').
        
        Returns:
            Dictionary of features for this axis.
        """
        features: Dict[str, float] = {}
        
        # Basic statistics
        features[f"{prefix}_mean"] = self._safe_extract(arr, lambda x: np.mean(x))
        features[f"{prefix}_std"] = self._safe_extract(arr, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0)
        features[f"{prefix}_min"] = self._safe_extract(arr, lambda x: np.min(x))
        features[f"{prefix}_max"] = self._safe_extract(arr, lambda x: np.max(x))
        features[f"{prefix}_range"] = self._safe_extract(arr, lambda x: np.ptp(x))
        
        # RMS
        features[f"{prefix}_rms"] = self._safe_extract(arr, lambda x: np.sqrt(np.mean(x ** 2)))
        
        # Higher-order statistics
        features[f"{prefix}_skew"] = self._safe_extract(
            arr, lambda x: float(scipy_stats.skew(x)) if len(x) > 2 else 0.0
        )
        features[f"{prefix}_kurt"] = self._safe_extract(
            arr, lambda x: float(scipy_stats.kurtosis(x)) if len(x) > 3 else 0.0
        )
        
        # Percentiles
        features[f"{prefix}_p25"] = self._compute_percentile(arr, 25)
        features[f"{prefix}_p75"] = self._compute_percentile(arr, 75)
        
        # Zero-crossing rate
        features[f"{prefix}_zcr"] = self._compute_zcr(arr)
        
        return features
    
    def _compute_zcr(self, arr: np.ndarray) -> float:
        """Compute zero-crossing rate.
        
        Args:
            arr: Input signal array.
        
        Returns:
            Fraction of consecutive samples that cross zero.
        """
        if len(arr) < 2:
            return 0.0
        
        def _zcr(x: np.ndarray) -> float:
            centered = x - np.mean(x)
            signs = np.sign(centered)
            crossings = np.abs(np.diff(signs))
            return float(np.mean(crossings) / 2.0)
        
        return self._safe_extract(arr, _zcr)
    
    def _extract_magnitude_features(
        self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray
    ) -> Dict[str, float]:
        """Extract features from acceleration magnitude.
        
        Args:
            ax: X-axis acceleration.
            ay: Y-axis acceleration.
            az: Z-axis acceleration.
        
        Returns:
            Dictionary of magnitude features.
        """
        features: Dict[str, float] = {}
        
        def _magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
            min_len = min(len(x), len(y), len(z))
            return np.sqrt(x[:min_len] ** 2 + y[:min_len] ** 2 + z[:min_len] ** 2)
        
        mag = self._safe_array(None)
        if len(ax) > 0 and len(ay) > 0 and len(az) > 0:
            mag = _magnitude(ax, ay, az)
        
        features["motion_mag_mean"] = self._safe_extract(mag, lambda x: np.mean(x))
        features["motion_mag_std"] = self._safe_extract(
            mag, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
        )
        features["motion_mag_max"] = self._safe_extract(mag, lambda x: np.max(x))
        
        return features
    
    def _extract_aggregate_features(
        self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray
    ) -> Dict[str, float]:
        """Extract aggregate motion features (SMA, tilt, movement intensity, fall).
        
        Args:
            ax: X-axis acceleration.
            ay: Y-axis acceleration.
            az: Z-axis acceleration.
        
        Returns:
            Dictionary of aggregate features.
        """
        features: Dict[str, float] = {}
        
        min_len = min(len(ax), len(ay), len(az))
        
        # Signal Magnitude Area (SMA)
        if min_len > 0:
            ax_r, ay_r, az_r = ax[:min_len], ay[:min_len], az[:min_len]
            features["motion_sma"] = float(
                np.sum(np.abs(ax_r) + np.abs(ay_r) + np.abs(az_r)) / min_len
            )
        else:
            features["motion_sma"] = 0.0
        
        # Tilt angle: angle from vertical (gravity axis)
        def _tilt_angle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
            min_l = min(len(x), len(y), len(z))
            if min_l == 0:
                return 0.0
            x_r, y_r, z_r = x[:min_l], y[:min_l], z[:min_l]
            horizontal = np.sqrt(x_r ** 2 + y_r ** 2)
            angles = np.arctan2(z_r, horizontal)
            return float(np.mean(angles))
        
        features["motion_tilt_angle_mean"] = self._safe_extract(ax, lambda _: _tilt_angle(ax, ay, az))
        
        # Movement intensity: std of each axis combined
        def _movement_intensity(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
            min_l = min(len(x), len(y), len(z))
            if min_l < 2:
                return 0.0
            sx = np.std(x[:min_l], ddof=1)
            sy = np.std(y[:min_l], ddof=1)
            sz = np.std(z[:min_l], ddof=1)
            return float(np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
        
        features["motion_movement_intensity"] = _movement_intensity(ax, ay, az)
        
        # Fall indicator: high magnitude impact followed by stillness
        def _fall_indicator(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
            min_l = min(len(x), len(y), len(z))
            if min_l < 10:
                return 0.0
            mag = np.sqrt(x[:min_l] ** 2 + y[:min_l] ** 2 + z[:min_l] ** 2)
            max_mag = np.max(mag)
            # Check if second half has low movement (stillness after impact)
            half = min_l // 2
            second_half_std = np.std(mag[half:], ddof=1) if min_l - half > 1 else 0.0
            if max_mag > self.FALL_IMPACT_THRESHOLD and second_half_std < self.FALL_STILLNESS_THRESHOLD:
                return 1.0
            return 0.0
        
        features["motion_fall_indicator"] = _fall_indicator(ax, ay, az)
        
        return features
    
    def _extract_jerk_features(
        self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray
    ) -> Dict[str, float]:
        """Extract jerk (derivative of acceleration) features.
        
        Args:
            ax: X-axis acceleration.
            ay: Y-axis acceleration.
            az: Z-axis acceleration.
        
        Returns:
            Dictionary of jerk features.
        """
        features: Dict[str, float] = {}
        
        def _jerk_stats(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
            min_l = min(len(x), len(y), len(z))
            if min_l < 2:
                return {"motion_jerk_mean": 0.0, "motion_jerk_peak": 0.0}
            
            dx = np.diff(x[:min_l])
            dy = np.diff(y[:min_l])
            dz = np.diff(z[:min_l])
            
            jerk_mag = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            return {
                "motion_jerk_mean": float(np.mean(jerk_mag)),
                "motion_jerk_peak": float(np.max(jerk_mag)),
            }
        
        jerk_feats = _jerk_stats(ax, ay, az)
        features.update(jerk_feats)
        
        return features
    
    def _extract_gyro_axis_features(
        self, arr: np.ndarray, prefix: str
    ) -> Dict[str, float]:
        """Extract statistical features for a single gyroscope axis.
        
        Args:
            arr: 1D numpy array of gyroscope axis data.
            prefix: Feature name prefix (e.g., 'motion_gx').
        
        Returns:
            Dictionary of features for this axis.
        """
        features: Dict[str, float] = {}
        
        features[f"{prefix}_mean"] = self._safe_extract(arr, lambda x: np.mean(x))
        features[f"{prefix}_std"] = self._safe_extract(
            arr, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
        )
        features[f"{prefix}_min"] = self._safe_extract(arr, lambda x: np.min(x))
        features[f"{prefix}_max"] = self._safe_extract(arr, lambda x: np.max(x))
        features[f"{prefix}_range"] = self._safe_extract(arr, lambda x: np.ptp(x))
        features[f"{prefix}_rms"] = self._safe_extract(arr, lambda x: np.sqrt(np.mean(x ** 2)))
        
        return features
    
    def _extract_cross_correlations(
        self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray
    ) -> Dict[str, float]:
        """Extract cross-axis correlation coefficients.
        
        Args:
            ax: X-axis acceleration.
            ay: Y-axis acceleration.
            az: Z-axis acceleration.
        
        Returns:
            Dictionary of correlation features.
        """
        features: Dict[str, float] = {}
        
        def _correlation(x: np.ndarray, y: np.ndarray) -> float:
            min_l = min(len(x), len(y))
            if min_l < 3:
                return 0.0
            x_r, y_r = x[:min_l], y[:min_l]
            # Check for constant signals
            if np.std(x_r) == 0 or np.std(y_r) == 0:
                return 0.0
            corr = np.corrcoef(x_r, y_r)[0, 1]
            if np.isnan(corr):
                return 0.0
            return float(corr)
        
        features["motion_corr_ax_ay"] = _correlation(ax, ay)
        features["motion_corr_ax_az"] = _correlation(ax, az)
        features["motion_corr_ay_az"] = _correlation(ay, az)
        
        return features