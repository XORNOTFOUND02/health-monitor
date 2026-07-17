"""
Feature extraction from HMC5883L 3-axis magnetometer data.

Provides ~20 features: statistical, frequency-domain, and derived
heading/movement signatures for health symptom detection.

Expected input: dict with key "magnetometer" containing a numpy array
of shape (n, 3) — columns are [Mx, My, Mz] in microtesla.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal, stats

logger = logging.getLogger(__name__)


def extract_magnetometer_features(
    magnetometer: np.ndarray,
    sample_rate: float = 25.0,
) -> Dict[str, float]:
    """Extract features from 3-axis magnetometer data.
    
    Parameters
    ----------
    magnetometer : np.ndarray
        Shape (n, 3) — [Mx, My, Mz] in microtesla.
    sample_rate : float
        Sampling rate in Hz (default 25 Hz).
    
    Returns
    -------
    dict
        Feature name -> scalar value.
    """
    features: Dict[str, float] = {}
    
    if magnetometer.size == 0 or magnetometer.shape[0] < 5:
        logger.warning("Magnetometer data too short (%d samples)", magnetometer.shape[0] if magnetometer.size > 0 else 0)
        return _default_features()
    
    mx, my, mz = magnetometer[:, 0], magnetometer[:, 1], magnetometer[:, 2]
    
    # ---- 1. Basic statistics per axis (9 features) ----
    for axis_name, axis_data in [("mx", mx), ("my", my), ("mz", mz)]:
        features[f"mag_{axis_name}_mean"] = float(np.mean(axis_data))
        features[f"mag_{axis_name}_std"] = float(np.std(axis_data))
        features[f"mag_{axis_name}_min"] = float(np.min(axis_data))
        features[f"mag_{axis_name}_max"] = float(np.max(axis_data))
        features[f"mag_{axis_name}_range"] = float(np.ptp(axis_data))
    
    # ---- 2. Magnetic field magnitude (1 feature) ----
    magnitude = np.sqrt(mx**2 + my**2 + mz**2)
    features["mag_magnitude_mean"] = float(np.mean(magnitude))
    features["mag_magnitude_std"] = float(np.std(magnitude))
    
    # ---- 3. Heading (compass direction) features (3 features) ----
    # Heading = atan2(My, Mx) in degrees
    heading = np.rad2deg(np.arctan2(my, mx)) % 360
    features["mag_heading_mean"] = float(np.mean(heading))
    features["mag_heading_std"] = float(np.std(heading))
    # Heading change rate (degrees per second)
    heading_diff = np.diff(heading)
    # Handle wrap-around (e.g., 359 -> 1 should be diff of 2, not -358)
    heading_diff = (heading_diff + 180) % 360 - 180
    features["mag_heading_change_rate"] = float(np.mean(np.abs(heading_diff))) * sample_rate
    
    # ---- 4. Frequency-domain features (3 features) ----
    freqs, psd = signal.periodogram(magnitude, fs=sample_rate)
    if len(psd) > 0:
        features["mag_dominant_freq"] = float(freqs[np.argmax(psd)])
        features["mag_power_total"] = float(np.sum(psd))
        # Power in step frequency band (1-3.5 Hz)
        step_band = (freqs >= 1.0) & (freqs <= 3.5)
        if np.any(step_band):
            features["mag_power_step_band"] = float(np.sum(psd[step_band]))
        else:
            features["mag_power_step_band"] = 0.0
    else:
        features["mag_dominant_freq"] = 0.0
        features["mag_power_total"] = 0.0
        features["mag_power_step_band"] = 0.0
    
    # ---- 5. Movement/change features (2 features) ----
    # Magnetic field change magnitude (norm of diff)
    mag_diff = np.sqrt(np.sum(np.diff(magnetometer, axis=0)**2, axis=1))
    features["mag_change_mean"] = float(np.mean(mag_diff))
    features["mag_change_max"] = float(np.max(mag_diff))
    
    return features


def _default_features() -> Dict[str, float]:
    """Return zero-valued features when data is unavailable."""
    return {
        "mag_mx_mean": 0.0, "mag_mx_std": 0.0, "mag_mx_min": 0.0, "mag_mx_max": 0.0, "mag_mx_range": 0.0,
        "mag_my_mean": 0.0, "mag_my_std": 0.0, "mag_my_min": 0.0, "mag_my_max": 0.0, "mag_my_range": 0.0,
        "mag_mz_mean": 0.0, "mag_mz_std": 0.0, "mag_mz_min": 0.0, "mag_mz_max": 0.0, "mag_mz_range": 0.0,
        "mag_magnitude_mean": 0.0, "mag_magnitude_std": 0.0,
        "mag_heading_mean": 0.0, "mag_heading_std": 0.0, "mag_heading_change_rate": 0.0,
        "mag_dominant_freq": 0.0, "mag_power_total": 0.0, "mag_power_step_band": 0.0,
        "mag_change_mean": 0.0, "mag_change_max": 0.0,
    }


def list_mag_feature_names() -> List[str]:
    """Return all magnetometer feature names (for feature registration)."""
    return list(_default_features().keys())
