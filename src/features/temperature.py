"""
Temperature feature extraction from dual temperature sensors.

Extracts 8 features from STTS22H (digital) and LM35 (analog) temperature
sensors including:
- Mean temperatures from both sensors
- Temperature difference (sensor disagreement)
- Max/min temperature
- Temperature variance
- Fever indicators (>= 38°C and >= 37.5°C)
- Rate of temperature change
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats as scipy_stats

from .base import BaseFeatureExtractor


class TemperatureFeatures(BaseFeatureExtractor):
    """Extract temperature features from dual temperature sensors.
    
    This extractor computes features from both STTS22H (digital) and
    LM35 (analog) temperature sensors within each analysis window.
    
    Features (8 total):
        temp_stts22h_mean: Mean temperature from STTS22H sensor
        temp_lm35_mean: Mean temperature from LM35 sensor
        temp_difference: STTS22H - LM35 (sensor disagreement)
        temp_max: Maximum temperature across both sensors
        temp_min: Minimum temperature across both sensors
        temp_variance: Variance of combined temperature data
        temp_fever_indicator: Binary (1.0 if max temp >= 38°C)
        temp_low_grade_fever_indicator: Binary (1.0 if max temp >= 37.5°C)
        temp_rate_of_change: Rate of temperature change (slope)
    
    Clinical thresholds:
        - Normal: < 37.5°C
        - Low-grade fever: >= 37.5°C
        - Fever: >= 38°C
    """
    
    FEVER_THRESHOLD: float = 38.0
    LOW_GRADE_FEVER_THRESHOLD: float = 37.5
    
    def __init__(self) -> None:
        """Initialize temperature feature extractor."""
        super().__init__()
        self.feature_names = [
            "temp_stts22h_mean", "temp_lm35_mean",
            "temp_difference", "temp_max", "temp_min",
            "temp_variance", "temp_fever_indicator",
            "temp_low_grade_fever_indicator", "temp_rate_of_change",
        ]
    
    def extract(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract temperature features from both sensors.
        
        Args:
            window_data: Dictionary containing temperature data with keys
                'temperature' -> 'stts22h_celsius' and 'lm35_celsius'.
        
        Returns:
            Dictionary of temperature feature names to float values.
        """
        features: Dict[str, float] = {}
        
        # Get temperature data from both sensors
        stts22h = self._get_sensor_array(window_data, "temperature", "stts22h_celsius")
        lm35 = self._get_sensor_array(window_data, "temperature", "lm35_celsius")
        
        # Filter to physiologically plausible values (20-45°C)
        if len(stts22h) > 0:
            valid_mask = (stts22h >= 20) & (stts22h <= 45)
            stts22h = stts22h[valid_mask]
        
        if len(lm35) > 0:
            valid_mask = (lm35 >= 20) & (lm35 <= 45)
            lm35 = lm35[valid_mask]
        
        # Mean temperatures
        features["temp_stts22h_mean"] = self._safe_extract(stts22h, lambda x: np.mean(x))
        features["temp_lm35_mean"] = self._safe_extract(lm35, lambda x: np.mean(x))
        
        # Temperature difference (sensor disagreement)
        features["temp_difference"] = self._compute_difference(stts22h, lm35)
        
        # Combine both sensors for max/min/variance
        combined = self._combine_sensors(stts22h, lm35)
        
        features["temp_max"] = self._safe_extract(combined, lambda x: np.max(x))
        features["temp_min"] = self._safe_extract(combined, lambda x: np.min(x))
        features["temp_variance"] = self._safe_extract(
            combined, lambda x: np.var(x) if len(x) > 0 else 0.0
        )
        
        # Fever indicators (based on max temperature)
        max_temp = features["temp_max"]
        features["temp_fever_indicator"] = 1.0 if max_temp >= self.FEVER_THRESHOLD else 0.0
        features["temp_low_grade_fever_indicator"] = (
            1.0 if max_temp >= self.LOW_GRADE_FEVER_THRESHOLD else 0.0
        )
        
        # Rate of temperature change
        features["temp_rate_of_change"] = self._compute_rate_of_change(stts22h, lm35)
        
        return features
    
    def _compute_difference(
        self, stts22h: np.ndarray, lm35: np.ndarray
    ) -> float:
        """Compute temperature difference between sensors.
        
        Uses mean of both sensors if available, otherwise returns 0.
        
        Args:
            stts22h: STTS22H temperature array.
            lm35: LM35 temperature array.
        
        Returns:
            Temperature difference (STTS22H - LM35).
        """
        mean_stts = np.mean(stts22h) if len(stts22h) > 0 else None
        mean_lm = np.mean(lm35) if len(lm35) > 0 else None
        
        if mean_stts is not None and mean_lm is not None:
            return float(mean_stts - mean_lm)
        return 0.0
    
    def _combine_sensors(
        self, stts22h: np.ndarray, lm35: np.ndarray
    ) -> np.ndarray:
        """Combine data from both sensors into a single array.
        
        Args:
            stts22h: STTS22H temperature array.
            lm35: LM35 temperature array.
        
        Returns:
            Combined array of temperature values.
        """
        arrays = []
        if len(stts22h) > 0:
            arrays.append(stts22h)
        if len(lm35) > 0:
            arrays.append(lm35)
        
        if arrays:
            return np.concatenate(arrays)
        return np.array([], dtype=np.float64)
    
    def _compute_rate_of_change(
        self, stts22h: np.ndarray, lm35: np.ndarray
    ) -> float:
        """Compute rate of temperature change using linear regression.
        
        Uses STTS22H data if available, otherwise LM35.
        
        Args:
            stts22h: STTS22H temperature array.
            lm35: LM35 temperature array.
        
        Returns:
            Slope of temperature change (degrees per sample).
        """
        # Prefer STTS22H (digital sensor, usually more accurate)
        temp_data = stts22h if len(stts22h) > 1 else lm35
        
        if len(temp_data) < 2:
            return 0.0
        
        def _slope(x: np.ndarray) -> float:
            t = np.arange(len(x))
            slope, _, _, _, _ = scipy_stats.linregress(t, x)
            return float(slope)
        
        return self._safe_extract(temp_data, _slope)