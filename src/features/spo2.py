"""
SpO2 (blood oxygen saturation) feature extraction.

Extracts 10 features from SpO2 time series including:
- Basic statistics (mean, std, min)
- Hypoxemia indicators (% time < 95%, < 90%)
- SpO2 nadir (minimum value)
- Desaturation event count
- SpO2 variability
- SpO2 trend
- Oxygen desaturation index (ODI)
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats as scipy_stats

from .base import BaseFeatureExtractor


class SpO2Features(BaseFeatureExtractor):
    """Extract SpO2 features from blood oxygen saturation time series.
    
    This extractor computes statistical and clinical features from SpO2
    measurements within each analysis window.
    
    Features (10 total):
        spo2_mean, spo2_std, spo2_min
        spo2_hypoxemia_pct (% time < 95%)
        spo2_severe_hypoxemia_pct (% time < 90%)
        spo2_nadir (minimum value)
        spo2_drop_count (desaturation events > 3% drop)
        spo2_variability (coeff of variation)
        spo2_trend_slope
        spo2_odi (desaturation events per hour, approximated)
    
    Clinical thresholds:
        - Normal SpO2: >= 95%
        - Hypoxemia: < 95%
        - Severe hypoxemia: < 90%
        - Desaturation event: > 3% drop from local baseline
    """
    
    HYPOXEMIA_THRESHOLD: float = 95.0
    SEVERE_HYPOXEMIA_THRESHOLD: float = 90.0
    DESATURATION_DROP_THRESHOLD: float = 3.0  # percent drop
    
    def __init__(self) -> None:
        """Initialize SpO2 feature extractor."""
        super().__init__()
        self.feature_names = [
            "spo2_mean", "spo2_std", "spo2_min",
            "spo2_hypoxemia_pct", "spo2_severe_hypoxemia_pct",
            "spo2_nadir", "spo2_drop_count",
            "spo2_variability", "spo2_trend_slope",
            "spo2_odi",
        ]
    
    def extract(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract SpO2 features from oxygen saturation time series.
        
        Args:
            window_data: Dictionary containing SpO2 data with key
                'heart_rate' -> 'spo2' array.
        
        Returns:
            Dictionary of SpO2 feature names to float values.
        """
        features: Dict[str, float] = {}
        
        # Get SpO2 data
        spo2 = self._get_sensor_array(window_data, "heart_rate", "spo2")
        
        # Filter to physiologically plausible values (50-100%)
        if len(spo2) > 0:
            valid_mask = (spo2 >= 50) & (spo2 <= 100)
            spo2 = spo2[valid_mask]
        
        if len(spo2) == 0:
            for name in self.feature_names:
                features[name] = 0.0
            return features
        
        # Basic statistics
        features["spo2_mean"] = self._safe_extract(spo2, lambda x: np.mean(x))
        features["spo2_std"] = self._safe_extract(
            spo2, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
        )
        features["spo2_min"] = self._safe_extract(spo2, lambda x: np.min(x))
        
        # Hypoxemia indicators
        features["spo2_hypoxemia_pct"] = self._compute_pct_below(
            spo2, self.HYPOXEMIA_THRESHOLD
        )
        features["spo2_severe_hypoxemia_pct"] = self._compute_pct_below(
            spo2, self.SEVERE_HYPOXEMIA_THRESHOLD
        )
        
        # SpO2 nadir (minimum value)
        features["spo2_nadir"] = features["spo2_min"]
        
        # Desaturation event count
        features["spo2_drop_count"] = self._compute_drop_count(spo2)
        
        # SpO2 variability (coefficient of variation)
        features["spo2_variability"] = self._compute_variability(spo2)
        
        # SpO2 trend (slope)
        features["spo2_trend_slope"] = self._compute_trend(spo2)
        
        # Oxygen Desaturation Index (approximated)
        features["spo2_odi"] = self._compute_odi(spo2)
        
        return features
    
    def _compute_pct_below(self, spo2: np.ndarray, threshold: float) -> float:
        """Compute percentage of time SpO2 is below a threshold.
        
        Args:
            spo2: SpO2 array.
            threshold: SpO2 threshold.
        
        Returns:
            Fraction of samples below threshold (0.0 to 1.0).
        """
        def _pct(x: np.ndarray) -> float:
            return float(np.mean(x < threshold))
        return self._safe_extract(spo2, _pct)
    
    def _compute_drop_count(self, spo2: np.ndarray) -> float:
        """Count desaturation events (> 3% drop from local baseline).
        
        A desaturation event is defined as a drop of more than 3 percentage
        points from a local maximum.
        
        Args:
            spo2: SpO2 array.
        
        Returns:
            Count of desaturation events as float.
        """
        if len(spo2) < 5:
            return 0.0
        
        def _drops(x: np.ndarray) -> float:
            # Find local maxima
            from scipy.signal import find_peaks
            
            # Use a simple approach: compare each sample to rolling maximum
            window_size = max(5, len(x) // 10)
            if window_size > len(x):
                window_size = len(x)
            
            drop_count = 0
            in_drop = False
            
            # Rolling window to establish local baseline
            for i in range(window_size, len(x)):
                local_baseline = np.max(x[max(0, i - window_size):i])
                drop = local_baseline - x[i]
                
                if drop > self.DESATURATION_DROP_THRESHOLD:
                    if not in_drop:
                        drop_count += 1
                        in_drop = True
                else:
                    in_drop = False
            
            return float(drop_count)
        
        return self._safe_extract(spo2, _drops)
    
    def _compute_variability(self, spo2: np.ndarray) -> float:
        """Compute SpO2 variability (coefficient of variation).
        
        Args:
            spo2: SpO2 array.
        
        Returns:
            Coefficient of variation (std/mean).
        """
        def _cv(x: np.ndarray) -> float:
            mean_val = np.mean(x)
            if mean_val == 0:
                return 0.0
            return float(np.std(x, ddof=1) / mean_val) if len(x) > 1 else 0.0
        return self._safe_extract(spo2, _cv)
    
    def _compute_trend(self, spo2: np.ndarray) -> float:
        """Compute SpO2 trend as linear regression slope.
        
        Args:
            spo2: SpO2 array.
        
        Returns:
            Slope of linear regression (SpO2 units per sample).
        """
        if len(spo2) < 2:
            return 0.0
        
        def _slope(x: np.ndarray) -> float:
            t = np.arange(len(x))
            slope, _, _, _, _ = scipy_stats.linregress(t, x)
            return float(slope)
        
        return self._safe_extract(spo2, _slope)
    
    def _compute_odi(self, spo2: np.ndarray) -> float:
        """Compute Oxygen Desaturation Index (events per hour, approximated).
        
        ODI is typically events/hour. Within a window, we approximate by
        scaling the drop count to an hourly rate.
        
        Args:
            spo2: SpO2 array.
        
        Returns:
            Approximate ODI value.
        """
        if len(spo2) < 5:
            return 0.0
        
        # Get sampling rate from metadata if available, else default
        # Assuming ~1 Hz for SpO2
        samples_per_hour = 3600.0  # Approximate samples per hour at 1 Hz
        
        drop_count = self._compute_drop_count(spo2)
        
        if len(spo2) > 0:
            # Scale to hourly rate
            return float(drop_count * samples_per_hour / len(spo2))
        return 0.0