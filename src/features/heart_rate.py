"""
Heart rate feature extraction from BPM time series.

Extracts 15 features from heart rate data including:
- Basic statistics (mean, std, min, max, range)
- Resting HR (10th percentile), Peak HR (90th percentile)
- HR recovery indicator
- Tachycardia/bradycardia indicators
- HR trend (linear regression slope)
- HR acceleration
- Cubic mean (power mean)
- HR entropy (sample entropy approximation)
"""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats as scipy_stats

from .base import BaseFeatureExtractor


class HeartRateFeatures(BaseFeatureExtractor):
    """Extract heart rate features from BPM time series.
    
    This extractor computes statistical, temporal, and clinical features
    from heart rate measurements within each analysis window.
    
    Features (15 total):
        hr_mean, hr_std, hr_min, hr_max, hr_range
        hr_resting (10th percentile), hr_peak (90th percentile)
        hr_recovery_indicator (peak to end-of-window difference)
        hr_tachycardia_pct (% time > 100 BPM)
        hr_bradycardia_pct (% time < 60 BPM)
        hr_trend_slope (linear regression slope)
        hr_acceleration (mean of first derivative)
        hr_cubic_mean (power mean with p=3)
        hr_entropy (sample entropy approximation)
    
    Note:
        Clinical thresholds:
        - Tachycardia: HR > 100 BPM
        - Bradycardia: HR < 60 BPM
        - Normal resting HR: 60-100 BPM
    """
    
    TACHYCARDIA_THRESHOLD: float = 100.0
    BRADYCARDIA_THRESHOLD: float = 60.0
    
    def __init__(self) -> None:
        """Initialize heart rate feature extractor."""
        super().__init__()
        self.feature_names = [
            "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
            "hr_resting", "hr_peak", "hr_recovery_indicator",
            "hr_tachycardia_pct", "hr_bradycardia_pct",
            "hr_trend_slope", "hr_acceleration",
            "hr_cubic_mean", "hr_entropy",
        ]
    
    def extract(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract heart rate features from BPM time series.
        
        Args:
            window_data: Dictionary containing heart rate data with key
                'heart_rate' -> 'bpm' array.
        
        Returns:
            Dictionary of heart rate feature names to float values.
        """
        features: Dict[str, float] = {}
        
        # Get HR BPM data
        hr = self._get_sensor_array(window_data, "heart_rate", "bpm")
        
        if len(hr) == 0:
            # Return default values for all features
            for name in self.feature_names:
                features[name] = 0.0
            return features
        
        # Basic statistics
        features["hr_mean"] = self._safe_extract(hr, lambda x: np.mean(x))
        features["hr_std"] = self._safe_extract(
            hr, lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0
        )
        features["hr_min"] = self._safe_extract(hr, lambda x: np.min(x))
        features["hr_max"] = self._safe_extract(hr, lambda x: np.max(x))
        features["hr_range"] = self._safe_extract(hr, lambda x: np.ptp(x))
        
        # Resting and peak HR (percentiles)
        features["hr_resting"] = self._compute_percentile(hr, 10)
        features["hr_peak"] = self._compute_percentile(hr, 90)
        
        # HR recovery indicator: peak minus last value in window
        features["hr_recovery_indicator"] = self._compute_recovery(hr)
        
        # Clinical indicators
        features["hr_tachycardia_pct"] = self._compute_pct_above(
            hr, self.TACHYCARDIA_THRESHOLD
        )
        features["hr_bradycardia_pct"] = self._compute_pct_below(
            hr, self.BRADYCARDIA_THRESHOLD
        )
        
        # HR trend: slope of linear regression
        features["hr_trend_slope"] = self._compute_trend(hr)
        
        # HR acceleration: mean of first derivative
        features["hr_acceleration"] = self._compute_acceleration(hr)
        
        # Cubic mean (power mean with p=3)
        features["hr_cubic_mean"] = self._compute_cubic_mean(hr)
        
        # HR entropy (sample entropy approximation)
        features["hr_entropy"] = self._compute_sample_entropy(hr)
        
        return features
    
    def _compute_recovery(self, hr: np.ndarray) -> float:
        """Compute HR recovery indicator (peak to end-of-window difference).
        
        A positive value indicates HR decreased from peak during window.
        
        Args:
            hr: Heart rate BPM array.
        
        Returns:
            Difference between peak HR and last HR value.
        """
        if len(hr) < 2:
            return 0.0
        
        def _recovery(x: np.ndarray) -> float:
            peak_val = np.percentile(x, 90)
            end_val = x[-1]
            return float(peak_val - end_val)
        
        return self._safe_extract(hr, _recovery)
    
    def _compute_pct_above(self, hr: np.ndarray, threshold: float) -> float:
        """Compute percentage of time HR is above a threshold.
        
        Args:
            hr: Heart rate BPM array.
            threshold: HR threshold value.
        
        Returns:
            Fraction of samples above threshold (0.0 to 1.0).
        """
        def _pct(x: np.ndarray) -> float:
            return float(np.mean(x > threshold))
        return self._safe_extract(hr, _pct)
    
    def _compute_pct_below(self, hr: np.ndarray, threshold: float) -> float:
        """Compute percentage of time HR is below a threshold.
        
        Args:
            hr: Heart rate BPM array.
            threshold: HR threshold value.
        
        Returns:
            Fraction of samples below threshold (0.0 to 1.0).
        """
        def _pct(x: np.ndarray) -> float:
            return float(np.mean(x < threshold))
        return self._safe_extract(hr, _pct)
    
    def _compute_trend(self, hr: np.ndarray) -> float:
        """Compute HR trend as linear regression slope.
        
        Args:
            hr: Heart rate BPM array.
        
        Returns:
            Slope of linear regression (BPM per sample).
        """
        if len(hr) < 2:
            return 0.0
        
        def _slope(x: np.ndarray) -> float:
            t = np.arange(len(x))
            slope, _, _, _, _ = scipy_stats.linregress(t, x)
            return float(slope)
        
        return self._safe_extract(hr, _slope)
    
    def _compute_acceleration(self, hr: np.ndarray) -> float:
        """Compute HR acceleration (mean of first derivative).
        
        Args:
            hr: Heart rate BPM array.
        
        Returns:
            Mean rate of HR change.
        """
        if len(hr) < 2:
            return 0.0
        
        def _accel(x: np.ndarray) -> float:
            return float(np.mean(np.diff(x)))
        
        return self._safe_extract(hr, _accel)
    
    def _compute_cubic_mean(self, hr: np.ndarray) -> float:
        """Compute cubic mean (power mean with p=3).
        
        The cubic mean emphasizes larger values, useful for detecting
        elevated HR episodes.
        
        Args:
            hr: Heart rate BPM array.
        
        Returns:
            Cubic mean value.
        """
        def _cubic(x: np.ndarray) -> float:
            return float(np.cbrt(np.mean(x ** 3)))
        
        return self._safe_extract(hr, _cubic)
    
    def _compute_sample_entropy(
        self, hr: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        """Approximate sample entropy of HR signal.
        
        Sample entropy measures signal complexity/irregularity.
        Higher values indicate more irregular HR patterns.
        
        Args:
            hr: Heart rate BPM array.
            m: Embedding dimension.
            r: Tolerance (fraction of std).
        
        Returns:
            Approximate sample entropy value.
        """
        if len(hr) < m + 2:
            return 0.0
        
        def _sampen(x: np.ndarray) -> float:
            n = len(x)
            if n < m + 2:
                return 0.0
            
            # Normalize
            std_x = np.std(x)
            if std_x == 0:
                return 0.0
            x_norm = (x - np.mean(x)) / std_x
            r_val = r * std_x  # Use r as fraction of std
            
            # Count matches for dimension m
            def _count_matches(data: np.ndarray, dim: int) -> int:
                count = 0
                n_d = len(data)
                for i in range(n_d - dim):
                    template = data[i:i + dim]
                    for j in range(i + 1, n_d - dim):
                        candidate = data[j:j + dim]
                        if np.max(np.abs(template - candidate)) < r_val:
                            count += 1
                return count
            
            # For efficiency with large arrays, subsample
            if n > 200:
                indices = np.linspace(0, n - 1, 200, dtype=int)
                x_norm = x_norm[indices]
                n = len(x_norm)
            
            A = _count_matches(x_norm, m)
            B = _count_matches(x_norm, m + 1)
            
            if A == 0 or B == 0:
                return 0.0
            
            return float(-np.log(B / A))
        
        return self._safe_extract(hr, _sampen)