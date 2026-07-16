"""
Base class for all feature extractors.

Provides common functionality for extracting features from sliding windows
of wearable sensor data, including safe extraction utilities that handle
edge cases like empty arrays, NaN values, and computation errors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors.
    
    All feature extractors in this package inherit from this class and
    implement the `extract()` method. This base class provides:
    - Feature name tracking
    - Safe extraction helpers that handle edge cases
    - Consistent interface for the orchestrator
    
    Attributes:
        feature_names: List of feature names this extractor produces.
    """
    
    def __init__(self) -> None:
        """Initialize the feature extractor."""
        self.feature_names: List[str] = []
    
    @abstractmethod
    def extract(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from a window of sensor data.
        
        Args:
            window_data: Dict containing sensor arrays for this window.
                Expected structure:
                {
                    'accelerometer': {'ax': np.ndarray, 'ay': np.ndarray, 'az': np.ndarray},
                    'gyroscope': {'gx': np.ndarray, 'gy': np.ndarray, 'gz': np.ndarray},
                    'heart_rate': {'bpm': np.ndarray, 'spo2': np.ndarray, 'ppg_raw': np.ndarray},
                    'temperature': {'stts22h_celsius': np.ndarray, 'lm35_celsius': np.ndarray},
                    'metadata': {'activity_state': str, 'sampling_rate': float, ...}
                }
        
        Returns:
            Dict[str, float]: Feature name -> value mapping.
        """
        pass
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces.
        
        Returns:
            List of feature name strings.
        """
        return self.feature_names
    
    def _safe_extract(
        self,
        arr: Optional[np.ndarray],
        func: callable,
        default: float = 0.0,
    ) -> float:
        """Safely extract a feature, returning default if array is empty/invalid.
        
        Handles common edge cases:
        - None input
        - Empty array
        - All-NaN array
        - Function execution errors
        
        Args:
            arr: Input numpy array (or None).
            func: Function to apply to the array.
            default: Value to return if extraction fails.
        
        Returns:
            Result of func(arr) or default if extraction fails.
        """
        if arr is None or len(arr) == 0:
            return default
        try:
            result = func(arr)
            if result is None or np.isnan(result) or np.isinf(result):
                return default
            return float(result)
        except Exception:
            return default
    
    def _safe_array(
        self,
        arr: Optional[np.ndarray],
        fill_value: float = 0.0,
        dtype: type = np.float64,
    ) -> np.ndarray:
        """Convert input to a safe numpy array, handling None and NaN.
        
        Args:
            arr: Input array (or None).
            fill_value: Value to replace NaN/None with.
            dtype: Target numpy dtype.
        
        Returns:
            Safe numpy array with no NaN values.
        """
        if arr is None:
            return np.array([], dtype=dtype)
        result = np.asarray(arr, dtype=dtype)
        if result.size == 0:
            return result
        nan_mask = np.isnan(result) | np.isinf(result)
        if np.any(nan_mask):
            result = result.copy()
            result[nan_mask] = fill_value
        return result
    
    def _get_nested(
        self,
        data: Dict[str, Any],
        *keys: str,
        default: Any = None,
    ) -> Any:
        """Safely get a nested value from a dictionary.
        
        Args:
            data: Source dictionary.
            *keys: Sequence of keys to traverse.
            default: Default value if any key is missing.
        
        Returns:
            The nested value or default.
        """
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def _get_sensor_array(
        self,
        window_data: Dict[str, Any],
        *keys: str,
        default: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get a sensor array from window data with safe conversion.
        
        Args:
            window_data: Full window data dictionary.
            *keys: Keys to traverse (e.g., 'accelerometer', 'ax').
            default: Default array if not found.
        
        Returns:
            Safe numpy array.
        """
        arr = self._get_nested(window_data, *keys, default=default)
        return self._safe_array(arr)
    
    def _compute_percentile(
        self,
        arr: np.ndarray,
        q: float,
        default: float = 0.0,
    ) -> float:
        """Compute percentile with safe handling."""
        return self._safe_extract(arr, lambda x: np.percentile(x, q), default)
    
    def _compute_entropy(self, arr: np.ndarray, bins: int = 50, default: float = 0.0) -> float:
        """Compute Shannon entropy of a signal's amplitude distribution.
        
        Args:
            arr: Input signal.
            bins: Number of histogram bins.
            default: Default value on failure.
        
        Returns:
            Shannon entropy value.
        """
        def _entropy(x: np.ndarray) -> float:
            if len(x) < 2:
                return 0.0
            hist, _ = np.histogram(x, bins=bins, density=True)
            hist = hist[hist > 0]
            if len(hist) == 0:
                return 0.0
            bin_width = (np.max(x) - np.min(x)) / bins if np.max(x) != np.min(x) else 1.0
            if bin_width <= 0:
                bin_width = 1.0
            prob = hist * bin_width
            prob = prob[prob > 0]
            return float(-np.sum(prob * np.log(prob)))
        return self._safe_extract(arr, _entropy, default)