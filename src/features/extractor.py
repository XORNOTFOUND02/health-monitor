"""
Feature extraction orchestrator.

Runs all feature extractors and combines results into a single feature
vector. This module serves as the main interface for extracting all
features from a window of sensor data.
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np

from .base import BaseFeatureExtractor
from .motion import MotionFeatures
from .heart_rate import HeartRateFeatures
from .hrv import HRVFeatures
from .spo2 import SpO2Features
from .temperature import TemperatureFeatures
from .cross_sensor import CrossSensorFeatures
from .frequency_domain import FrequencyDomainFeatures
from .mag_features import extract_magnetometer_features, list_mag_feature_names


# Module-level logger
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Orchestrates all feature extraction modules.
    
    This class provides a unified interface for extracting all health
    monitoring features from a window of sensor data. It runs each
    feature extractor independently and combines results.
    
    Features extracted:
        - MotionFeatures: 60+ motion features
        - HeartRateFeatures: 15 HR features
        - HRVFeatures: 18 HRV features
        - SpO2Features: 10 SpO2 features
        - TemperatureFeatures: 9 temperature features
        - CrossSensorFeatures: 15 cross-sensor features
        - FrequencyDomainFeatures: 27 frequency features
        - Magnetometer features: ~23 magnetometer features
        Total: ~185 features (some may be zeroed due to missing data)
    
    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract_all(window_data)
        # features is a dict of feature_name -> value
    
    Example window_data structure:
        {
            'accelerometer': {
                'ax': np.array([...]),
                'ay': np.array([...]),
                'az': np.array([...])
            },
            'gyroscope': {
                'gx': np.array([...]),
                'gy': np.array([...]),
                'gz': np.array([...])
            },
            'heart_rate': {
                'bpm': np.array([...]),
                'spo2': np.array([...]),
                'ppg_raw': np.array([...])
            },
            'temperature': {
                'stts22h_celsius': np.array([...]),
                'lm35_celsius': np.array([...])
            },
            'magnetometer': np.array([...]),  # Shape (n, 3) — [Mx, My, Mz] in microtesla
            'metadata': {
                'sampling_rate': 30.0,
                'hr_sampling_rate': 4.0,
                'ppg_sampling_rate': 100.0,
                'activity_state': 'walking'
            }
        }
    """
    
    def __init__(self, enable_logging: bool = True) -> None:
        """Initialize feature extractor with all sub-extractors.
        
        Args:
            enable_logging: If True, log warnings for failed extractors.
        """
        self.enable_logging = enable_logging
        
        # Initialize all feature extractors
        self.extractors: Dict[str, BaseFeatureExtractor] = {
            "motion": MotionFeatures(),
            "heart_rate": HeartRateFeatures(),
            "hrv": HRVFeatures(),
            "spo2": SpO2Features(),
            "temperature": TemperatureFeatures(),
            "cross_sensor": CrossSensorFeatures(),
            "frequency": FrequencyDomainFeatures(),
        }
        
        # Cache feature names
        self._feature_names: Optional[List[str]] = None
    
    def extract_all(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Run all feature extractors and combine results.
        
        Each extractor is run independently. If an extractor fails,
        its features are set to 0.0 and an error is logged (if logging enabled).
        
        Args:
            window_data: Dictionary containing all sensor data for the window.
        
        Returns:
            Combined dictionary of all feature names to float values.
        """
        all_features: Dict[str, float] = {}
        
        for name, extractor in self.extractors.items():
            try:
                features = extractor.extract(window_data)
                all_features.update(features)
            except Exception as e:
                if self.enable_logging:
                    logger.warning(
                        f"Extractor '{name}' failed: {e}. "
                        f"Features will be set to 0.0."
                    )
                # Set all features from this extractor to 0.0
                for feature_name in extractor.get_feature_names():
                    all_features[feature_name] = 0.0
        
        # --- Magnetometer features ---
        try:
            from ..config import SAMPLING_RATES
            mag_sample_rate = float(SAMPLING_RATES.MAG)
        except Exception:
            mag_sample_rate = 25.0
        
        mag_data = window_data.get("magnetometer", None)
        if mag_data is not None:
            mag_arr = np.asarray(mag_data, dtype=np.float64)
            if mag_arr.ndim == 1:
                mag_arr = mag_arr.reshape(-1, 1)
            mag_feats = extract_magnetometer_features(mag_arr, sample_rate=mag_sample_rate)
            all_features.update(mag_feats)
        
        return all_features
    
    def get_all_feature_names(self) -> List[str]:
        """Get all feature names from all extractors.
        
        Returns:
            Sorted list of all feature names.
        """
        if self._feature_names is None:
            names: List[str] = []
            for extractor in self.extractors.values():
                names.extend(extractor.get_feature_names())
            # Add magnetometer feature names
            names.extend(list_mag_feature_names())
            self._feature_names = sorted(names)
        return self._feature_names.copy()
    
    def feature_count(self) -> int:
        """Get total number of features.
        
        Returns:
            Total number of features across all extractors.
        """
        return len(self.get_all_feature_names())
    
    def get_extractor_names(self) -> List[str]:
        """Get names of all registered extractors.
        
        Returns:
            List of extractor names.
        """
        return list(self.extractors.keys())
    
    def get_extractor(self, name: str) -> Optional[BaseFeatureExtractor]:
        """Get a specific extractor by name.
        
        Args:
            name: Extractor name (e.g., 'motion', 'heart_rate').
        
        Returns:
            The extractor instance, or None if not found.
        """
        return self.extractors.get(name)
    
    def extract_single(
        self, window_data: Dict[str, Any], extractor_name: str
    ) -> Dict[str, float]:
        """Run a single feature extractor.
        
        Args:
            window_data: Dictionary containing sensor data.
            extractor_name: Name of the extractor to run.
        
        Returns:
            Dictionary of features from that extractor.
        
        Raises:
            KeyError: If extractor_name is not registered.
        """
        if extractor_name not in self.extractors:
            raise KeyError(
                f"Unknown extractor: {extractor_name}. "
                f"Available: {list(self.extractors.keys())}"
            )
        
        return self.extractors[extractor_name].extract(window_data)
    
    def feature_names_by_extractor(self) -> Dict[str, List[str]]:
        """Get feature names grouped by extractor.
        
        Returns:
            Dictionary mapping extractor names to their feature name lists.
        """
        return {
            name: extractor.get_feature_names()
            for name, extractor in self.extractors.items()
        }
    
    def describe(self) -> str:
        """Get a human-readable description of the feature extractor.
        
        Returns:
            Multi-line string describing all extractors and their feature counts.
        """
        lines = ["FeatureExtractor Summary:", "=" * 40]
        
        total = 0
        for name, extractor in self.extractors.items():
            count = len(extractor.get_feature_names())
            total += count
            lines.append(f"  {name:20s}: {count:3d} features")
        
        lines.append("-" * 40)
        lines.append(f"  {'Total':20s}: {total:3d} features")
        
        return "\n".join(lines)