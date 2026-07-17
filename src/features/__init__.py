"""
Feature extraction modules for health symptom detection from wearable sensor data.

This package provides feature extractors for:
- Motion (accelerometer/gyroscope)
- Heart rate
- Heart rate variability (HRV)
- Blood oxygen saturation (SpO2)
- Body temperature
- Cross-sensor features
- Frequency-domain features

All extractors inherit from BaseFeatureExtractor and produce dictionaries
of named features from sliding windows of time-series data.
"""

from .base import BaseFeatureExtractor
from .motion import MotionFeatures
from .heart_rate import HeartRateFeatures
from .hrv import HRVFeatures
from .spo2 import SpO2Features
from .temperature import TemperatureFeatures
from .cross_sensor import CrossSensorFeatures
from .frequency_domain import FrequencyDomainFeatures
from .mag_features import extract_magnetometer_features, list_mag_feature_names
from .extractor import FeatureExtractor

__all__ = [
    "BaseFeatureExtractor",
    "MotionFeatures",
    "HeartRateFeatures",
    "HRVFeatures",
    "SpO2Features",
    "TemperatureFeatures",
    "CrossSensorFeatures",
    "FrequencyDomainFeatures",
    "extract_magnetometer_features",
    "list_mag_feature_names",
    "FeatureExtractor",
]