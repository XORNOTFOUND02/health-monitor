"""
Inference pipeline for health symptom detection.

Provides ensemble prediction from trained LightGBM models and rule-based
detectors on sliding windows of wearable sensor data.

Modules:
    predictor       - Loads models, extracts features, runs inference.
    temporal_smoother - N-of-M voting to suppress single-window false positives.
    response_builder - Standardized JSON response formatting.
"""

from .predictor import Predictor
from .temporal_smoother import TemporalSmoother
from .response_builder import ResponseBuilder

__all__ = [
    "Predictor",
    "TemporalSmoother",
    "ResponseBuilder",
]
