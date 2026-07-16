"""
Model definitions and rule-based detection for health symptom classification.

This package provides:

- ``CardiacModel``  – Multi-output LightGBM for tachycardia and
  irregular_rhythm detection.
- ``RespiratoryModel`` – Binary LightGBM for low SpO2 detection.
- ``ActivityModel`` – Multi-output LightGBM for fall detection, sleep
  problems, and fatigue.
- ``RuleEngine`` – Deterministic rule-based fever detection and input
  validation.
"""

from .cardiac_model import CardiacModel
from .respiratory_model import RespiratoryModel
from .activity_model import ActivityModel
from .rule_engine import RuleEngine

__all__ = [
    "CardiacModel",
    "RespiratoryModel",
    "ActivityModel",
    "RuleEngine",
]
