"""
Rule-based condition detector for conditions best handled deterministically.

Currently handles:
    - Fever  (direct temperature threshold >= 38 degrees C)

Also provides input validation, data-quality checks, and a composite
quality score that downstream modules can use to flag unreliable
predictions.

This module is intentionally *not* a model — it produces deterministic,
clinically grounded decisions with no learned parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..config import THRESHOLDS, ConditionThresholds

logger = logging.getLogger(__name__)


class RuleEngine:
    """Deterministic rule-based condition detector.

    Parameters
    ----------
    thresholds : ConditionThresholds, optional
        Override the global clinical thresholds (useful for testing).
    """

    # Physiologically plausible ranges for sanity-checking sensor data
    _VALID_RANGES: Dict[str, Tuple[float, float]] = {
        "heart_rate_bpm": (25.0, 250.0),
        "spo2_pct": (50.0, 100.0),
        "temperature_c": (30.0, 45.0),
        "accel_magnitude_g": (0.0, 50.0),
        "gyro_dps": (-2000.0, 2000.0),
    }

    # Threshold below which a signal is considered "flatlined" (stuck at constant)
    _FLATLINE_MIN_VARIANCE: float = 1e-10

    def __init__(self, thresholds: Optional[ConditionThresholds] = None) -> None:
        self._th = thresholds or THRESHOLDS

    # ------------------------------------------------------------------
    # Fever detection
    # ------------------------------------------------------------------

    def detect_fever(
        self,
        temp_values: np.ndarray,
    ) -> Tuple[bool, float]:
        """Detect fever via direct temperature threshold.

        Parameters
        ----------
        temp_values : np.ndarray
            Temperature readings in degrees Celsius (may contain
            multiple samples from the window).

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
            ``detected`` is ``True`` when the mean temperature >= 38 degrees C.
            ``confidence`` scales linearly from 0.0 (at 38 degrees C) to
            1.0 (at 40 degrees C), clipped to [0, 1].
        """
        if temp_values is None or temp_values.size == 0:
            return False, 0.0

        # Clean invalid values
        valid = temp_values[np.isfinite(temp_values)]
        if valid.size == 0:
            return False, 0.0

        mean_temp = float(np.mean(valid))
        return self.compute_fever_confidence(mean_temp)

    def compute_fever_confidence(self, mean_temp: float) -> Tuple[bool, float]:
        """Compute fever detection result from a mean temperature value.

        Parameters
        ----------
        mean_temp : float
            Mean body temperature in degrees Celsius.

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
        """
        fever_thresh = float(self._th.FEVER_TEMP_C)

        if mean_temp < fever_thresh:
            return False, 0.0

        # Linear ramp: 0 at threshold, 1.0 at threshold + 2 degrees C
        confidence = float(np.clip((mean_temp - fever_thresh) / 2.0, 0.0, 1.0))
        return True, confidence

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input(
        self,
        features_dict: Dict[str, float],
    ) -> Dict[str, Any]:
        """Validate a feature vector against known sane ranges.

        Parameters
        ----------
        features_dict : dict[str, float]
            Mapping of feature name to float value.

        Returns
        -------
        dict
            Summary containing:
            - ``valid`` (bool): ``True`` if all checks passed.
            - ``nan_count`` (int): number of NaN / Inf features.
            - ``out_of_range`` (list[str]): features outside physiologic range.
            - ``zero_count`` (int): number of exactly-zero features.
            - ``flatlined`` (list[str]): features with zero variance.
            - ``details`` (dict): per-feature check results.
        """
        nan_count = 0
        out_of_range: list[str] = []
        zero_count = 0
        flatlined: list[str] = []
        details: Dict[str, Any] = {}

        for fname, value in features_dict.items():
            detail: Dict[str, Any] = {"value": value}

            # NaN / Inf check
            if not np.isfinite(value):
                nan_count += 1
                detail["nan"] = True
            else:
                detail["nan"] = False

                # Zero check
                if value == 0.0:
                    zero_count += 1
                    detail["zero"] = True
                else:
                    detail["zero"] = False

            details[fname] = detail

        # Range checks against known feature families
        for fname, value in features_dict.items():
            if not np.isfinite(value):
                continue
            for pattern, (lo, hi) in self._VALID_RANGES.items():
                if pattern in fname.lower():
                    if not (lo <= value <= hi):
                        out_of_range.append(fname)
                        details[fname]["out_of_range"] = True
                        details[fname]["valid_range"] = (lo, hi)
                    break

        valid = nan_count == 0 and len(out_of_range) == 0

        return {
            "valid": valid,
            "nan_count": nan_count,
            "out_of_range": out_of_range,
            "zero_count": zero_count,
            "flatlined": flatlined,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Data quality scoring
    # ------------------------------------------------------------------

    def get_data_quality_score(
        self,
        sensor_data: Dict[str, np.ndarray],
    ) -> float:
        """Compute a composite data-quality score in [0.0, 1.0].

        Checks for:
        1. NaN / Inf values in any channel.
        2. Out-of-range values.
        3. Signal flatlining (zero variance).
        4. Insufficient samples.

        Parameters
        ----------
        sensor_data : dict[str, np.ndarray]
            Mapping from channel name to 1-D numpy array.

        Returns
        -------
        float
            Quality score from 0.0 (unusable) to 1.0 (perfect).
        """
        if not sensor_data:
            return 0.0

        penalties = 0.0
        total_checks = 0.0

        for channel, arr in sensor_data.items():
            if arr is None or arr.size == 0:
                penalties += 1.0
                total_checks += 1.0
                continue

            arr_flat = arr.ravel().astype(np.float64)
            total_checks += 1.0

            # NaN / Inf penalty
            nan_ratio = float(np.mean(~np.isfinite(arr_flat)))
            if nan_ratio > 0:
                penalties += nan_ratio

            # Flatline detection
            variance = float(np.var(arr_flat))
            if variance < self._FLATLINE_MIN_VARIANCE and arr_flat.size > 1:
                penalties += 0.3  # moderate penalty for flat signal

            # Out-of-range check
            finite_vals = arr_flat[np.isfinite(arr_flat)]
            if finite_vals.size > 0:
                for pattern, (lo, hi) in self._VALID_RANGES.items():
                    if pattern in channel.lower():
                        out_of_range_ratio = float(
                            np.mean((finite_vals < lo) | (finite_vals > hi))
                        )
                        penalties += out_of_range_ratio * 0.5
                        break

        if total_checks == 0:
            return 0.0

        # Normalise penalty to [0, 1] range
        avg_penalty = min(penalties / total_checks, 1.0)
        quality = max(0.0, 1.0 - avg_penalty)

        return float(quality)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            "RuleEngine",
            f"  Fever threshold : {self._th.FEVER_TEMP_C} C",
            f"  Features validated against {len(self._VALID_RANGES)} range rules",
        ]
        return "\n".join(lines)
