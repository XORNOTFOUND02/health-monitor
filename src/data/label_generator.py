"""
Rule-based label generator module.

Provides ``LabelGenerator`` that produces ground-truth labels and
confidence scores for each health condition using evidence-based
clinical thresholds defined in :mod:`src.config`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..config import THRESHOLDS, ConditionThresholds

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generate labels and confidence scores for a single analysis window.

    All public ``label_*`` methods return a ``(detected, confidence)``
    tuple where ``detected`` is a boolean flag and ``confidence`` is a
    float in ``[0.0, 1.0]`` that reflects how strongly the signal
    supports the detection.

    Parameters
    ----------
    thresholds : ConditionThresholds, optional
        Override the global clinical thresholds (useful for testing).
    """

    def __init__(self, thresholds: Optional[ConditionThresholds] = None) -> None:
        self._th = thresholds or THRESHOLDS

    # ------------------------------------------------------------------
    # Individual condition labelers
    # ------------------------------------------------------------------

    def label_tachycardia(
        self,
        hr_values: np.ndarray,
        is_resting: bool = True,
    ) -> Tuple[bool, float]:
        """Detect tachycardia (elevated resting heart rate).

        Parameters
        ----------
        hr_values : np.ndarray
            Heart-rate samples (BPM) within the window.
        is_resting : bool
            Whether the subject is at rest.  When ``False`` a higher
            threshold is applied (exercise-induced elevation is normal).

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
        """
        if hr_values.size == 0:
            return False, 0.0

        mean_hr = float(np.mean(hr_values))
        threshold = float(self._th.TACHYCARDIA_BPM)

        if not is_resting:
            # During activity, require a higher bar
            threshold += 20

        if mean_hr < threshold:
            return False, 0.0

        # Confidence scales linearly from 0 at threshold to 1 at 150 BPM
        upper = max(threshold + 50, 150)
        confidence = float(np.clip((mean_hr - threshold) / (upper - threshold), 0.0, 1.0))
        return True, confidence

    def label_irregular_rhythm(
        self,
        rr_intervals: np.ndarray,
    ) -> Tuple[bool, float]:
        """Detect irregular heart rhythm using RR-interval variability.

        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in **seconds** (inter-beat intervals).

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
        """
        if rr_intervals.size < 3:
            return False, 0.0

        mean_rr = float(np.mean(rr_intervals))
        if mean_rr < 1e-6:
            return False, 0.0

        std_rr = float(np.std(rr_intervals))
        cv = std_rr / mean_rr  # coefficient of variation

        threshold = float(self._th.IRREGULAR_RR_CV_THRESHOLD)

        if cv < threshold:
            return False, 0.0

        # Confidence grows with CV; maxes out at CV = 0.5
        confidence = float(np.clip((cv - threshold) / (0.5 - threshold), 0.0, 1.0))
        return True, confidence

    def label_low_spo2(
        self,
        spo2_values: np.ndarray,
    ) -> Tuple[bool, float]:
        """Detect low blood-oxygen saturation (hypoxemia).

        Parameters
        ----------
        spo2_values : np.ndarray
            SpO₂ readings (percent, 0–100).

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
        """
        if spo2_values.size == 0:
            return False, 0.0

        mean_spo2 = float(np.mean(spo2_values))
        low_thresh = float(self._th.LOW_SPO2_THRESHOLD)
        severe_thresh = float(self._th.SEVERE_LOW_SPO2)

        if mean_spo2 >= low_thresh:
            return False, 0.0

        # Confidence ramps from 0 at 95 % to 1 at 85 %
        confidence = float(np.clip((low_thresh - mean_spo2) / (low_thresh - 85.0), 0.0, 1.0))
        return True, confidence

    def label_fever(
        self,
        temp_values: np.ndarray,
    ) -> Tuple[bool, float]:
        """Detect fever based on body-temperature readings.

        Parameters
        ----------
        temp_values : np.ndarray
            Temperature values in degrees Celsius.

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
        """
        if temp_values.size == 0:
            return False, 0.0

        mean_temp = float(np.mean(temp_values))
        fever_thresh = float(self._th.FEVER_TEMP_C)

        if mean_temp < fever_thresh:
            return False, 0.0

        # Confidence: 0 at 38 °C, 1 at 40 °C
        confidence = float(np.clip((mean_temp - fever_thresh) / 2.0, 0.0, 1.0))
        return True, confidence

    def label_fall(
        self,
        accel_mag: np.ndarray,
        window_duration: float,
    ) -> Tuple[bool, float]:
        """Detect a fall event from accelerometer magnitude.

        A fall is characterised by a high-acceleration spike followed by
        a period of near-stillness.

        Parameters
        ----------
        accel_mag : np.ndarray
            1-D magnitude of accelerometer signal (g-units).
        window_duration : float
            Duration of the window in seconds.

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
        """
        if accel_mag.size == 0:
            return False, 0.0

        fall_thresh = float(self._th.FALL_ACCEL_THRESHOLD_G)
        still_thresh = float(self._th.FALL_STILLNESS_THRESHOLD)
        still_dur = float(self._th.FALL_STILLNESS_DURATION_SEC)

        # High-impact phase: any sample exceeding the g-threshold
        has_impact = bool(np.any(accel_mag > fall_thresh))
        if not has_impact:
            return False, 0.0

        # Post-impact stillness: trailing portion of window
        ref_rate = accel_mag.size / window_duration if window_duration > 0 else 50.0
        still_samples = int(still_dur * ref_rate)
        trailing = accel_mag[-still_samples:] if still_samples > 0 else accel_mag[-1:]
        std_trailing = float(np.std(trailing))

        if std_trailing > still_thresh:
            # Movement after impact → not a fall (e.g., jump)
            return False, 0.0

        # Confidence based on impact magnitude
        max_impact = float(np.max(accel_mag))
        confidence = float(np.clip((max_impact - fall_thresh) / (10.0 - fall_thresh), 0.0, 1.0))
        return True, confidence

    def label_sleep_problem(
        self,
        motion: np.ndarray,
        hr: np.ndarray,
        spo2: np.ndarray,
        is_sleep_period: bool = False,
    ) -> Tuple[bool, float]:
        """Detect sleep-related problems (restlessness, apnoea signs).

        Parameters
        ----------
        motion : np.ndarray
            1-D accelerometer magnitude (g-units).
        hr : np.ndarray
            Heart-rate samples (BPM).
        spo2 : np.ndarray
            SpO₂ samples (%).
        is_sleep_period : bool
            Whether the window falls within expected sleep hours.

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
        """
        if motion.size == 0:
            return False, 0.0

        motion_thresh = float(self._th.SLEEP_MOTION_THRESHOLD)
        apnea_drop = float(self._th.APNEA_SPO2_DROP_THRESHOLD)

        signals: list[float] = []
        reasons = 0

        # 1. Excessive motion during sleep
        motion_std = float(np.std(motion))
        if is_sleep_period and motion_std > motion_thresh:
            reasons += 1
            signals.append(min(motion_std / (2.0 * motion_thresh), 1.0))

        # 2. SpO₂ dips (possible apnoea)
        if spo2.size > 1:
            baseline = float(np.percentile(spo2, 90))
            min_spo2 = float(np.min(spo2))
            drop = baseline - min_spo2
            if drop > apnea_drop:
                reasons += 1
                signals.append(min(drop / (2.0 * apnea_drop), 1.0))

        # 3. Elevated HR during sleep
        if is_sleep_period and hr.size > 0:
            mean_hr = float(np.mean(hr))
            # Rough baseline: if HR > 85 during sleep it may be abnormal
            if mean_hr > 85:
                reasons += 1
                signals.append(min((mean_hr - 85) / 30.0, 1.0))

        if reasons == 0:
            return False, 0.0

        confidence = float(np.clip(np.mean(signals), 0.0, 1.0))
        return True, confidence

    def label_fatigue(
        self,
        hrv_rmssd: float,
        resting_hr: float,
        activity_level: float,
        sleep_quality: float,
    ) -> Tuple[bool, float]:
        """Detect fatigue from HRV, resting HR, activity, and sleep quality.

        Parameters
        ----------
        hrv_rmssd : float
            Root-mean-square of successive RR differences (ms).
        resting_hr : float
            Resting heart rate (BPM).
        activity_level : float
            Normalised activity level ``[0, 1]`` (1 = very active).
        sleep_quality : float
            Normalised sleep quality ``[0, 1]`` (1 = excellent).

        Returns
        -------
        tuple of (bool, float)
            ``(detected, confidence)``.
        """
        hrv_thresh = float(self._th.FATIGUE_HRV_RMSSD_THRESHOLD)
        hr_elev = float(self._th.FATIGUE_RESTING_HR_ELEVATION)
        normal_resting = 70.0  # typical resting HR baseline

        scores: list[float] = []
        reasons = 0

        # Low HRV (sympathetic dominance → stress / fatigue)
        if hrv_rmssd < hrv_thresh:
            reasons += 1
            scores.append(min((hrv_thresh - hrv_rmssd) / hrv_thresh, 1.0))

        # Elevated resting HR
        if resting_hr > normal_resting + hr_elev:
            reasons += 1
            scores.append(
                min((resting_hr - (normal_resting + hr_elev)) / 30.0, 1.0)
            )

        # Low activity + poor sleep
        if activity_level < 0.3 and sleep_quality < 0.4:
            reasons += 1
            combined = (1.0 - activity_level) + (1.0 - sleep_quality)
            scores.append(min(combined / 2.0, 1.0))

        # Very poor sleep quality alone
        if sleep_quality < 0.2:
            reasons += 1
            scores.append(1.0 - sleep_quality)

        if reasons == 0:
            return False, 0.0

        confidence = float(np.clip(np.mean(scores), 0.0, 1.0))
        return True, confidence

    # ------------------------------------------------------------------
    # Aggregate label generator
    # ------------------------------------------------------------------

    def generate_labels(self, window_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run every condition labeler on *window_data*.

        Parameters
        ----------
        window_data : dict
            A single window dict as produced by
            :class:`WindowGenerator`.  Expected keys include
            ``"sensor_data"`` (with sub-keys for each channel) and
            optionally ``"metadata"``.

        Returns
        -------
        dict
            ``{condition_name: {"detected": bool, "confidence": float}}``
        """
        sensor = window_data.get("sensor_data", {})
        meta = window_data.get("metadata", {})
        duration = window_data.get("end_time", 30.0) - window_data.get(
            "start_time", 0.0
        )
        if duration <= 0:
            duration = 30.0

        # Extract available channels
        accel = sensor.get("accelerometer", np.array([]))
        gyro = sensor.get("gyroscope", np.array([]))
        hr = sensor.get("heart_rate", np.array([]))
        spo2 = sensor.get("spo2", np.array([]))
        temp = sensor.get("temperature", np.array([]))
        ppg = sensor.get("ppg", np.array([]))

        # Derived signals
        if accel.ndim == 2 and accel.shape[1] >= 3:
            accel_mag = np.sqrt(np.sum(accel[:, :3] ** 2, axis=1))
        elif accel.ndim == 1:
            accel_mag = accel
        else:
            accel_mag = np.array([])

        # Compute RR intervals from heart rate if available
        rr_intervals = np.array([])
        if hr.size > 1:
            valid_hr = hr[hr > 0]
            if valid_hr.size > 0:
                rr_intervals = 60.0 / valid_hr  # seconds

        # Is this a sleep period?
        is_sleep = bool(meta.get("is_sleep_period", False))

        # Activity level proxy from accel magnitude std
        activity_level = float(np.clip(
            float(np.std(accel_mag)) / 2.0 if accel_mag.size > 0 else 0.0,
            0.0,
            1.0,
        ))

        # Sleep quality proxy (inverse of motion std)
        sleep_quality = float(np.clip(
            1.0 - float(np.std(accel_mag)) / 2.0 if accel_mag.size > 0 else 0.5,
            0.0,
            1.0,
        ))

        # HRV RMSSD proxy from RR intervals
        hrv_rmssd = 0.0
        if rr_intervals.size > 1:
            diffs = np.diff(rr_intervals)
            hrv_rmssd = float(np.sqrt(np.mean(diffs ** 2))) * 1000.0  # ms

        resting_hr = float(np.mean(hr)) if hr.size > 0 else 70.0

        # ---- Run labelers ----
        labels: Dict[str, Dict[str, Any]] = {}

        detected, conf = self.label_tachycardia(hr, is_resting=True)
        labels["tachycardia"] = {"detected": detected, "confidence": conf}

        detected, conf = self.label_irregular_rhythm(rr_intervals)
        labels["irregular_rhythm"] = {"detected": detected, "confidence": conf}

        detected, conf = self.label_low_spo2(spo2)
        labels["low_spo2"] = {"detected": detected, "confidence": conf}

        detected, conf = self.label_fever(temp)
        labels["fever"] = {"detected": detected, "confidence": conf}

        detected, conf = self.label_fall(accel_mag, duration)
        labels["fall_detected"] = {"detected": detected, "confidence": conf}

        detected, conf = self.label_sleep_problem(
            accel_mag, hr, spo2, is_sleep_period=is_sleep
        )
        labels["sleep_problem"] = {"detected": detected, "confidence": conf}

        detected, conf = self.label_fatigue(
            hrv_rmssd, resting_hr, activity_level, sleep_quality
        )
        labels["fatigue"] = {"detected": detected, "confidence": conf}

        return labels
