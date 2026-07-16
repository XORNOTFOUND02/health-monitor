"""
Synthetic data generator for health-monitor wearable sensor data.

Produces realistic time-series data for accelerometer, gyroscope, heart rate,
SpO2, PPG waveform, and temperature channels.  Supports normal healthy
sessions and sessions with specific clinical conditions (tachycardia,
irregular rhythm, low SpO2, fever, fall detection, sleep problems, fatigue).

The generator is designed for speed (~1 000 sessions in < 5 min on a laptop)
while maintaining physiological plausibility suitable for model training.

Usage
-----
>>> gen = SyntheticDataGenerator(seed=42)
>>> session = gen.generate_normal_session(duration_sec=60)
>>> session = gen.generate_condition_session("tachycardia", duration_sec=60)
>>> gen.generate_dataset(num_sessions=100, output_dir="data/synthetic/raw")
"""

from __future__ import annotations

import json
import logging
import math
import time as _time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAVITY: float = 9.81  # m/s^2

# Canonical condition list (order matches model output vectors)
CONDITIONS: List[str] = [
    "tachycardia",
    "irregular_rhythm",
    "low_spo2",
    "fever",
    "fall_detected",
    "sleep_problem",
    "fatigue",
]

# Sampling rates (Hz) — match src/config.py SamplingRates
ACCEL_HZ: int = 50
GYRO_HZ: int = 50
HR_HZ: int = 25
PPG_HZ: int = 25
SPO2_HZ: int = 25
TEMP_HZ: int = 1


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SubjectProfile:
    """Physiological profile for a synthetic subject."""

    age: int = 35
    sex: str = "male"
    resting_hr: float = 68.0
    activity_state: str = "resting"  # resting | walking | running
    weight_kg: float = 70.0
    height_cm: float = 170.0

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]] = None) -> "SubjectProfile":
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SamplingConfig:
    """Sampling-rate configuration embedded in every session."""

    accel_sample_rate_hz: int = ACCEL_HZ
    gyro_sample_rate_hz: int = GYRO_HZ
    hr_sample_rate_hz: int = HR_HZ
    ppg_sample_rate_hz: int = PPG_HZ
    spo2_sample_rate_hz: int = SPO2_HZ
    temp_sample_rate_hz: int = TEMP_HZ

    def to_dict(self) -> Dict[str, int]:
        return {
            "accel_sample_rate_hz": self.accel_sample_rate_hz,
            "gyro_sample_rate_hz": self.gyro_sample_rate_hz,
            "hr_sample_rate_hz": self.hr_sample_rate_hz,
            "ppg_sample_rate_hz": self.ppg_sample_rate_hz,
            "spo2_sample_rate_hz": self.spo2_sample_rate_hz,
            "temp_sample_rate_hz": self.temp_sample_rate_hz,
        }


# ---------------------------------------------------------------------------
# SensorSimulator
# ---------------------------------------------------------------------------
class SensorSimulator:
    """Generate realistic sensor noise and baseline signals.

    Uses pre-computed statistical models of MEMS sensor noise (accelerometer
    bias instability, gyroscope white noise, temperature drift, etc.) to
    produce time-series that closely resemble real hardware output.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    # -- Accelerometer ---------------------------------------------------

    def generate_accelerometer(
        self,
        n_samples: int,
        activity: str = "resting",
        gravity_axis: int = 1,
    ) -> np.ndarray:
        """Generate 3-axis accelerometer data.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        activity : str
            ``"resting"`` | ``"walking"`` | ``"running"``.
        gravity_axis : int
            Axis index (0=x, 1=y, 2=z) that aligns with gravity when the
            wrist is in a natural resting position.  Default ``1`` (y-axis).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 3)`` — columns are [Ax, Ay, Az] in m/s^2.
        """
        accel = np.zeros((n_samples, 3), dtype=np.float64)

        # Gravity vector on the selected axis
        accel[:, gravity_axis] = GRAVITY

        # Sensor noise (bias instability + white noise)
        accel += self.rng.normal(0, 0.03, size=(n_samples, 3))

        # Activity-specific motion
        t = np.arange(n_samples, dtype=np.float64)

        if activity == "walking":
            freq = self.rng.uniform(1.5, 2.2)  # step frequency Hz
            # Vertical oscillation (y-axis)
            accel[:, gravity_axis] += 0.8 * np.sin(2 * np.pi * freq * t / ACCEL_HZ)
            # Fore-aft sway (x-axis)
            accel[:, (gravity_axis + 1) % 3] += 0.4 * np.sin(
                2 * np.pi * freq * t / ACCEL_HZ + 0.3
            )
            # Lateral sway (z-axis)
            accel[:, (gravity_axis + 2) % 3] += 0.2 * np.sin(
                2 * np.pi * freq * t / ACCEL_HZ + 1.2
            )
            accel += self.rng.normal(0, 0.15, size=(n_samples, 3))

        elif activity == "running":
            freq = self.rng.uniform(2.5, 3.5)
            accel[:, gravity_axis] += 1.5 * np.sin(2 * np.pi * freq * t / ACCEL_HZ)
            accel[:, (gravity_axis + 1) % 3] += 0.8 * np.sin(
                2 * np.pi * freq * t / ACCEL_HZ + 0.3
            )
            accel[:, (gravity_axis + 2) % 3] += 0.4 * np.sin(
                2 * np.pi * freq * t / ACCEL_HZ + 1.2
            )
            accel += self.rng.normal(0, 0.30, size=(n_samples, 3))

        return accel

    # -- Gyroscope -------------------------------------------------------

    def generate_gyroscope(
        self,
        n_samples: int,
        activity: str = "resting",
    ) -> np.ndarray:
        """Generate 3-axis gyroscope data (rad/s).

        Parameters
        ----------
        n_samples : int
            Number of samples.
        activity : str
            Activity state.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 3)`` — [Gx, Gy, Gz] in rad/s.
        """
        gyro = np.zeros((n_samples, 3), dtype=np.float64)

        if activity == "resting":
            gyro += self.rng.normal(0, 0.003, size=(n_samples, 3))
        elif activity == "walking":
            t = np.arange(n_samples, dtype=np.float64)
            freq = self.rng.uniform(1.5, 2.2)
            gyro[:, 0] = 0.05 * np.sin(2 * np.pi * freq * t / GYRO_HZ)
            gyro[:, 1] = 0.03 * np.sin(2 * np.pi * freq * t / GYRO_HZ + 0.5)
            gyro[:, 2] = 0.02 * np.sin(2 * np.pi * freq * t / GYRO_HZ + 1.0)
            gyro += self.rng.normal(0, 0.015, size=(n_samples, 3))
        elif activity == "running":
            t = np.arange(n_samples, dtype=np.float64)
            freq = self.rng.uniform(2.5, 3.5)
            gyro[:, 0] = 0.12 * np.sin(2 * np.pi * freq * t / GYRO_HZ)
            gyro[:, 1] = 0.08 * np.sin(2 * np.pi * freq * t / GYRO_HZ + 0.5)
            gyro[:, 2] = 0.05 * np.sin(2 * np.pi * freq * t / GYRO_HZ + 1.0)
            gyro += self.rng.normal(0, 0.03, size=(n_samples, 3))

        return gyro

    # -- SpO2 ------------------------------------------------------------

    def generate_spo2(
        self,
        n_samples: int,
        baseline: float = 97.5,
        condition_modifier: Optional[str] = None,
    ) -> np.ndarray:
        """Generate SpO2 readings (%).

        Parameters
        ----------
        n_samples : int
            Number of samples.
        baseline : float
            Normal resting SpO2 (96-99%).
        condition_modifier : str or None
            ``"low_spo2"`` to produce desaturation patterns.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` with SpO2 in %.
        """
        spo2 = np.full(n_samples, baseline, dtype=np.float64)

        # Natural variability
        spo2 += self.rng.normal(0, 0.4, size=n_samples)

        # Slow drift (respiratory modulation)
        t = np.arange(n_samples, dtype=np.float64)
        breath_freq = self.rng.uniform(0.12, 0.25)
        spo2 += 0.8 * np.sin(2 * np.pi * breath_freq * t / SPO2_HZ)

        if condition_modifier == "low_spo2":
            # Progressive desaturation
            drift = np.linspace(0, self.rng.uniform(5, 12), n_samples)
            spo2 -= drift
            # Intermittent deeper desaturations
            n_dips = self.rng.integers(2, 6)
            for _ in range(n_dips):
                start = self.rng.integers(0, max(1, n_samples - 50))
                dip_len = min(self.rng.integers(20, 60), n_samples - start)
                spo2[start : start + dip_len] -= self.rng.uniform(3, 8)

        return np.clip(spo2, 70.0, 100.0)

    # -- Temperature -----------------------------------------------------

    def generate_temperature(
        self,
        n_samples: int,
        baseline: float = 36.5,
        circadian_phase: float = 0.0,
    ) -> np.ndarray:
        """Generate temperature readings (deg C).

        Parameters
        ----------
        n_samples : int
            Number of samples (at TEMP_HZ).
        baseline : float
            Baseline temperature in deg C.
        circadian_phase : float
            Phase offset for circadian variation (radians).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` with temperature in deg C.
        """
        temp = np.full(n_samples, baseline, dtype=np.float64)

        # Circadian variation (±0.5 °C over 24 h)
        t_hours = np.arange(n_samples, dtype=np.float64) / TEMP_HZ / 3600.0
        temp += 0.5 * np.sin(2 * np.pi * t_hours / 24.0 + circadian_phase)

        # Sensor noise
        temp += self.rng.normal(0, 0.08, size=n_samples)

        return temp


# ---------------------------------------------------------------------------
# PPGSimulator
# ---------------------------------------------------------------------------
class PPGSimulator:
    """Generate synthetic photoplethysmogram (PPG) waveforms.

    Uses a multi-sinusoid model with harmonics plus a dicrotic-notch
    component to approximate realistic PPG morphology.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(
        self,
        n_samples: int,
        heart_rate_bpm: float,
        sample_rate: int = PPG_HZ,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generate a synthetic PPG waveform.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        heart_rate_bpm : float
            Instantaneous heart rate in BPM.
        sample_rate : int
            Sampling rate of the PPG channel.
        amplitude : float
            Peak-to-peak amplitude scaling factor.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` — synthetic PPG signal (arbitrary units).
        """
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        hr_freq = heart_rate_bpm / 60.0  # beats per second

        # Fundamental + harmonics (Windkessel-like model)
        ppg = np.zeros(n_samples, dtype=np.float64)
        # Fundamental (systolic peak)
        ppg += 1.0 * np.sin(2 * np.pi * hr_freq * t - 0.5)
        # 2nd harmonic (sharpens peak, creates dicrotic notch)
        ppg += 0.45 * np.sin(2 * np.pi * 2 * hr_freq * t - 0.8)
        # 3rd harmonic
        ppg += 0.20 * np.sin(2 * np.pi * 3 * hr_freq * t - 1.0)
        # 4th harmonic
        ppg += 0.08 * np.sin(2 * np.pi * 4 * hr_freq * t - 1.2)

        # Asymmetric systolic / diastolic shape (clip the trough)
        ppg = ppg - ppg.min()
        ppg = ppg / (ppg.max() + 1e-10)

        # Baseline wander (respiratory)
        breath_freq = self.rng.uniform(0.12, 0.25)
        ppg += 0.05 * np.sin(2 * np.pi * breath_freq * t)

        # Sensor noise
        ppg += self.rng.normal(0, 0.015, size=n_samples)

        # Scale
        ppg = ppg * amplitude

        return ppg


# ---------------------------------------------------------------------------
# HeartRhythmSimulator
# ---------------------------------------------------------------------------
class HeartRhythmSimulator:
    """Generate realistic heart-rate time-series and RR intervals.

    Supports normal sinus rhythm, tachycardia, bradycardia, and irregular
    rhythms using a pulse-coupled oscillator model with phase-resetting.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate_rr_intervals(
        self,
        n_beats: int,
        base_hr: float = 70.0,
        rhythm_type: str = "normal",
    ) -> np.ndarray:
        """Generate a sequence of RR intervals (ms).

        Parameters
        ----------
        n_beats : int
            Number of inter-beat intervals to produce.
        base_hr : float
            Target heart rate in BPM.
        rhythm_type : str
            ``"normal"`` | ``"tachycardia"`` | ``"bradycardia"`` |
            ``"irregular"``.

        Returns
        -------
        np.ndarray
            Shape ``(n_beats,)`` — RR intervals in milliseconds.
        """
        base_rr_ms = 60_000.0 / base_hr
        rr = np.full(n_beats, base_rr_ms, dtype=np.float64)

        if rhythm_type == "normal":
            # Physiological HRV (RMSSD ≈ 30-60 ms for healthy adults)
            rr += self.rng.normal(0, 8.0, size=n_beats)
            # Respiratory sinus arrhythmia (~0.2 Hz modulation)
            t_beats = np.cumsum(rr) / 1000.0
            rr += 12.0 * np.sin(2 * np.pi * 0.2 * t_beats)

        elif rhythm_type == "tachycardia":
            # Reduced HRV
            rr += self.rng.normal(0, 3.0, size=n_beats)
            t_beats = np.cumsum(rr) / 1000.0
            rr += 4.0 * np.sin(2 * np.pi * 0.25 * t_beats)

        elif rhythm_type == "bradycardia":
            rr += self.rng.normal(0, 10.0, size=n_beats)
            t_beats = np.cumsum(rr) / 1000.0
            rr += 15.0 * np.sin(2 * np.pi * 0.15 * t_beats)

        elif rhythm_type == "irregular":
            # Markov-model based: state machine for beat-to-beat variation
            rr = self._irregular_rhythm_intervals(n_beats, base_rr_ms)

        # Ensure physically plausible (200-2000 ms)
        rr = np.clip(rr, 200.0, 2000.0)
        return rr

    def _irregular_rhythm_intervals(
        self, n_beats: int, base_rr_ms: float
    ) -> np.ndarray:
        """Generate irregular rhythm RR intervals with premature beats.

        Uses a two-state Markov model:
        - State 0 (normal): emit normal-ish RR
        - State 1 (ectopic): emit short RR followed by compensatory pause
        """
        rr = np.zeros(n_beats, dtype=np.float64)
        state = 0
        ectopic_prob = 0.12
        recovery_prob = 0.7

        for i in range(n_beats):
            if state == 0:
                # Normal beat
                rr[i] = base_rr_ms + self.rng.normal(0, 12.0)
                if self.rng.random() < ectopic_prob:
                    state = 1
            else:
                # Premature beat (short RR)
                rr[i] = base_rr_ms * self.rng.uniform(0.45, 0.65)
                state = 2  # next will be compensatory pause
            if state == 2:
                # Compensatory pause (long RR)
                if i + 1 < n_beats:
                    rr[i + 1] = base_rr_ms * self.rng.uniform(1.3, 1.7)
                    i += 1  # skip the compensatory beat in the loop
                state = 0

        # Fix any zeros from the loop skip
        rr[rr == 0] = base_rr_ms + self.rng.normal(0, 10.0)
        return rr

    def generate_hr_from_rr(self, rr_ms: np.ndarray, sample_rate: int = HR_HZ) -> np.ndarray:
        """Convert RR intervals to a heart-rate time-series at *sample_rate*.

        Parameters
        ----------
        rr_ms : np.ndarray
            RR intervals in milliseconds.
        sample_rate : int
            Desired output sample rate (Hz).

        Returns
        -------
        np.ndarray
            Heart-rate values in BPM, one per sample at *sample_rate*.
        """
        total_duration_s = rr_ms.sum() / 1000.0
        n_output = int(total_duration_s * sample_rate)
        hr = np.zeros(n_output, dtype=np.float64)

        # Map each output sample to the instantaneous HR from its RR interval
        beat_times_s = np.cumsum(rr_ms) / 1000.0
        beat_hr = 60_000.0 / rr_ms  # BPM at each beat

        output_times = np.arange(n_output, dtype=np.float64) / sample_rate
        # Linear interpolation of HR between beats
        hr = np.interp(output_times, beat_times_s, beat_hr, left=beat_hr[0], right=beat_hr[-1])
        return hr

    def generate_hr_time_series(
        self,
        n_samples: int,
        base_hr: float = 70.0,
        rhythm_type: str = "normal",
        sample_rate: int = HR_HZ,
    ) -> np.ndarray:
        """Generate a heart-rate time-series directly.

        Parameters
        ----------
        n_samples : int
            Number of output samples.
        base_hr : float
            Target resting HR in BPM.
        rhythm_type : str
            Rhythm type string.
        sample_rate : int
            Output sample rate (Hz).

        Returns
        -------
        np.ndarray
            Heart-rate in BPM, shape ``(n_samples,)``.
        """
        duration_s = n_samples / sample_rate
        n_beats = int(base_hr / 60.0 * duration_s * 1.1)  # 10% margin
        rr = self.generate_rr_intervals(n_beats, base_hr, rhythm_type)
        hr = self.generate_hr_from_rr(rr, sample_rate)
        # Trim or pad to exact length
        if len(hr) >= n_samples:
            return hr[:n_samples]
        return np.pad(hr, (0, n_samples - len(hr)), mode="edge")


# ---------------------------------------------------------------------------
# FallSimulator
# ---------------------------------------------------------------------------
class FallSimulator:
    """Generate realistic fall acceleration patterns.

    Models three phases:
    1. **Pre-fall** — normal motion or imbalance
    2. **Freefall** — reduced gravity (~0.3-0.8 g) for 0.3-0.5 s
    3. **Impact** — sharp spike (3-8 g) for 50-100 ms
    4. **Post-impact** — damped oscillation decaying to stillness
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(
        self,
        n_samples: int,
        sample_rate: int = ACCEL_HZ,
    ) -> np.ndarray:
        """Generate a fall event embedded in *n_samples* of data.

        The fall occurs roughly 20-30 % into the recording.

        Parameters
        ----------
        n_samples : int
            Total samples to generate.
        sample_rate : int
            Accelerometer sample rate.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 3)`` — accelerometer during a fall.
        """
        accel = np.zeros((n_samples, 3), dtype=np.float64)
        accel[:, 1] = GRAVITY  # gravity on y-axis

        # Fall onset: 20-30% into the recording
        onset_idx = int(n_samples * self.rng.uniform(0.20, 0.30))
        onset_idx = max(10, min(onset_idx, n_samples - int(3.0 * sample_rate)))

        # --- Phase 1: Free-fall (0.3-0.5 s) ---
        freefall_dur = self.rng.uniform(0.3, 0.5)
        freefall_samples = int(freefall_dur * sample_rate)
        freefall_end = min(onset_idx + freefall_samples, n_samples)

        # Gravity drops to ~0.5-0.8 g (not perfect freefall due to air resistance / clothing)
        gravity_reduction = self.rng.uniform(0.5, 0.8)
        accel[onset_idx:freefall_end, 1] = GRAVITY * (1 - gravity_reduction)

        # Slight forward tilt during fall
        tilt_rate = self.rng.uniform(1.0, 3.0)  # rad/s
        t_free = np.arange(freefall_samples, dtype=np.float64) / sample_rate
        accel[onset_idx:freefall_end, 0] += tilt_rate * t_free * 0.5

        # --- Phase 2: Impact spike (50-100 ms) ---
        impact_start = freefall_end
        impact_dur = self.rng.uniform(0.05, 0.10)
        impact_samples = max(int(impact_dur * sample_rate), 2)
        impact_end = min(impact_start + impact_samples, n_samples)

        impact_g = self.rng.uniform(3.0, 8.0)  # peak impact in g
        impact_height = impact_g * GRAVITY

        if impact_end > impact_start:
            # Sharp spike
            t_impact = np.arange(impact_end - impact_start, dtype=np.float64)
            spike = impact_height * np.exp(-3.0 * t_impact / max(1, len(t_impact)))
            accel[impact_start:impact_end, 1] += spike
            accel[impact_start:impact_end, 0] += spike * self.rng.uniform(0.1, 0.3)

        # --- Phase 3: Post-impact damped oscillation + stillness ---
        still_start = impact_end
        oscillation_dur = self.rng.uniform(0.3, 0.8)
        oscillation_samples = int(oscillation_dur * sample_rate)
        oscillation_end = min(still_start + oscillation_samples, n_samples)

        if oscillation_end > still_start:
            t_osc = np.arange(oscillation_end - still_start, dtype=np.float64) / sample_rate
            osc_freq = self.rng.uniform(5.0, 15.0)
            decay = 0.3 * GRAVITY * np.exp(-8.0 * t_osc)
            accel[still_start:oscillation_end, 1] += decay * np.sin(
                2 * np.pi * osc_freq * t_osc
            )
            accel[still_start:oscillation_end, 0] += decay * 0.3 * np.cos(
                2 * np.pi * osc_freq * t_osc
            )

        # Post-fall stillness (low variance)
        if oscillation_end < n_samples:
            accel[oscillation_end:, 1] = GRAVITY + self.rng.normal(0, 0.02, size=n_samples - oscillation_end)
            accel[oscillation_end:, 0] = self.rng.normal(0, 0.02, size=n_samples - oscillation_end)
            accel[oscillation_end:, 2] = self.rng.normal(0, 0.02, size=n_samples - oscillation_end)

        # Pre-fall normal motion
        if onset_idx > 0:
            accel[:onset_idx, 1] += 0.5 * np.sin(
                2 * np.pi * 1.8 * np.arange(onset_idx, dtype=np.float64) / sample_rate
            )
            accel[:onset_idx] += self.rng.normal(0, 0.05, size=(onset_idx, 3))

        # General noise
        accel += self.rng.normal(0, 0.03, size=(n_samples, 3))

        return accel


# ---------------------------------------------------------------------------
# SyntheticDataGenerator (main class)
# ---------------------------------------------------------------------------
class SyntheticDataGenerator:
    """Generate synthetic wearable sensor sessions for model training.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = SyntheticDataGenerator(seed=42)
    >>> session = gen.generate_normal_session(duration_sec=60)
    >>> session = gen.generate_condition_session("fever", duration_sec=120)
    """

    ALL_CONDITIONS: List[str] = list(CONDITIONS)

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._session_counter = 0
        self._base_rng = np.random.default_rng(seed)
        self._sensor = SensorSimulator(self._base_rng)
        self._ppg = PPGSimulator(self._base_rng)
        self._rhythm = HeartRhythmSimulator(self._base_rng)
        self._fall = FallSimulator(self._base_rng)

    def _next_session_id(self) -> str:
        self._session_counter += 1
        return f"synth_{self._session_counter:04d}"

    def _make_timestamp(self, offset_sec: float = 0.0) -> str:
        base = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        return (base + timedelta(seconds=offset_sec)).isoformat()

    # ------------------------------------------------------------------
    # Normal session
    # ------------------------------------------------------------------
    def generate_normal_session(
        self,
        duration_sec: int = 300,
        subject_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a session of normal / healthy data.

        Parameters
        ----------
        duration_sec : int
            Duration of the session in seconds.
        subject_profile : dict or None
            Optional subject parameters (age, sex, resting_hr, activity_state).

        Returns
        -------
        dict
            Complete session dictionary ready for JSON serialisation.
        """
        profile = SubjectProfile.from_dict(subject_profile)
        rng = np.random.default_rng(self._base_rng.integers(0, 2**31))

        # Override sensor simulator RNG for this session
        sensor = SensorSimulator(rng)
        ppg_sim = PPGSimulator(rng)
        rhythm_sim = HeartRhythmSimulator(rng)

        session_id = self._next_session_id()
        ts_start = self._make_timestamp(self._session_counter * duration_sec)

        activity = profile.activity_state
        base_hr = profile.resting_hr

        # Generate full-session continuous signals, then slice into windows
        session_data = self._generate_full_session(
            duration_sec=duration_sec,
            base_hr=base_hr,
            activity=activity,
            condition=None,
            severity=None,
            rng=rng,
            sensor=sensor,
            ppg_sim=ppg_sim,
            rhythm_sim=rhythm_sim,
        )

        windows = self._segment_into_windows(
            session_data, duration_sec=duration_sec
        )

        ground_truth = {c: False for c in CONDITIONS}

        return {
            "session_id": session_id,
            "timestamp_start": ts_start,
            "sampling_config": SamplingConfig().to_dict(),
            "windows": windows,
            "metadata": {
                "subject_profile": {
                    "age": profile.age,
                    "sex": profile.sex,
                    "resting_hr": profile.resting_hr,
                },
                "condition": "normal",
                "severity": "none",
                "activity_state": activity,
                "sensor_placement": "wrist",
                "data_quality_score": round(rng.uniform(0.90, 0.99), 3),
            },
            "ground_truth": ground_truth,
        }

    # ------------------------------------------------------------------
    # Condition session
    # ------------------------------------------------------------------
    def generate_condition_session(
        self,
        condition: str,
        duration_sec: int = 300,
        severity: str = "moderate",
    ) -> Dict[str, Any]:
        """Generate a session with a specific health condition.

        Parameters
        ----------
        condition : str
            One of the supported condition names.
        duration_sec : int
            Duration in seconds.
        severity : str
            ``"mild"`` | ``"moderate"`` | ``"severe"``.

        Returns
        -------
        dict
            Complete session dictionary.

        Raises
        ------
        ValueError
            If *condition* is not recognised.
        """
        if condition not in CONDITIONS:
            raise ValueError(
                f"Unknown condition '{condition}'. Must be one of {CONDITIONS}"
            )

        rng = np.random.default_rng(self._base_rng.integers(0, 2**31))
        sensor = SensorSimulator(rng)
        ppg_sim = PPGSimulator(rng)
        rhythm_sim = HeartRhythmSimulator(rng)

        profile = SubjectProfile()
        base_hr = profile.resting_hr
        activity = "resting"

        # Condition-specific parameter overrides
        params = self._condition_params(condition, severity, rng)
        base_hr = params.get("base_hr", base_hr)
        activity = params.get("activity", activity)

        session_id = self._next_session_id()
        ts_start = self._make_timestamp(self._session_counter * duration_sec)

        if condition == "fall_detected":
            session_data = self._generate_fall_session(
                duration_sec=duration_sec,
                rng=rng,
                sensor=sensor,
            )
        elif condition == "sleep_problem":
            session_data = self._generate_sleep_problem_session(
                duration_sec=duration_sec,
                base_hr=base_hr,
                rng=rng,
                sensor=sensor,
                ppg_sim=ppg_sim,
                rhythm_sim=rhythm_sim,
            )
        else:
            session_data = self._generate_full_session(
                duration_sec=duration_sec,
                base_hr=base_hr,
                activity=activity,
                condition=condition,
                severity=severity,
                rng=rng,
                sensor=sensor,
                ppg_sim=ppg_sim,
                rhythm_sim=rhythm_sim,
            )

        windows = self._segment_into_windows(session_data, duration_sec)

        ground_truth = {c: c == condition for c in CONDITIONS}

        return {
            "session_id": session_id,
            "timestamp_start": ts_start,
            "sampling_config": SamplingConfig().to_dict(),
            "windows": windows,
            "metadata": {
                "subject_profile": {
                    "age": profile.age,
                    "sex": profile.sex,
                    "resting_hr": profile.resting_hr,
                },
                "condition": condition,
                "severity": severity,
                "activity_state": activity,
                "sensor_placement": "wrist",
                "data_quality_score": round(rng.uniform(0.88, 0.98), 3),
            },
            "ground_truth": ground_truth,
        }

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------
    def generate_dataset(
        self,
        num_sessions: int,
        conditions: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        include_labels: bool = True,
        duration_sec: int = 300,
    ) -> List[Dict[str, Any]]:
        """Generate multiple sessions and optionally save as JSON.

        Parameters
        ----------
        num_sessions : int
            Number of sessions to generate.
        conditions : list of str or None
            Conditions to include.  ``None`` → balanced mix of all conditions
            plus normal sessions.
        output_dir : str or None
            Directory to write JSON files.  ``None`` → return in-memory only.
        include_labels : bool
            Whether to include ``ground_truth`` in the output.

        Returns
        -------
        list of dict
            All generated session dictionaries.
        """
        if conditions is None:
            # Balanced: ~1 normal per condition + extra normals
            n_normal = max(1, num_sessions // (len(CONDITIONS) + 1))
            n_per_cond = (num_sessions - n_normal) // len(CONDITIONS)
            plan: List[Tuple[Optional[str], str]] = [("normal", "none")] * n_normal
            for cond in CONDITIONS:
                plan.extend([(cond, "moderate")] * n_per_cond)
            # Fill any remainder with normals
            while len(plan) < num_sessions:
                plan.append(("normal", "none"))
            plan = plan[:num_sessions]
        else:
            n_per = num_sessions // len(conditions)
            plan = []
            for c in conditions:
                plan.extend([(c, "moderate")] * n_per)
            while len(plan) < num_sessions:
                plan.append((conditions[0], "moderate"))
            plan = plan[:num_sessions]

        # Shuffle for variety
        rng_perm = np.random.default_rng(self.seed + 999)
        rng_perm.shuffle(plan)

        sessions: List[Dict[str, Any]] = []
        out_path = Path(output_dir) if output_dir else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        for idx, (cond, sev) in enumerate(plan):
            t0 = _time.time()
            if cond == "normal":
                sess = self.generate_normal_session(duration_sec=duration_sec)
            else:
                sess = self.generate_condition_session(
                    condition=cond, duration_sec=duration_sec, severity=sev
                )
            elapsed = _time.time() - t0
            logger.info(
                "Session %d/%d (%s) generated in %.2fs",
                idx + 1,
                num_sessions,
                cond,
                elapsed,
            )

            if not include_labels:
                sess.pop("ground_truth", None)

            sessions.append(sess)

            if out_path is not None:
                fname = out_path / f"{sess['session_id']}.json"
                with fname.open("w", encoding="utf-8") as fh:
                    json.dump(sess, fh, indent=2, ensure_ascii=False)

        logger.info("Dataset generation complete: %d sessions", len(sessions))
        return sessions

    # ------------------------------------------------------------------
    # Internal: full-session signal generation
    # ------------------------------------------------------------------
    def _generate_full_session(
        self,
        duration_sec: int,
        base_hr: float,
        activity: str,
        condition: Optional[str],
        severity: Optional[str],
        rng: np.random.Generator,
        sensor: SensorSimulator,
        ppg_sim: PPGSimulator,
        rhythm_sim: HeartRhythmSimulator,
    ) -> Dict[str, np.ndarray]:
        """Generate all sensor channels for the full session duration."""

        n_accel = duration_sec * ACCEL_HZ
        n_gyro = duration_sec * GYRO_HZ
        n_hr = duration_sec * HR_HZ
        n_ppg = duration_sec * PPG_HZ
        n_spo2 = duration_sec * SPO2_HZ
        n_temp = duration_sec * TEMP_HZ

        # --- Heart rate ---
        rhythm_type = "normal"
        effective_hr = base_hr

        if condition == "tachycardia":
            sev_mult = {"mild": 1.0, "moderate": 1.5, "severe": 2.0}.get(severity, 1.5)
            effective_hr = base_hr + 40 * sev_mult
            rhythm_type = "tachycardia"
        elif condition == "irregular_rhythm":
            rhythm_type = "irregular"
        elif condition == "fever":
            sev_mult = {"mild": 1.0, "moderate": 1.5, "severe": 2.0}.get(severity, 1.5)
            temp_offset = 2.0 * sev_mult
            effective_hr = base_hr + 10 * temp_offset
            rhythm_type = "tachycardia"
        elif condition == "fatigue":
            effective_hr = base_hr + 5
            rhythm_type = "normal"

        hr_series = rhythm_sim.generate_hr_time_series(
            n_hr, base_hr=effective_hr, rhythm_type=rhythm_type, sample_rate=HR_HZ
        )

        # --- Accelerometer ---
        accel = sensor.generate_accelerometer(n_accel, activity=activity)

        # --- Gyroscope ---
        gyro = sensor.generate_gyroscope(n_gyro, activity=activity)

        # --- SpO2 ---
        spo2_mod = "low_spo2" if condition == "low_spo2" else None
        spo2_baseline = 97.5
        if condition == "low_spo2":
            sev_mult = {"mild": 1.0, "moderate": 1.5, "severe": 2.0}.get(severity, 1.5)
            spo2_baseline = 97.5 - 3.0 * sev_mult
        spo2 = sensor.generate_spo2(n_spo2, baseline=spo2_baseline, condition_modifier=spo2_mod)

        # --- Temperature ---
        temp_baseline = 36.5
        if condition == "fever":
            sev_mult = {"mild": 1.0, "moderate": 1.5, "severe": 2.0}.get(severity, 1.5)
            temp_baseline = 38.0 + 1.0 * sev_mult
        temp = sensor.generate_temperature(n_temp, baseline=temp_baseline)

        # --- PPG ---
        # Vary amplitude based on SpO2 and condition
        ppg_amplitude = 1.0
        if condition == "low_spo2":
            ppg_amplitude = 0.6
        ppg = ppg_sim.generate(n_ppg, heart_rate_bpm=np.mean(hr_series), amplitude=ppg_amplitude)

        return {
            "accelerometer": accel,
            "gyroscope": gyro,
            "heart_rate": hr_series,
            "spo2": spo2,
            "temperature": temp,
            "ppg": ppg,
        }

    def _generate_fall_session(
        self,
        duration_sec: int,
        rng: np.random.Generator,
        sensor: SensorSimulator,
    ) -> Dict[str, np.ndarray]:
        """Generate a fall event session with all sensor channels."""

        n_accel = duration_sec * ACCEL_HZ
        n_gyro = duration_sec * GYRO_HZ
        n_hr = duration_sec * HR_HZ
        n_ppg = duration_sec * PPG_HZ
        n_spo2 = duration_sec * SPO2_HZ
        n_temp = duration_sec * TEMP_HZ

        fall_sim = FallSimulator(rng)

        # Accelerometer — fall dominates
        accel = fall_sim.generate(n_accel, sample_rate=ACCEL_HZ)

        # Gyroscope — rotation during fall
        gyro = np.zeros((n_gyro, 3), dtype=np.float64)
        fall_onset_sample = int(n_gyro * rng.uniform(0.20, 0.30))
        # Moderate rotation during fall
        fall_window = min(int(0.8 * GYRO_HZ), n_gyro - fall_onset_sample)
        if fall_window > 0:
            t_rot = np.arange(fall_window, dtype=np.float64) / GYRO_HZ
            rot_freq = rng.uniform(2.0, 6.0)
            gyro[fall_onset_sample : fall_onset_sample + fall_window, 0] = (
                0.8 * np.sin(2 * np.pi * rot_freq * t_rot) * np.exp(-3 * t_rot)
            )
            gyro[fall_onset_sample : fall_onset_sample + fall_window, 1] = (
                0.4 * np.cos(2 * np.pi * rot_freq * t_rot) * np.exp(-3 * t_rot)
            )
        gyro += rng.normal(0, 0.005, size=(n_gyro, 3))

        # Heart rate — slight spike after impact (stress response)
        base_hr = 72.0
        hr_series = self._rhythm.generate_hr_time_series(
            n_hr, base_hr=base_hr, rhythm_type="normal", sample_rate=HR_HZ
        )
        # Post-fall HR elevation
        impact_time_s = duration_sec * rng.uniform(0.25, 0.35)
        impact_sample_hr = int(impact_time_s * HR_HZ)
        if impact_sample_hr < n_hr:
            elev_len = min(int(5 * HR_HZ), n_hr - impact_sample_hr)
            hr_series[impact_sample_hr : impact_sample_hr + elev_len] += rng.uniform(15, 30)

        # SpO2 — may drop slightly
        spo2 = sensor.generate_spo2(n_spo2, baseline=97.0)

        # Temperature — normal
        temp = sensor.generate_temperature(n_temp, baseline=36.5)

        # PPG — follows HR
        ppg = self._ppg.generate(n_ppg, heart_rate_bpm=np.mean(hr_series))

        return {
            "accelerometer": accel,
            "gyroscope": gyro,
            "heart_rate": hr_series,
            "spo2": spo2,
            "temperature": temp,
            "ppg": ppg,
        }

    def _generate_sleep_problem_session(
        self,
        duration_sec: int,
        base_hr: float,
        rng: np.random.Generator,
        sensor: SensorSimulator,
        ppg_sim: PPGSimulator,
        rhythm_sim: HeartRhythmSimulator,
    ) -> Dict[str, np.ndarray]:
        """Generate a sleep-problem session over a long window.

        Features: restless motion, poor HR dipping, periodic SpO2 desaturations.
        """
        n_accel = duration_sec * ACCEL_HZ
        n_gyro = duration_sec * GYRO_HZ
        n_hr = duration_sec * HR_HZ
        n_ppg = duration_sec * PPG_HZ
        n_spo2 = duration_sec * SPO2_HZ
        n_temp = duration_sec * TEMP_HZ

        # Accelerometer — restless: intermittent bursts of motion
        accel = np.zeros((n_accel, 3), dtype=np.float64)
        accel[:, 1] = GRAVITY
        # Create restlessness episodes
        n_episodes = max(3, duration_sec // 60)
        for _ in range(n_episodes):
            ep_start = rng.integers(0, max(1, n_accel - 200))
            ep_len = rng.integers(50, 200)
            ep_end = min(ep_start + ep_len, n_accel)
            t_ep = np.arange(ep_end - ep_start, dtype=np.float64) / ACCEL_HZ
            freq = rng.uniform(1.0, 2.0)
            accel[ep_start:ep_end, 0] += rng.uniform(-0.5, 0.5) * np.sin(2 * np.pi * freq * t_ep)
            accel[ep_start:ep_end, 1] += rng.uniform(-0.3, 0.3) * np.sin(2 * np.pi * freq * t_ep + 0.5)
            accel[ep_start:ep_end, 2] += rng.uniform(-0.2, 0.2) * np.sin(2 * np.pi * freq * t_ep + 1.0)
        accel += rng.normal(0, 0.02, size=(n_accel, 3))

        # Gyroscope — small rotations during restless periods
        gyro = rng.normal(0, 0.002, size=(n_gyro, 3))

        # Heart rate — poor dipping (<10% drop during "sleep")
        # Normal nocturnal dipping is 10-20%; here only 5%
        sleep_hr = base_hr - 3  # barely dips
        awake_hr = base_hr + 5
        hr_series = np.zeros(n_hr, dtype=np.float64)
        # Alternate sleep/wake blocks
        block_len = rng.integers(100, 300)  # in HR samples
        is_sleep = True
        pos = 0
        while pos < n_hr:
            bl = min(block_len, n_hr - pos)
            if is_sleep:
                hr_series[pos : pos + bl] = sleep_hr + rng.normal(0, 3, size=bl)
            else:
                hr_series[pos : pos + bl] = awake_hr + rng.normal(0, 5, size=bl)
            pos += bl
            is_sleep = not is_sleep
            block_len = rng.integers(100, 300)
        # Modulate with breathing
        t_hr = np.arange(n_hr, dtype=np.float64) / HR_HZ
        hr_series += 5.0 * np.sin(2 * np.pi * 0.18 * t_hr)
        hr_series += rng.normal(0, 2, size=n_hr)

        # SpO2 — periodic desaturations (apnea events)
        spo2 = sensor.generate_spo2(n_spo2, baseline=96.5)
        n_desats = max(2, duration_sec // 45)
        for _ in range(n_desats):
            d_start = rng.integers(0, max(1, n_spo2 - 80))
            d_len = rng.integers(30, 80)
            d_end = min(d_start + d_len, n_spo2)
            drop = rng.uniform(3.0, 7.0)
            t_d = np.linspace(0, 1, d_end - d_start)
            spo2[d_start:d_end] -= drop * np.sin(np.pi * t_d)  # smooth dip

        # Temperature — slightly elevated (sleep thermoregulation disruption)
        temp = sensor.generate_temperature(n_temp, baseline=36.7)

        # PPG
        ppg = ppg_sim.generate(n_ppg, heart_rate_bpm=np.mean(hr_series))

        return {
            "accelerometer": accel,
            "gyroscope": gyro,
            "heart_rate": hr_series,
            "spo2": spo2,
            "temperature": temp,
            "ppg": ppg,
        }

    # ------------------------------------------------------------------
    # Condition-specific parameters
    # ------------------------------------------------------------------
    @staticmethod
    def _condition_params(
        condition: str, severity: str, rng: np.random.Generator
    ) -> Dict[str, Any]:
        """Return condition-specific physiological parameter overrides."""
        sev_mult = {"mild": 1.0, "moderate": 1.5, "severe": 2.0}.get(severity, 1.5)

        if condition == "tachycardia":
            return {
                "base_hr": 70.0 + 40 * sev_mult,
                "activity": "resting",
            }
        elif condition == "irregular_rhythm":
            return {"base_hr": 72.0, "activity": "resting"}
        elif condition == "low_spo2":
            return {"base_hr": 75.0 + 10 * sev_mult, "activity": "resting"}
        elif condition == "fever":
            return {
                "base_hr": 70.0 + 10 * (2.0 * sev_mult),
                "activity": "resting",
            }
        elif condition == "fall_detected":
            return {"base_hr": 72.0, "activity": "walking"}
        elif condition == "sleep_problem":
            return {"base_hr": 68.0, "activity": "resting"}
        elif condition == "fatigue":
            return {"base_hr": 73.0, "activity": "resting"}
        return {"base_hr": 70.0, "activity": "resting"}

    # ------------------------------------------------------------------
    # Window segmentation
    # ------------------------------------------------------------------
    @staticmethod
    def _segment_into_windows(
        session_data: Dict[str, np.ndarray],
        duration_sec: int,
        window_sec: float = 30.0,
        stride_sec: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """Slice full-session arrays into overlapping windows.

        Parameters
        ----------
        session_data : dict
            Keys are sensor names, values are full-session np.ndarray.
        duration_sec : int
            Total session duration in seconds.
        window_sec : float
            Window length in seconds.
        stride_sec : float
            Stride (hop) between windows in seconds.

        Returns
        -------
        list of dict
            Window dictionaries with sensor_data sub-dicts.
        """
        windows: List[Dict[str, Any]] = []

        n_windows = max(1, int((duration_sec - window_sec) / stride_sec) + 1)
        n_windows = max(1, min(n_windows, 50))  # cap at 50 for JSON size

        accel = session_data["accelerometer"]
        gyro = session_data["gyroscope"]
        hr = session_data["heart_rate"]
        spo2 = session_data["spo2"]
        temp = session_data["temperature"]
        ppg = session_data["ppg"]

        for w_idx in range(n_windows):
            t_start = w_idx * stride_sec

            # Index boundaries for each channel
            a0 = int(t_start * ACCEL_HZ)
            a1 = int((t_start + window_sec) * ACCEL_HZ)
            g0 = int(t_start * GYRO_HZ)
            g1 = int((t_start + window_sec) * GYRO_HZ)
            h0 = int(t_start * HR_HZ)
            h1 = int((t_start + window_sec) * HR_HZ)
            s0 = int(t_start * SPO2_HZ)
            s1 = int((t_start + window_sec) * SPO2_HZ)
            t0 = int(t_start * TEMP_HZ)
            t1 = int((t_start + window_sec) * TEMP_HZ)
            p0 = int(t_start * PPG_HZ)
            p1 = int((t_start + window_sec) * PPG_HZ)

            # Clamp
            a1 = min(a1, len(accel))
            g1 = min(g1, len(gyro))
            h1 = min(h1, len(hr))
            s1 = min(s1, len(spo2))
            t1 = min(t1, len(temp))
            p1 = min(p1, len(ppg))

            # Convert accelerometer to list of [ax, ay, az]
            accel_window = accel[a0:a1].tolist() if a0 < a1 else [[0.0, 9.81, 0.0]]
            gyro_window = gyro[g0:g1].tolist() if g0 < g1 else [[0.0, 0.0, 0.0]]

            windows.append(
                {
                    "window_id": w_idx,
                    "timestamp": (datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
                                  + timedelta(seconds=t_start)).isoformat(),
                    "duration_sec": window_sec,
                    "sensor_data": {
                        "accelerometer": accel_window,
                        "gyroscope": gyro_window,
                        "heart_rate": hr[h0:h1].tolist() if h0 < h1 else [70.0],
                        "temperature": temp[t0:t1].tolist() if t0 < t1 else [36.5],
                        "spo2": spo2[s0:s1].tolist() if s0 < s1 else [97.0],
                        "ppg": ppg[p0:p1].tolist() if p0 < p1 else [0.0],
                    },
                }
            )

        return windows
