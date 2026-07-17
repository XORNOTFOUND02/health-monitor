"""
Central configuration module for the Health Monitor project.

Defines all project-wide constants, dataclass-based configurations,
path conventions, sensor specifications, and clinical thresholds.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SYNTHETIC_RAW_DIR = DATA_DIR / "synthetic" / "raw"
SYNTHETIC_PROCESSED_DIR = DATA_DIR / "synthetic" / "processed"


# ---------------------------------------------------------------------------
# Sampling rates (Hz)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingRates:
    """Sampling rates for each sensor channel (Hz)."""

    ACCEL: int = 50       # MPU6500 accelerometer
    GYRO: int = 50        # MPU6500 gyroscope
    HR: int = 25          # MAX30102 heart rate
    PPG: int = 25         # Raw PPG waveform from MAX30102
    SPO2: int = 25        # SpO2 readings from MAX30102
    TEMP: int = 1         # Temperature (STTS22H + LM35)
    MAG: int = 25         # HMC5883L magnetometer


SAMPLING_RATES = SamplingRates()


# ---------------------------------------------------------------------------
# Window configuration
# ---------------------------------------------------------------------------
@dataclass
class WindowConfig:
    """Sliding-window parameters for segmentation."""

    WINDOW_DURATION_SEC: float = 30.0    # 30-second windows
    STRIDE_SEC: float = 5.0              # 5-second stride (6× overlap)
    MIN_WINDOW_SAMPLES: int = 10         # Minimum samples to form a valid window

    @property
    def accel_samples_per_window(self) -> int:
        """Number of accelerometer samples expected per window."""
        return int(self.WINDOW_DURATION_SEC * SAMPLING_RATES.ACCEL)

    @property
    def gyro_samples_per_window(self) -> int:
        """Number of gyroscope samples expected per window."""
        return int(self.WINDOW_DURATION_SEC * SAMPLING_RATES.GYRO)

    @property
    def hr_samples_per_window(self) -> int:
        """Number of heart-rate samples expected per window."""
        return int(self.WINDOW_DURATION_SEC * SAMPLING_RATES.HR)

    @property
    def ppg_samples_per_window(self) -> int:
        """Number of raw PPG samples expected per window."""
        return int(self.WINDOW_DURATION_SEC * SAMPLING_RATES.PPG)

    @property
    def spo2_samples_per_window(self) -> int:
        """Number of SpO2 samples expected per window."""
        return int(self.WINDOW_DURATION_SEC * SAMPLING_RATES.SPO2)

    @property
    def temp_samples_per_window(self) -> int:
        """Number of temperature samples expected per window."""
        return int(self.WINDOW_DURATION_SEC * SAMPLING_RATES.TEMP)


WINDOW_CONFIG = WindowConfig()


# ---------------------------------------------------------------------------
# Clinical / condition thresholds (rule-based labeling)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConditionThresholds:
    """Evidence-based thresholds used for rule-based label generation."""

    # -- Cardiac --
    TACHYCARDIA_BPM: int = 100
    BRADYCARDIA_BPM: int = 60

    # -- SpO2 --
    LOW_SPO2_THRESHOLD: float = 95.0
    SEVERE_LOW_SPO2: float = 90.0

    # -- Temperature --
    FEVER_TEMP_C: float = 38.0
    LOW_GRADE_FEVER_C: float = 37.5
    HYPOTHERMIA_TEMP_C: float = 35.0

    # -- Fall detection --
    FALL_ACCEL_THRESHOLD_G: float = 3.0
    FALL_STILLNESS_THRESHOLD: float = 0.05
    FALL_STILLNESS_DURATION_SEC: float = 5.0

    # -- Fatigue --
    FATIGUE_HRV_RMSSD_THRESHOLD: float = 20.0   # ms
    FATIGUE_RESTING_HR_ELEVATION: int = 5        # BPM above baseline

    # -- Sleep --
    SLEEP_MOTION_THRESHOLD: float = 0.15         # Accel std below this → still
    APNEA_SPO2_DROP_THRESHOLD: float = 3.0       # % drop from baseline

    # -- Rhythm --
    IRREGULAR_RR_CV_THRESHOLD: float = 0.15      # Coefficient of variation


THRESHOLDS = ConditionThresholds()


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
CARDIAC_MODEL_PATH: Path = MODELS_DIR / "cardiac_model.onnx"
RESPIRATORY_MODEL_PATH: Path = MODELS_DIR / "respiratory_model.onnx"
ACTIVITY_MODEL_PATH: Path = MODELS_DIR / "activity_model.onnx"
FEATURE_NAMES_PATH: Path = MODELS_DIR / "feature_names.json"


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
GRAVITY: float = 9.81  # m/s²

# Canonical condition names (order matches model output vectors)
CONDITIONS: List[str] = [
    "tachycardia",
    "irregular_rhythm",
    "low_spo2",
    "fever",
    "fall_detected",
    "sleep_problem",
    "fatigue",
]
