"""
Sensor data preprocessor module.

Provides ``SensorPreprocessor`` with signal-processing utilities
(filtering, normalization, resampling, artifact detection, etc.)
and a full preprocessing pipeline for a loaded session dict.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from .loader import SensorDataLoader

logger = logging.getLogger(__name__)


class SensorPreprocessor:
    """Collection of static / instance methods for cleaning sensor data.

    The class is stateless; every method operates on NumPy arrays and
    returns new arrays (no in-place mutation).
    """

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def filter_signal(
        signal_data: np.ndarray,
        fs: float,
        lowcut: float = 0.5,
        highcut: float = 20.0,
        order: int = 4,
    ) -> np.ndarray:
        """Apply a Butterworth band-pass filter.

        Parameters
        ----------
        signal_data : np.ndarray
            1-D input signal.
        fs : float
            Sampling frequency in Hz.
        lowcut : float
            Lower cut-off frequency in Hz.
        highcut : float
            Upper cut-off frequency in Hz.
        order : int
            Filter order (default 4).

        Returns
        -------
        np.ndarray
            Filtered signal (same length as input).
        """
        if signal_data.ndim != 1:
            raise ValueError("signal_data must be 1-D")
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        # Clip to valid (0, 1) range for Butterworth design
        low = max(low, 1e-5)
        high = min(high, 1.0 - 1e-5)
        b, a = scipy_signal.butter(order, [low, high], btype="band")
        # Use filtfilt for zero-phase filtering
        filtered = scipy_signal.filtfilt(b, a, signal_data, padlen=min(3 * max(len(b), len(a)), len(signal_data) - 1))
        return filtered.astype(signal_data.dtype)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(
        signal_data: np.ndarray,
        method: str = "zscore",
    ) -> np.ndarray:
        """Normalize a 1-D signal.

        Parameters
        ----------
        signal_data : np.ndarray
            1-D input signal.
        method : str
            ``"zscore"`` (default) or ``"minmax"``.

        Returns
        -------
        np.ndarray
            Normalized signal.

        Raises
        ------
        ValueError
            If *method* is unknown or signal is constant for z-score.
        """
        if signal_data.ndim != 1:
            raise ValueError("signal_data must be 1-D")

        if method == "zscore":
            std = float(np.std(signal_data))
            if std < 1e-12:
                # Constant signal → return zeros
                return np.zeros_like(signal_data, dtype=np.float64)
            return ((signal_data - float(np.mean(signal_data))) / std).astype(
                np.float64
            )

        if method == "minmax":
            smin = float(np.min(signal_data))
            smax = float(np.max(signal_data))
            denom = smax - smin
            if denom < 1e-12:
                return np.full_like(signal_data, 0.5, dtype=np.float64)
            return ((signal_data - smin) / denom).astype(np.float64)

        raise ValueError(f"Unknown normalization method: {method!r}")

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def resample(
        signal_data: np.ndarray,
        orig_fs: float,
        target_fs: float,
    ) -> np.ndarray:
        """Resample *signal_data* from *orig_fs* to *target_fs*.

        Uses ``scipy.signal.resample`` for pure-rate changes and
        ``scipy.interpolate.interp1d`` as a fallback when durations
        are needed.

        Parameters
        ----------
        signal_data : np.ndarray
            1-D input signal.
        orig_fs : float
            Original sampling rate (Hz).
        target_fs : float
            Desired sampling rate (Hz).

        Returns
        -------
        np.ndarray
            Resampled signal.
        """
        if signal_data.ndim != 1:
            raise ValueError("signal_data must be 1-D")
        if orig_fs <= 0 or target_fs <= 0:
            raise ValueError("Sampling rates must be positive")

        n_orig = len(signal_data)
        duration = (n_orig - 1) / orig_fs  # seconds between first and last sample
        n_target = max(1, int(round(duration * target_fs)) + 1)

        if abs(orig_fs - target_fs) < 1e-9:
            return signal_data.copy()

        resampled = scipy_signal.resample(signal_data, n_target)
        return resampled.astype(signal_data.dtype)

    # ------------------------------------------------------------------
    # Outlier removal
    # ------------------------------------------------------------------

    @staticmethod
    def remove_outliers(
        signal_data: np.ndarray,
        n_std: float = 3.0,
    ) -> np.ndarray:
        """Clip values beyond *n_std* standard deviations from the mean.

        Parameters
        ----------
        signal_data : np.ndarray
            1-D input signal.
        n_std : float
            Number of standard deviations for clipping bounds.

        Returns
        -------
        np.ndarray
            Clipped signal.
        """
        if signal_data.ndim != 1:
            raise ValueError("signal_data must be 1-D")
        mu = float(np.mean(signal_data))
        sigma = float(np.std(signal_data))
        lo = mu - n_std * sigma
        hi = mu + n_std * sigma
        return np.clip(signal_data, lo, hi).astype(signal_data.dtype)

    # ------------------------------------------------------------------
    # Motion-artifact detection (for PPG)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_motion_artifacts(
        accel_mag: np.ndarray,
        threshold: float = 2.0,
    ) -> np.ndarray:
        """Return a boolean mask flagging high-motion segments.

        Parameters
        ----------
        accel_mag : np.ndarray
            1-D magnitude of accelerometer signal (g-units).
        threshold : float
            Acceleration threshold above which samples are marked as
            motion-corrupted (default 2.0 g).

        Returns
        -------
        np.ndarray
            Boolean array, ``True`` where motion artifacts are detected.
        """
        if accel_mag.ndim != 1:
            raise ValueError("accel_mag must be 1-D")
        return accel_mag > threshold

    # ------------------------------------------------------------------
    # Interpolation of missing data
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate_missing(
        data: np.ndarray,
        method: str = "linear",
        max_gap: int = 5,
    ) -> np.ndarray:
        """Fill small ``NaN`` gaps via interpolation.

        Gaps longer than *max_gap* consecutive ``NaN`` values are left
        as ``NaN``.

        Parameters
        ----------
        data : np.ndarray
            1-D array potentially containing ``NaN`` values.
        method : str
            Interpolation method forwarded to ``scipy.interpolate.interp1d``
            (default ``"linear"``).
        max_gap : int
            Maximum consecutive ``NaN`` values to interpolate (default 5).

        Returns
        -------
        np.ndarray
            Array with small gaps filled.
        """
        if data.ndim != 1:
            raise ValueError("data must be 1-D")

        out = data.copy()
        nans = np.isnan(out)
        if not np.any(nans):
            return out

        valid_idx = np.where(~nans)[0]
        if len(valid_idx) < 2:
            # Cannot interpolate with fewer than 2 valid points
            return out

        nan_idx = np.where(nans)[0]
        # Identify contiguous NaN runs
        gaps = np.split(nan_idx, np.where(np.diff(nan_idx) != 1)[0] + 1)

        x_valid = valid_idx.astype(np.float64)
        y_valid = out[valid_idx].astype(np.float64)
        f_interp = interp1d(
            x_valid, y_valid, kind=method, fill_value="extrapolate"
        )

        for gap in gaps:
            if len(gap) <= max_gap:
                out[gap] = f_interp(gap.astype(np.float64))

        return out.astype(data.dtype)

    # ------------------------------------------------------------------
    # Full session preprocessing pipeline
    # ------------------------------------------------------------------

    def process_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full preprocessing pipeline on a loaded session dict.

        Steps per window:
        1. Interpolate missing values.
        2. Remove outliers from accelerometer and gyroscope.
        3. Band-pass filter accelerometer, gyroscope, and PPG.
        4. Normalize accelerometer magnitude and PPG.

        Parameters
        ----------
        data : dict
            Session dict as returned by :class:`SensorDataLoader`.

        Returns
        -------
        dict
            Preprocessed session (same structure, arrays replaced).
        """
        from ..config import SAMPLING_RATES

        for win in data.get("windows", []):
            sensor = win.get("sensor_data", {})

            # 1. Interpolate missing data in all channels
            for key in list(sensor.keys()):
                arr = sensor[key]
                if arr.ndim == 1:
                    sensor[key] = self.interpolate_missing(arr)

            # 2. Outlier removal on accelerometer (3 axes → per-column)
            accel = sensor.get("accelerometer")
            if accel is not None and accel.ndim == 2:
                for col in range(accel.shape[1]):
                    accel[:, col] = self.remove_outliers(accel[:, col])
                sensor["accelerometer"] = accel

            # 3. Band-pass filter time-series channels
            accel_fs = float(SAMPLING_RATES.ACCEL)
            gyro_fs = float(SAMPLING_RATES.GYRO)
            ppg_fs = float(SAMPLING_RATES.PPG)

            if accel is not None and accel.ndim == 2:
                for col in range(accel.shape[1]):
                    accel[:, col] = self.filter_signal(
                        accel[:, col], accel_fs, lowcut=0.5, highcut=20.0
                    )
                sensor["accelerometer"] = accel

            gyro = sensor.get("gyroscope")
            if gyro is not None and gyro.ndim == 2:
                for col in range(gyro.shape[1]):
                    gyro[:, col] = self.filter_signal(
                        gyro[:, col], gyro_fs, lowcut=0.5, highcut=20.0
                    )
                sensor["gyroscope"] = gyro

            ppg = sensor.get("ppg")
            if ppg is not None and ppg.ndim == 1:
                sensor["ppg"] = self.filter_signal(
                    ppg, ppg_fs, lowcut=0.5, highcut=8.0
                )

            # 4. Normalize PPG
            if ppg is not None and ppg.ndim == 1:
                sensor["ppg"] = self.normalize(ppg, method="zscore")

        return data
