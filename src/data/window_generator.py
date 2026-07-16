"""
Sliding window generator module.

Provides ``WindowGenerator`` for segmenting a continuous sensor session
into fixed-duration, overlapping windows ready for feature extraction.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import SAMPLING_RATES, WINDOW_CONFIG, WindowConfig

logger = logging.getLogger(__name__)


class WindowGenerator:
    """Generate overlapping windows from a preprocessed session dict.

    Parameters
    ----------
    session_data : dict
        Session dictionary as returned by :class:`SensorDataLoader` (and
        optionally preprocessed by :class:`SensorPreprocessor`).
    window_duration_sec : float, optional
        Override the default window duration (seconds).
    stride_sec : float, optional
        Override the default stride (seconds).
    """

    def __init__(
        self,
        session_data: Dict[str, Any],
        window_duration_sec: Optional[float] = None,
        stride_sec: Optional[float] = None,
    ) -> None:
        self._session = session_data
        self._window_cfg = WindowConfig(
            WINDOW_DURATION_SEC=(
                window_duration_sec
                if window_duration_sec is not None
                else WINDOW_CONFIG.WINDOW_DURATION_SEC
            ),
            STRIDE_SEC=(
                stride_sec
                if stride_sec is not None
                else WINDOW_CONFIG.STRIDE_SEC
            ),
            MIN_WINDOW_SAMPLES=WINDOW_CONFIG.MIN_WINDOW_SAMPLES,
        )

    # ------------------------------------------------------------------
    # Public configuration helpers
    # ------------------------------------------------------------------

    def set_window_duration(self, duration_sec: float) -> None:
        """Update the window duration (seconds).

        Parameters
        ----------
        duration_sec : float
            New window length in seconds.  Must be positive.
        """
        if duration_sec <= 0:
            raise ValueError("window duration must be positive")
        self._window_cfg.WINDOW_DURATION_SEC = duration_sec

    def set_stride(self, stride_sec: float) -> None:
        """Update the stride between consecutive windows (seconds).

        Parameters
        ----------
        stride_sec : float
            New stride in seconds.  Must be positive and ≤ window duration.
        """
        if stride_sec <= 0:
            raise ValueError("stride must be positive")
        if stride_sec > self._window_cfg.WINDOW_DURATION_SEC:
            raise ValueError("stride cannot exceed window duration")
        self._window_cfg.STRIDE_SEC = stride_sec

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def generate_windows(self, session_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Segment the session into overlapping windows.

        Parameters
        ----------
        session_data : dict, optional
            If provided, overrides the session data supplied at
            construction time.

        Returns
        -------
        list of dict
            Each dict has the keys:

            - ``window_id`` – sequential integer id
            - ``start_sample`` – sample index of window start
            - ``end_sample`` – sample index of window end (exclusive)
            - ``start_time`` – start time in seconds
            - ``end_time`` – end time in seconds
            - ``sensor_data`` – trimmed sensor arrays for this window
            - ``metadata`` – session metadata copy

            Empty list is returned when the session is too short to
            produce even a single valid window.
        """
        data = session_data if session_data is not None else self._session
        windows_src = data.get("windows", [])
        if not windows_src:
            logger.warning("No windows found in session %s", data.get("session_id"))
            return []

        # Use the first window's sensor data as the full continuous signal.
        # In a real pipeline the session would already contain a single
        # continuous segment; here we concatenate all listed windows.
        sensor_channels: Dict[str, List[np.ndarray]] = {}
        for win in windows_src:
            for ch, arr in win.get("sensor_data", {}).items():
                # Ensure we always work with NumPy arrays
                if not isinstance(arr, np.ndarray):
                    arr = np.asarray(arr, dtype=np.float64)
                sensor_channels.setdefault(ch, []).append(arr)

        # Concatenate per channel
        continuous: Dict[str, np.ndarray] = {}
        for ch, parts in sensor_channels.items():
            if parts[0].ndim == 1:
                continuous[ch] = np.concatenate(parts)
            else:
                continuous[ch] = np.concatenate(parts, axis=0)

        # Determine master sample count from the highest-rate channel
        max_samples = max(arr.shape[0] for arr in continuous.values())
        if max_samples == 0:
            return []

        # Determine effective sampling rate (use accel as reference)
        ref_rate = float(SAMPLING_RATES.ACCEL)
        win_samples = int(self._window_cfg.WINDOW_DURATION_SEC * ref_rate)
        stride_samples = int(self._window_cfg.STRIDE_SEC * ref_rate)

        if win_samples < self._window_cfg.MIN_WINDOW_SAMPLES:
            raise ValueError(
                f"Window size ({win_samples} samples) is smaller than "
                f"MIN_WINDOW_SAMPLES ({self._window_cfg.MIN_WINDOW_SAMPLES})"
            )

        result: List[Dict[str, Any]] = []
        win_id = 0
        start = 0

        while start + win_samples <= max_samples:
            end = start + win_samples
            win_data: Dict[str, Any] = {
                "window_id": win_id,
                "start_sample": start,
                "end_sample": end,
                "start_time": start / ref_rate,
                "end_time": end / ref_rate,
                "sensor_data": {},
                "metadata": copy.deepcopy(data.get("metadata", {})),
            }

            for ch, full_arr in continuous.items():
                win_data["sensor_data"][ch] = full_arr[start:end]

            result.append(win_data)
            win_id += 1
            start += stride_samples

        logger.info(
            "Generated %d windows (duration=%.1fs, stride=%.1fs) from session %s",
            len(result),
            self._window_cfg.WINDOW_DURATION_SEC,
            self._window_cfg.STRIDE_SEC,
            data.get("session_id", "unknown"),
        )
        return result
