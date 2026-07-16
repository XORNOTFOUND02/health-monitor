"""
Sensor data loader module.

Provides ``SensorDataLoader`` for reading JSON session files produced by
the wearable data collection pipeline and converting them into
dictionary structures with NumPy arrays for time-series channels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when a session JSON fails schema validation."""


class SensorDataLoader:
    """Load and validate sensor-data JSON sessions.

    The loader expects JSON files with the following top-level structure::

        {
            "session_id": "...",
            "timestamp_start": "...",
            "sampling_config": { ... },
            "windows": [
                {
                    "window_id": 0,
                    "timestamp": "...",
                    "duration_sec": 30.0,
                    "sensor_data": {
                        "accelerometer": [[ax, ay, az], ...],
                        "gyroscope": [[gx, gy, gz], ...],
                        "heart_rate": [bpm, ...],
                        "temperature": [deg_c, ...],
                        "spo2": [pct, ...],
                        "ppg": [adc, ...]
                    }
                }
            ],
            "metadata": { ... }
        }

    Parameters
    ----------
    strict : bool, optional
        If ``True`` (default), raise on schema violations.  If ``False``,
        log a warning and skip invalid windows/sessions.
    """

    _TOP_LEVEL_KEYS = {
        "session_id",
        "timestamp_start",
        "sampling_config",
        "windows",
        "metadata",
    }

    _WINDOW_KEYS = {"window_id", "timestamp", "duration_sec", "sensor_data"}

    _SENSOR_KEYS = {"accelerometer", "gyroscope", "heart_rate", "temperature"}

    _OPTIONAL_SENSOR_KEYS = {"spo2", "ppg", "gyroscope"}

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_session(self, filepath: str | Path) -> Dict[str, Any]:
        """Load a single JSON session file.

        Parameters
        ----------
        filepath : str or Path
            Absolute path to the JSON file.

        Returns
        -------
        dict
            Parsed session with time-series data stored as NumPy arrays.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        SchemaValidationError
            If the JSON content fails validation and ``strict=True``.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Session file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)

        self.validate_schema(raw)
        return self._convert_arrays(raw)

    def load_dataset(
        self,
        directory: str | Path,
        pattern: str = "*.json",
    ) -> List[Dict[str, Any]]:
        """Load every JSON session matching *pattern* from *directory*.

        Parameters
        ----------
        directory : str or Path
            Folder containing session JSON files.
        pattern : str, optional
            Glob pattern to select files (default ``"*.json"``).

        Returns
        -------
        list of dict
            Successfully loaded and validated sessions.  Invalid files are
            skipped (with a log warning) when ``strict=False``; otherwise
            the first validation error is raised.
        """
        folder = Path(directory)
        if not folder.is_dir():
            raise NotADirectoryError(f"Dataset directory does not exist: {folder}")

        sessions: List[Dict[str, Any]] = []
        for path in sorted(folder.glob(pattern)):
            try:
                session = self.load_session(path)
                sessions.append(session)
            except (SchemaValidationError, json.JSONDecodeError, KeyError) as exc:
                msg = f"Skipping {path.name}: {exc}"
                if self.strict:
                    raise
                logger.warning(msg)

        logger.info("Loaded %d sessions from %s", len(sessions), folder)
        return sessions

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def validate_schema(self, data: Dict[str, Any]) -> bool:
        """Validate *data* against the expected session schema.

        Parameters
        ----------
        data : dict
            Parsed JSON session.

        Returns
        -------
        bool
            ``True`` when the schema is valid.

        Raises
        ------
        SchemaValidationError
            When any required key or structure is missing / malformed.
        """
        # Top-level keys
        missing = self._TOP_LEVEL_KEYS - set(data.keys())
        if missing:
            raise SchemaValidationError(
                f"Missing top-level keys: {sorted(missing)}"
            )

        # windows must be a non-empty list
        windows = data.get("windows")
        if not isinstance(windows, list) or len(windows) == 0:
            raise SchemaValidationError("'windows' must be a non-empty list")

        for idx, win in enumerate(windows):
            self._validate_window(win, idx)

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_window(self, win: Dict[str, Any], idx: int) -> None:
        """Validate a single window dict."""
        missing = self._WINDOW_KEYS - set(win.keys())
        if missing:
            raise SchemaValidationError(
                f"Window {idx}: missing keys {sorted(missing)}"
            )

        if not isinstance(win.get("window_id"), (int, np.integer)):
            raise SchemaValidationError(
                f"Window {idx}: 'window_id' must be an integer"
            )

        if not isinstance(win.get("duration_sec"), (int, float, np.floating)):
            raise SchemaValidationError(
                f"Window {idx}: 'duration_sec' must be numeric"
            )

        sensor = win.get("sensor_data")
        if not isinstance(sensor, dict):
            raise SchemaValidationError(
                f"Window {idx}: 'sensor_data' must be a dict"
            )

        missing_sensors = self._SENSOR_KEYS - set(sensor.keys())
        if missing_sensors:
            raise SchemaValidationError(
                f"Window {idx}: missing sensor channels {sorted(missing_sensors)}"
            )

        # Each required sensor value must be a non-empty list
        for key in self._SENSOR_KEYS:
            val = sensor[key]
            if not isinstance(val, list) or len(val) == 0:
                raise SchemaValidationError(
                    f"Window {idx}: sensor '{key}' must be a non-empty list"
                )

    def _convert_arrays(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert list-based sensor data to NumPy arrays in-place."""
        for win in data.get("windows", []):
            sensor = win.get("sensor_data", {})
            for key, values in sensor.items():
                arr = np.asarray(values, dtype=np.float64)
                # Flatten 1-D channels that might be nested lists of scalars
                if arr.ndim > 1 and key in ("heart_rate", "temperature", "spo2", "ppg"):
                    arr = arr.ravel()
                sensor[key] = arr
        return data
