"""
Temporal smoothing for health symptom predictions.

Applies N-of-M voting across a sliding window of consecutive predictions
to suppress single-window false positives.  A condition is flagged as
``detected`` only when at least *M* of the last *N* windows agree.

Also implements a cooldown mechanism so that chronic conditions (e.g.
fever lasting hours) do not produce repetitive alerts.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """Temporal smoothing using N-of-M voting across consecutive windows.

    Maintains a sliding buffer of the last *window_buffer_size* window
    predictions.  A condition is ``detected`` only if it was detected in
    at least *min_detections* of the buffered windows.

    Parameters
    ----------
    window_buffer_size : int
        Number of past windows kept in the sliding buffer (default ``5``).
    min_detections : int
        Minimum number of windows that must detect a condition for it to
        be reported after smoothing (default ``3``).
    cooldown_seconds : float
        After a condition is triggered, suppress re-triggering for this
        many seconds (default ``60``).  Set to ``0`` to disable cooldown.

    Notes
    -----
    The cooldown is per-condition — different conditions maintain
    independent cooldown timers.
    """

    def __init__(
        self,
        window_buffer_size: int = 5,
        min_detections: int = 3,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if window_buffer_size < 1:
            raise ValueError("window_buffer_size must be >= 1")
        if min_detections < 1:
            raise ValueError("min_detections must be >= 1")
        if min_detections > window_buffer_size:
            raise ValueError(
                "min_detections must be <= window_buffer_size"
            )

        self._buffer_size = window_buffer_size
        self._min_detections = min_detections
        self._cooldown_seconds = cooldown_seconds

        # Sliding buffer of (timestamp, prediction_dict) tuples
        self._buffer: deque = deque(maxlen=window_buffer_size)

        # Cooldown state: condition_name → timestamp of last trigger
        self._cooldown_state: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self, window_prediction: Dict[str, Any], timestamp: float
    ) -> Dict[str, Any]:
        """Add a new window prediction and return the smoothed result.

        Parameters
        ----------
        window_prediction : dict
            Output of :meth:`Predictor.predict` — a dict mapping condition
            names to ``{"detected": bool, "probability": float,
            "confidence": float}``.
        timestamp : float
            Unix-epoch timestamp (seconds) of this window.

        Returns
        -------
        dict
            Smoothed prediction dict in the same structure as the input.
            ``detected`` flags are updated according to N-of-M voting and
            cooldown logic.  ``probability`` and ``confidence`` values are
            averaged across the buffer.
        """
        # Add to buffer
        self._buffer.append((timestamp, window_prediction))

        # Apply smoothing
        return self._smooth(timestamp)

    def reset(self) -> None:
        """Clear the sliding buffer and cooldown state."""
        self._buffer.clear()
        self._cooldown_state.clear()
        logger.debug("TemporalSmoother buffer reset")

    def get_status(self) -> Dict[str, Any]:
        """Return the current buffer state for debugging.

        Returns
        -------
        dict
            Contains ``buffer_size``, ``min_detections``,
            ``cooldown_seconds``, ``buffered_windows``, and
            ``cooldown_state``.
        """
        return {
            "buffer_size": self._buffer_size,
            "min_detections": self._min_detections,
            "cooldown_seconds": self._cooldown_seconds,
            "buffered_windows": len(self._buffer),
            "cooldown_state": dict(self._cooldown_state),
        }

    # ------------------------------------------------------------------
    # Internal: smoothing logic
    # ------------------------------------------------------------------

    def _smooth(self, current_timestamp: float) -> Dict[str, Any]:
        """Apply N-of-M voting and cooldown to the current buffer."""
        if not self._buffer:
            return {}

        # Determine the full set of condition names from the latest window
        latest_pred = self._buffer[-1][1]
        condition_names = list(latest_pred.keys())

        smoothed: Dict[str, Any] = {}

        for cond in condition_names:
            # Count detections across buffer
            detection_count = 0
            total_prob = 0.0
            total_conf = 0.0
            n_windows = len(self._buffer)

            for _ts, pred in self._buffer:
                entry = pred.get(cond, {})
                if entry.get("detected", False):
                    detection_count += 1
                total_prob += entry.get("probability", 0.0)
                total_conf += entry.get("confidence", 0.0)

            # Average probability and confidence across buffered windows
            avg_prob = total_prob / n_windows if n_windows > 0 else 0.0
            avg_conf = total_conf / n_windows if n_windows > 0 else 0.0

            # N-of-M voting
            voted_detected = detection_count >= self._min_detections

            # Cooldown check
            in_cooldown = self._is_in_cooldown(cond, current_timestamp)

            # Final detection flag
            if voted_detected and not in_cooldown:
                final_detected = True
                # Record cooldown trigger
                self._record_trigger(cond, current_timestamp)
            else:
                final_detected = False

            smoothed[cond] = {
                "detected": final_detected,
                "probability": round(avg_prob, 6),
                "confidence": round(avg_conf, 6),
                "_detection_count": detection_count,
                "_voted": voted_detected,
                "_in_cooldown": in_cooldown,
            }

        return smoothed

    # ------------------------------------------------------------------
    # Internal: cooldown helpers
    # ------------------------------------------------------------------

    def _is_in_cooldown(self, condition: str, current_time: float) -> bool:
        """Check whether *condition* is still in its cooldown window."""
        if self._cooldown_seconds <= 0:
            return False

        last_trigger = self._cooldown_state.get(condition)
        if last_trigger is None:
            return False

        elapsed = current_time - last_trigger
        return elapsed < self._cooldown_seconds

    def _record_trigger(self, condition: str, timestamp: float) -> None:
        """Record the trigger time for cooldown tracking."""
        self._cooldown_state[condition] = timestamp
