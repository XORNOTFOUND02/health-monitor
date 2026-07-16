"""
Standardised JSON response builder for the health monitor API.

Formats model predictions, data-quality scores, and metadata into a
consistent envelope that the front-end and mobile clients can parse
reliably.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Medical disclaimer (legal requirement)
# ---------------------------------------------------------------------------
_DISCLAIMER: str = (
    "This is NOT a medical device. Consult a physician for health concerns."
)

# ---------------------------------------------------------------------------
# Alert classification
# ---------------------------------------------------------------------------
# Conditions that require immediate attention
_CRITICAL_ALERTS: frozenset = frozenset({"low_spo2", "fall_detected"})
# Conditions that warrant a doctor visit
_WARNINGS: frozenset = frozenset({"tachycardia", "irregular_rhythm", "fever"})
# Informational / lifestyle conditions
_INFO: frozenset = frozenset({"sleep_problem", "fatigue"})


class ResponseBuilder:
    """Builds standardised JSON responses for the health monitor API.

    Methods
    -------
    build_response(predictions, data_quality, metadata=None)
        Format predictions into the canonical API envelope.
    build_health_status()
        Return a health-check response for ``/health``.
    build_error_response(error, code=500)
        Return a standardised error response.
    """

    def build_response(
        self,
        predictions: Dict[str, Any],
        data_quality: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format all predictions into a consistent response structure.

        Parameters
        ----------
        predictions : dict
            Mapping of condition name → ``{"detected": bool,
            "probability": float, "confidence": float}``.
        data_quality : float
            Composite data-quality score in [0.0, 1.0].
        metadata : dict, optional
            Extra metadata (e.g. model versions, processing time).

        Returns
        -------
        dict
            Complete API response payload.
        """
        # Ensure data_quality is within bounds
        data_quality = max(0.0, min(1.0, float(data_quality)))

        # Build per-condition predictions (strip internal keys)
        clean_predictions = self._clean_predictions(predictions)

        # Categorise alerts
        alert_summary = self._build_alert_summary(clean_predictions)

        # Timestamp in ISO-8601 UTC
        timestamp = self._utc_now_iso()

        # Model versions from metadata
        model_versions = {}
        if metadata and "model_versions" in metadata:
            model_versions = metadata["model_versions"]

        response: Dict[str, Any] = {
            "status": "success",
            "timestamp": timestamp,
            "data_quality_score": round(data_quality, 4),
            "predictions": clean_predictions,
            "alert_summary": alert_summary,
            "disclaimer": _DISCLAIMER,
            "model_versions": model_versions,
        }

        # Attach optional metadata
        if metadata:
            for key in ("processing_time_ms", "window_duration_sec", "feature_count"):
                if key in metadata:
                    response[key] = metadata[key]

        return response

    def build_health_status(self) -> Dict[str, Any]:
        """Return a health-check response for the ``/health`` endpoint.

        Returns
        -------
        dict
            Minimal status payload indicating the service is alive.
        """
        return {
            "status": "healthy",
            "timestamp": self._utc_now_iso(),
            "service": "health-monitor-inference",
            "version": "1.0.0",
        }

    def build_error_response(
        self, error: str, code: int = 500
    ) -> Dict[str, Any]:
        """Return a standardised error response.

        Parameters
        ----------
        error : str
            Human-readable error description.
        code : int
            HTTP-style status code (default ``500``).

        Returns
        -------
        dict
            Error payload with ``status``, ``error``, and ``code``.
        """
        return {
            "status": "error",
            "error": str(error),
            "code": int(code),
            "timestamp": self._utc_now_iso(),
            "disclaimer": _DISCLAIMER,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_predictions(
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Strip internal keys (those starting with ``_``) from predictions."""
        cleaned: Dict[str, Any] = {}
        for cond, entry in predictions.items():
            if not isinstance(entry, dict):
                cleaned[cond] = {
                    "detected": False,
                    "probability": 0.0,
                    "confidence": 0.0,
                }
                continue

            cleaned[cond] = {
                "detected": bool(entry.get("detected", False)),
                "probability": round(float(entry.get("probability", 0.0)), 6),
                "confidence": round(float(entry.get("confidence", 0.0)), 6),
            }
        return cleaned

    @staticmethod
    def _build_alert_summary(
        predictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Categorise detected conditions into critical / warning / info."""
        critical: List[str] = []
        warnings: List[str] = []
        info: List[str] = []

        for cond, entry in predictions.items():
            if not entry.get("detected", False):
                continue

            if cond in _CRITICAL_ALERTS:
                critical.append(cond)
            elif cond in _WARNINGS:
                warnings.append(cond)
            elif cond in _INFO:
                info.append(cond)
            else:
                # Unknown condition — treat as warning
                warnings.append(cond)

        return {
            "critical_alerts": sorted(critical),
            "warnings": sorted(warnings),
            "info": sorted(info),
            "total_alerts": len(critical) + len(warnings) + len(info),
        }

    @staticmethod
    def _utc_now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat(timespec="seconds")
