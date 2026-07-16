"""
Ensemble inference engine for health symptom detection.

Loads trained LightGBM models (cardiac, respiratory, activity) and a
rule-based fever detector, extracts features from incoming sensor windows,
and produces per-condition probability predictions.

Designed for Hugging Face Spaces CPU environments (2 vCPU, 16 GB RAM).
Models are loaded once at module scope so that cold-start latencies are
minimised across requests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import warnings

from ..config import CONDITIONS, MODELS_DIR
from ..features.extractor import FeatureExtractor
from ..models.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature order (loaded once from models/feature_names.json)
# ---------------------------------------------------------------------------
_DEFAULT_FEATURE_NAMES_PATH: Path = MODELS_DIR / "feature_names.json"

# ---------------------------------------------------------------------------
# Cardiac / respiratory / activity condition names (must match model output
# column order used during training)
# ---------------------------------------------------------------------------
_CARDIAC_CONDITIONS: List[str] = ["tachycardia", "irregular_rhythm"]
_RESPIRATORY_CONDITIONS: List[str] = ["low_spo2"]
_ACTIVITY_CONDITIONS: List[str] = ["fall_detected", "sleep_problem", "fatigue"]


class Predictor:
    """Loads trained models and runs ensemble inference on sensor windows.

    Pipeline for a single window:

    1. Extract features using :class:`FeatureExtractor`.
    2. Align the feature vector with the canonical order from
       ``feature_names.json`` (fill missing with 0.0).
    3. Run each model:
       a. ``CardiacModel`` → tachycardia, irregular_rhythm probabilities
       b. ``RespiratoryModel`` → low_spo2 probability
       c. ``ActivityModel`` → fall_detected, sleep_problem, fatigue probabilities
       d. ``RuleEngine`` → fever detection + data-quality score
    4. Combine into a unified prediction dict.

    Parameters
    ----------
    models_dir : str or Path
        Directory containing ``*.joblib`` model files,
        ``*.meta.json`` metadata, and ``feature_names.json``.
    use_onnx : bool
        If *True*, prefer ONNX models when available (currently unused
        fallback path).
    """

    def __init__(
        self,
        models_dir: str | Path = str(MODELS_DIR),
        use_onnx: bool = False,
    ) -> None:
        self._models_dir = Path(models_dir)
        self._use_onnx = use_onnx

        # Feature extractor (shared across calls)
        self._feature_extractor = FeatureExtractor(enable_logging=True)

        # Rule-based engine
        self._rule_engine = RuleEngine()

        # Canonical feature order — loaded once
        self._feature_names: List[str] = self._load_feature_names()

        # Model containers (populated by load_models)
        self._cardiac_models: Dict[str, Any] = {}
        self._respiratory_model: Any = None
        self._activity_models: Dict[str, Any] = {}

        # Metadata for response builder
        self._model_versions: Dict[str, str] = {}

        # Load all models eagerly so the first request is fast
        self.load_models(self._models_dir, self._use_onnx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full inference pipeline on a single sensor window.

        Parameters
        ----------
        window_data : dict
            Raw sensor data dictionary (same format expected by
            :meth:`FeatureExtractor.extract_all`).

        Returns
        -------
        dict
            Mapping of condition name → ``{"detected": bool,
            "probability": float, "confidence": float}``.

        Raises
        ------
        ValueError
            If *window_data* is empty or cannot be processed.
        """
        if not window_data:
            raise ValueError("window_data must be a non-empty dictionary")

        # Normalise sensor data to canonical dict format
        normalised = self._normalize_input(window_data)

        features = self._extract_features(normalised)
        X = self._align_features(features)
        data_quality = self._compute_data_quality(normalised)

        return self._run_ensemble(X, normalised, data_quality)

    def predict_proba(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return raw probabilities without binary thresholds.

        Identical to :meth:`predict` but the ``"detected"`` key is always
        ``False`` so callers can apply their own thresholds.

        Parameters
        ----------
        window_data : dict
            Raw sensor data dictionary.

        Returns
        -------
        dict
            Same structure as :meth:`predict` with ``detected=False``.
        """
        if not window_data:
            raise ValueError("window_data must be a non-empty dictionary")

        normalised = self._normalize_input(window_data)
        features = self._extract_features(normalised)
        X = self._align_features(features)
        data_quality = self._compute_data_quality(normalised)

        return self._run_ensemble(X, normalised, data_quality, threshold=None)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_models(
        self,
        models_dir: str | Path,
        use_onnx: bool = False,
    ) -> None:
        """Load all trained models from disk.

        Tries to use the model class ``.load()`` methods first; falls back
        to raw ``joblib.load()`` for the booster objects when the class
        import is unavailable.

        Parameters
        ----------
        models_dir : str or Path
            Directory containing model files.
        use_onnx : bool
            Prefer ONNX when available (not yet implemented).
        """
        models_dir = Path(models_dir)

        # ---- Cardiac ----
        self._cardiac_models = self._load_cardiac(models_dir)

        # ---- Respiratory ----
        self._respiratory_model = self._load_respiratory(models_dir)

        # ---- Activity ----
        self._activity_models = self._load_activity(models_dir)

        # ---- Metadata ----
        self._model_versions = self._load_model_versions(models_dir)

        loaded = []
        if self._cardiac_models:
            loaded.append("cardiac")
        if self._respiratory_model is not None:
            loaded.append("respiratory")
        if self._activity_models:
            loaded.append("activity")
        logger.info("Loaded models: %s", ", ".join(loaded) or "(none)")

    # ------------------------------------------------------------------
    # Feature alignment
    # ------------------------------------------------------------------

    def _align_features(self, features: Dict[str, float]) -> np.ndarray:
        """Order features according to ``feature_names.json``.

        Missing features are filled with ``0.0`` and any extra features
        not present in the canonical list are silently dropped.

        Parameters
        ----------
        features : dict
            Raw feature dictionary from the extractor.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(1, n_features)`` ready for model input.
        """
        aligned = np.zeros(len(self._feature_names), dtype=np.float64)
        for idx, name in enumerate(self._feature_names):
            aligned[idx] = features.get(name, 0.0)

        # Replace any NaN / Inf that may have slipped through
        finite_mask = np.isfinite(aligned)
        if not finite_mask.all():
            nan_count = int(np.sum(~finite_mask))
            logger.warning(
                "Replacing %d non-finite feature values with 0.0", nan_count
            )
            aligned[~finite_mask] = 0.0

        return aligned.reshape(1, -1)

    # ------------------------------------------------------------------
    # Internal: feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, window_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from raw sensor window."""
        try:
            return self._feature_extractor.extract_all(window_data)
        except Exception as exc:
            logger.error("Feature extraction failed: %s", exc)
            # Return zero features so downstream code doesn't crash
            return {name: 0.0 for name in self._feature_names}

    # ------------------------------------------------------------------
    # Internal: data quality
    # ------------------------------------------------------------------

    def _compute_data_quality(self, window_data: Dict[str, Any]) -> float:
        """Compute composite data-quality score from raw sensor arrays."""
        # Flatten all numeric arrays for the rule engine
        sensor_arrays: Dict[str, np.ndarray] = {}
        for key in ("accelerometer", "gyroscope", "heart_rate", "temperature"):
            section = window_data.get(key)
            if isinstance(section, dict):
                for sub_key, values in section.items():
                    if isinstance(values, np.ndarray) and values.size > 0:
                        sensor_arrays[f"{key}_{sub_key}"] = values.ravel()
        try:
            return self._rule_engine.get_data_quality_score(sensor_arrays)
        except Exception as exc:
            logger.warning("Data quality computation failed: %s", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Internal: input normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_input(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat-list sensor format to the canonical dict format.

        The synthetic data generator produces flat lists::

            {"accelerometer": [[ax, ay, az], ...], "temperature": [t1, t2, ...]}

        while the :class:`FeatureExtractor` expects::

            {"accelerometer": {"ax": np.array([...]), "ay": [...], "az": [...]},
             "temperature": {"stts22h_celsius": np.array([...]),
                             "lm35_celsius": np.array([...])}}

        This method detects which format was passed and converts if needed.
        """
        # Quick heuristic: if accelerometer is a list of lists (not a dict)
        # we need to convert.
        accel = data.get("accelerometer")
        if isinstance(accel, dict):
            return data  # already in canonical format

        logger.debug("Normalising sensor data from flat-list to dict format")
        result: Dict[str, Any] = {}
        result["metadata"] = data.get("metadata", {})

        # --- Accelerometer (list of [ax, ay, az]) ---
        accel_arr = np.asarray(accel, dtype=np.float64) if accel is not None else np.array([])
        if accel_arr.ndim == 2 and accel_arr.shape[1] >= 3:
            result["accelerometer"] = {
                "ax": accel_arr[:, 0],
                "ay": accel_arr[:, 1],
                "az": accel_arr[:, 2],
            }
        elif accel_arr.ndim == 1:
            result["accelerometer"] = {
                "ax": accel_arr,
                "ay": np.zeros_like(accel_arr),
                "az": np.zeros_like(accel_arr),
            }
        else:
            result["accelerometer"] = {"ax": np.array([]), "ay": np.array([]), "az": np.array([])}

        # --- Gyroscope (list of [gx, gy, gz]) ---
        gyro = data.get("gyroscope")
        gyro_arr = np.asarray(gyro, dtype=np.float64) if gyro is not None else np.array([])
        if gyro_arr.ndim == 2 and gyro_arr.shape[1] >= 3:
            result["gyroscope"] = {
                "gx": gyro_arr[:, 0],
                "gy": gyro_arr[:, 1],
                "gz": gyro_arr[:, 2],
            }
        elif gyro_arr.ndim == 1:
            result["gyroscope"] = {
                "gx": gyro_arr,
                "gy": np.zeros_like(gyro_arr),
                "gz": np.zeros_like(gyro_arr),
            }
        else:
            result["gyroscope"] = {"gx": np.array([]), "gy": np.array([]), "gz": np.array([])}

        # --- Heart rate (flat list -> bpm) ---
        hr = data.get("heart_rate")
        hr_arr = np.asarray(hr, dtype=np.float64) if hr is not None else np.array([])
        # SpO2 and PPG may be separate top-level keys or inside heart_rate dict
        spo2_arr = np.asarray(data.get("spo2", []), dtype=np.float64)
        ppg_arr = np.asarray(data.get("ppg", []), dtype=np.float64)
        result["heart_rate"] = {
            "bpm": hr_arr,
            "spo2": spo2_arr,
            "ppg_raw": ppg_arr,
        }

        # --- Temperature (flat list -> both stts22h and lm35) ---
        temp = data.get("temperature")
        temp_arr = np.asarray(temp, dtype=np.float64) if temp is not None else np.array([])
        result["temperature"] = {
            "stts22h_celsius": temp_arr.copy(),
            "lm35_celsius": temp_arr.copy(),
        }

        return result

    # ------------------------------------------------------------------
    # Internal: ensemble runner
    # ------------------------------------------------------------------

    def _run_ensemble(
        self,
        X: np.ndarray,
        window_data: Dict[str, Any],
        data_quality: float,
        threshold: Optional[float] = 0.5,
    ) -> Dict[str, Any]:
        """Run all models and combine into a unified prediction dict."""
        predictions: Dict[str, Any] = {}

        # --- Cardiac ---
        if self._cardiac_models:
            cardiac_probs = self._predict_cardiac(X)
            for i, cond in enumerate(_CARDIAC_CONDITIONS):
                prob = float(cardiac_probs[0, i])
                predictions[cond] = {
                    "detected": prob >= threshold if threshold is not None else False,
                    "probability": prob,
                    "confidence": data_quality,
                }
        else:
            for cond in _CARDIAC_CONDITIONS:
                predictions[cond] = {
                    "detected": False,
                    "probability": 0.0,
                    "confidence": 0.0,
                }

        # --- Respiratory ---
        if self._respiratory_model is not None:
            resp_probs = self._predict_respiratory(X)
            for i, cond in enumerate(_RESPIRATORY_CONDITIONS):
                prob = float(resp_probs[i])
                predictions[cond] = {
                    "detected": prob >= threshold if threshold is not None else False,
                    "probability": prob,
                    "confidence": data_quality,
                }
        else:
            for cond in _RESPIRATORY_CONDITIONS:
                predictions[cond] = {
                    "detected": False,
                    "probability": 0.0,
                    "confidence": 0.0,
                }

        # --- Activity ---
        if self._activity_models:
            act_probs = self._predict_activity(X)
            for i, cond in enumerate(_ACTIVITY_CONDITIONS):
                prob = float(act_probs[0, i])
                predictions[cond] = {
                    "detected": prob >= threshold if threshold is not None else False,
                    "probability": prob,
                    "confidence": data_quality,
                }
        else:
            for cond in _ACTIVITY_CONDITIONS:
                predictions[cond] = {
                    "detected": False,
                    "probability": 0.0,
                    "confidence": 0.0,
                }

        # --- Fever (rule engine) ---
        fever_detected, fever_confidence = self._detect_fever(window_data)
        predictions["fever"] = {
            "detected": fever_detected if threshold is not None else False,
            "probability": fever_confidence,
            "confidence": data_quality,
        }

        return predictions

    # ------------------------------------------------------------------
    # Internal: per-model prediction helpers
    # ------------------------------------------------------------------

    def _predict_cardiac(self, X: np.ndarray) -> np.ndarray:
        """Run the cardiac sub-models and return shape (1, 2) probs."""
        try:
            # Use the loaded booster objects directly
            probs = np.zeros((1, len(_CARDIAC_CONDITIONS)), dtype=np.float64)
            for i, cond in enumerate(_CARDIAC_CONDITIONS):
                booster = self._cardiac_models.get(cond)
                if booster is not None:
                    probs[0, i] = float(booster.predict(X)[0])
            return probs
        except Exception as exc:
            logger.error("Cardiac prediction failed: %s", exc)
            return np.zeros((1, len(_CARDIAC_CONDITIONS)), dtype=np.float64)

    def _predict_respiratory(self, X: np.ndarray) -> np.ndarray:
        """Run the respiratory sub-model and return shape (1,) probs."""
        try:
            return self._respiratory_model.predict(X)
        except Exception as exc:
            logger.error("Respiratory prediction failed: %s", exc)
            return np.zeros(len(_RESPIRATORY_CONDITIONS), dtype=np.float64)

    def _predict_activity(self, X: np.ndarray) -> np.ndarray:
        """Run the activity sub-models and return shape (1, 3) probs."""
        try:
            probs = np.zeros((1, len(_ACTIVITY_CONDITIONS)), dtype=np.float64)
            for i, cond in enumerate(_ACTIVITY_CONDITIONS):
                booster = self._activity_models.get(cond)
                if booster is not None:
                    probs[0, i] = float(booster.predict(X)[0])
            return probs
        except Exception as exc:
            logger.error("Activity prediction failed: %s", exc)
            return np.zeros((1, len(_ACTIVITY_CONDITIONS)), dtype=np.float64)

    def _detect_fever(
        self, window_data: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Detect fever using the rule engine on raw temperature data."""
        temp_data = window_data.get("temperature", {})
        temp_arrays: List[np.ndarray] = []
        for key in ("stts22h_celsius", "lm35_celsius"):
            arr = temp_data.get(key)
            if isinstance(arr, np.ndarray) and arr.size > 0:
                temp_arrays.append(arr.ravel())

        if not temp_arrays:
            return False, 0.0

        # Concatenate all available temperature channels
        combined = np.concatenate(temp_arrays)
        try:
            return self._rule_engine.detect_fever(combined)
        except Exception as exc:
            logger.warning("Fever detection failed: %s", exc)
            return False, 0.0

    # ------------------------------------------------------------------
    # Internal: file loaders
    # ------------------------------------------------------------------

    def _load_feature_names(self) -> List[str]:
        """Load the canonical feature-name list from ``feature_names.json``."""
        path = self._models_dir / "feature_names.json"
        if not path.exists():
            logger.warning(
                "feature_names.json not found at %s; using empty list", path
            )
            return []
        try:
            with path.open("r", encoding="utf-8") as fh:
                names = json.load(fh)
            if not isinstance(names, list):
                raise TypeError("Expected a JSON list")
            logger.info("Loaded %d canonical feature names", len(names))
            return list(names)
        except Exception as exc:
            logger.error("Failed to load feature names: %s", exc)
            return []

    def _load_cardiac(
        self, models_dir: Path
    ) -> Dict[str, Any]:
        """Load cardiac model boosters.

        Returns a dict mapping condition name → LightGBM Booster.
        """
        joblib_path = models_dir / "cardiac.joblib"
        if not joblib_path.exists():
            logger.warning("Cardiac model not found at %s", joblib_path)
            return {}

        try:
            # joblib file stores dict of {condition_name: Booster}
            data = joblib.load(joblib_path)
            if isinstance(data, dict):
                # Already a dict of boosters keyed by condition name
                logger.info(
                    "Loaded cardiac model with conditions: %s",
                    list(data.keys()),
                )
                return data
            else:
                logger.warning(
                    "Unexpected cardiac model format: %s", type(data).__name__
                )
                return {}
        except Exception as exc:
            logger.error("Failed to load cardiac model: %s", exc)
            return {}

    def _load_respiratory(
        self, models_dir: Path
    ) -> Any:
        """Load respiratory model booster.

        Returns a single LightGBM Booster object.
        """
        joblib_path = models_dir / "respiratory.joblib"
        if not joblib_path.exists():
            logger.warning("Respiratory model not found at %s", joblib_path)
            return None

        try:
            model = joblib.load(joblib_path)
            logger.info("Loaded respiratory model from %s", joblib_path)
            return model
        except Exception as exc:
            logger.error("Failed to load respiratory model: %s", exc)
            return None

    def _load_activity(
        self, models_dir: Path
    ) -> Dict[str, Any]:
        """Load activity model boosters.

        Returns a dict mapping condition name → LightGBM Booster.
        """
        joblib_path = models_dir / "activity.joblib"
        if not joblib_path.exists():
            logger.warning("Activity model not found at %s", joblib_path)
            return {}

        try:
            data = joblib.load(joblib_path)
            if isinstance(data, dict):
                logger.info(
                    "Loaded activity model with conditions: %s",
                    list(data.keys()),
                )
                return data
            else:
                logger.warning(
                    "Unexpected activity model format: %s", type(data).__name__
                )
                return {}
        except Exception as exc:
            logger.error("Failed to load activity model: %s", exc)
            return {}

    def _load_model_versions(self, models_dir: Path) -> Dict[str, str]:
        """Extract version strings from ``*.meta.json`` files."""
        versions: Dict[str, str] = {}
        for name in ("cardiac", "respiratory", "activity"):
            meta_path = models_dir / f"{name}.meta.json"
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    # Use train_history best_iteration as a simple version proxy
                    train_hist = meta.get("train_history", {})
                    if train_hist:
                        # Grab the first condition's best_iteration
                        first_cond = next(iter(train_hist.values()), {})
                        best_iter = first_cond.get("best_iteration", 1)
                        versions[name] = f"1.0.iter{best_iter}"
                    else:
                        versions[name] = "1.0"
                except Exception:
                    versions[name] = "1.0"
            else:
                versions[name] = "N/A"
        return versions

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> List[str]:
        """Canonical feature names used for alignment."""
        return self._feature_names.copy()

    @property
    def model_versions(self) -> Dict[str, str]:
        """Version metadata for each model group."""
        return self._model_versions.copy()

    @property
    def is_loaded(self) -> bool:
        """``True`` if at least one model was successfully loaded."""
        return bool(
            self._cardiac_models
            or self._respiratory_model is not None
            or self._activity_models
        )

    def describe(self) -> str:
        """Human-readable summary of the predictor state."""
        lines = [
            "Predictor",
            f"  Models dir       : {self._models_dir}",
            f"  Feature count    : {len(self._feature_names)}",
            f"  Cardiac models   : {list(self._cardiac_models.keys()) or '(not loaded)'}",
            f"  Respiratory model: {'loaded' if self._respiratory_model else '(not loaded)'}",
            f"  Activity models  : {list(self._activity_models.keys()) or '(not loaded)'}",
            f"  Versions         : {self._model_versions}",
        ]
        return "\n".join(lines)
