"""
Health Monitor - AI-powered health symptom detection from wearable sensor data.

Hugging Face Spaces Gradio App. Entry point for HF Spaces deployment.

Usage:
    python hf_space/app.py  (run locally)

Deployed on HF Spaces: auto-loaded from app.py
"""

import sys
import os
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is on sys.path so that src.* and
# data.* packages can be imported from the HF Spaces working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

import gradio as gr

import plotly.graph_objects as go

from src.visualization.visualizer_3d import (
    plot_accelerometer_3d,
    plot_gyroscope_3d,
    plot_vitals_3d,
    plot_feature_space_3d,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("health_monitor.app")

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (module scope for HF Spaces cold-start optimisation)
# ---------------------------------------------------------------------------
_feature_extractor = None
_cardiac_model = None
_respiratory_model = None
_activity_model = None
_rule_engine = None
_feature_names: Optional[List[str]] = None

_MODELS_DIR = str(Path(_PROJECT_ROOT) / "models")
_MODELS_LOADED = False
_MODELS_ERROR: Optional[str] = None


def _load_resources() -> None:
    """Load all ML models, feature extractor, and rule engine once."""
    global _feature_extractor, _cardiac_model, _respiratory_model
    global _activity_model, _rule_engine, _feature_names
    global _MODELS_LOADED, _MODELS_ERROR

    if _MODELS_LOADED:
        return

    try:
        from src.features.extractor import FeatureExtractor
        from src.models.cardiac_model import CardiacModel
        from src.models.respiratory_model import RespiratoryModel
        from src.models.activity_model import ActivityModel
        from src.models.rule_engine import RuleEngine

        _feature_extractor = FeatureExtractor(enable_logging=False)
        _rule_engine = RuleEngine()

        models_dir = Path(_MODELS_DIR)

        # Load feature name ordering
        feature_names_path = models_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as fh:
                _feature_names = json.load(fh)
        else:
            _feature_names = sorted(_feature_extractor.get_all_feature_names())

        # Load trained models (joblib)
        cardiac_path = models_dir / "cardiac"
        respiratory_path = models_dir / "respiratory"
        activity_path = models_dir / "activity"

        if cardiac_path.with_suffix(".joblib").exists():
            _cardiac_model = CardiacModel.load(str(cardiac_path), use_gpu=False)
            logger.info("Loaded cardiac model")
        else:
            logger.warning("Cardiac model not found at %s", cardiac_path)

        if respiratory_path.with_suffix(".joblib").exists():
            _respiratory_model = RespiratoryModel.load(str(respiratory_path), use_gpu=False)
            logger.info("Loaded respiratory model")
        else:
            logger.warning("Respiratory model not found at %s", respiratory_path)

        if activity_path.with_suffix(".joblib").exists():
            _activity_model = ActivityModel.load(str(activity_path), use_gpu=False)
            logger.info("Loaded activity model")
        else:
            logger.warning("Activity model not found at %s", activity_path)

        _MODELS_LOADED = True
        logger.info("All resources loaded successfully")

    except Exception as exc:
        _MODELS_ERROR = str(exc)
        logger.error("Failed to load resources: %s", exc)
        logger.error(traceback.format_exc())


def _resources_ready() -> bool:
    """Check whether the inference pipeline is ready."""
    return (
        _MODELS_LOADED
        and _feature_extractor is not None
        and _rule_engine is not None
    )


# ---------------------------------------------------------------------------
# Sensor data conversion: raw JSON -> FeatureExtractor format
# ---------------------------------------------------------------------------

def _convert_raw_to_extractor_format(
    sensor_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a single window of raw sensor JSON into FeatureExtractor format.

    Raw format (from sensor / synthetic generator):
        {
            "accelerometer": [[ax, ay, az], ...],
            "gyroscope": [[gx, gy, gz], ...],
            "heart_rate": [hr1, hr2, ...],
            "spo2": [spo2_1, spo2_2, ...],
            "temperature": [temp1, temp2, ...],
            "ppg": [ppg1, ppg2, ...]
        }

    FeatureExtractor format:
        {
            "accelerometer": {"ax": arr, "ay": arr, "az": arr},
            "gyroscope": {"gx": arr, "gy": arr, "gz": arr},
            "heart_rate": {"bpm": arr, "spo2": arr, "ppg_raw": arr},
            "temperature": {"stts22h_celsius": arr, "lm35_celsius": arr},
            "metadata": {"sampling_rate": ..., "hr_sampling_rate": ..., ...}
        }
    """
    result: Dict[str, Any] = {}

    # --- Accelerometer ---
    accel_raw = sensor_data.get("accelerometer", [])
    if isinstance(accel_raw, list) and len(accel_raw) > 0:
        accel_arr = np.array(accel_raw, dtype=np.float64)
        if accel_arr.ndim == 2 and accel_arr.shape[1] >= 3:
            result["accelerometer"] = {
                "ax": accel_arr[:, 0],
                "ay": accel_arr[:, 1],
                "az": accel_arr[:, 2],
            }
        else:
            result["accelerometer"] = {"ax": np.array([]), "ay": np.array([]), "az": np.array([])}
    else:
        result["accelerometer"] = {"ax": np.array([]), "ay": np.array([]), "az": np.array([])}

    # --- Gyroscope ---
    gyro_raw = sensor_data.get("gyroscope", [])
    if isinstance(gyro_raw, list) and len(gyro_raw) > 0:
        gyro_arr = np.array(gyro_raw, dtype=np.float64)
        if gyro_arr.ndim == 2 and gyro_arr.shape[1] >= 3:
            result["gyroscope"] = {
                "gx": gyro_arr[:, 0],
                "gy": gyro_arr[:, 1],
                "gz": gyro_arr[:, 2],
            }
        else:
            result["gyroscope"] = {"gx": np.array([]), "gy": np.array([]), "gz": np.array([])}
    else:
        result["gyroscope"] = {"gx": np.array([]), "gy": np.array([]), "gz": np.array([])}

    # --- Heart rate / SpO2 / PPG ---
    hr_raw = sensor_data.get("heart_rate", [])
    spo2_raw = sensor_data.get("spo2", [])
    ppg_raw = sensor_data.get("ppg", [])

    result["heart_rate"] = {
        "bpm": np.array(hr_raw, dtype=np.float64) if hr_raw else np.array([]),
        "spo2": np.array(spo2_raw, dtype=np.float64) if spo2_raw else np.array([]),
        "ppg_raw": np.array(ppg_raw, dtype=np.float64) if ppg_raw else np.array([]),
    }

    # --- Temperature ---
    temp_raw = sensor_data.get("temperature", [])
    temp_arr = np.array(temp_raw, dtype=np.float64) if temp_raw else np.array([])
    # The synthetic generator produces a single temperature channel.
    # Duplicate it for both STTS22H and LM35 so the feature extractor works.
    result["temperature"] = {
        "stts22h_celsius": temp_arr,
        "lm35_celsius": temp_arr,
    }

    # --- Metadata ---
    n_accel = len(accel_raw) if isinstance(accel_raw, list) else 0
    n_hr = len(hr_raw) if isinstance(hr_raw, list) else 0
    n_ppg = len(ppg_raw) if isinstance(ppg_raw, list) else 0

    result["metadata"] = {
        "sampling_rate": max(n_accel / 30.0, 1.0) if n_accel > 0 else 30.0,
        "hr_sampling_rate": max(n_hr / 30.0, 1.0) if n_hr > 0 else 25.0,
        "ppg_sampling_rate": max(n_ppg / 30.0, 1.0) if n_ppg > 0 else 25.0,
        "activity_state": "resting",
    }

    return result


def _build_feature_vector(window_data: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract features and return an ordered numpy vector matching the
    trained model's expected feature order.

    Returns None if extraction fails.
    """
    if _feature_extractor is None or _feature_names is None:
        return None

    try:
        features_dict = _feature_extractor.extract_all(window_data)
    except Exception as exc:
        logger.warning("Feature extraction failed: %s", exc)
        return None

    # Order features according to the trained model's expected ordering
    vec = np.array(
        [features_dict.get(name, 0.0) for name in _feature_names],
        dtype=np.float64,
    )

    # Replace any remaining NaN / Inf with 0.0
    bad_mask = ~np.isfinite(vec)
    if np.any(bad_mask):
        vec[bad_mask] = 0.0

    return vec


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Canonical condition names (order matches model output vectors)
CONDITION_LABELS: List[str] = [
    "tachycardia",
    "irregular_rhythm",
    "low_spo2",
    "fever",
    "fall_detected",
    "sleep_problem",
    "fatigue",
]

# Human-friendly display names
CONDITION_DISPLAY: Dict[str, str] = {
    "tachycardia": "Tachycardia (High Heart Rate)",
    "irregular_rhythm": "Irregular Heart Rhythm",
    "low_spo2": "Low Blood Oxygen (SpO2)",
    "fever": "Fever",
    "fall_detected": "Fall Detected",
    "sleep_problem": "Sleep Problem",
    "fatigue": "Fatigue",
}

# Severity colour mapping
_SEVERITY_COLORS = {
    "normal": "#2e7d32",      # green
    "mild": "#f9a825",        # amber
    "moderate": "#ef6c00",    # orange
    "severe": "#c62828",      # red
    "critical": "#b71c1c",    # dark red
}


def _run_inference(
    sensor_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the full inference pipeline on a single window of sensor data.

    Returns a structured result dict.
    """
    if not _resources_ready():
        return {
            "status": "error",
            "error": (
                "Models are not loaded. Please ensure trained model files "
                f"exist in {_MODELS_DIR}."
            ),
            "predictions": {},
        }

    # 1. Convert raw data
    window_data = _convert_raw_to_extractor_format(sensor_data)

    # 2. Extract feature vector
    feature_vec = _build_feature_vector(window_data)
    if feature_vec is None:
        return {
            "status": "error",
            "error": "Feature extraction failed. Check that sensor data is valid.",
            "predictions": {},
        }

    X = feature_vec.reshape(1, -1)

    # 3. Data quality score
    quality_score = 0.0
    try:
        # Convert to numpy arrays for quality check
        quality_data = {}
        for key in ("heart_rate", "spo2", "temperature"):
            arr = window_data.get(key, {})
            if isinstance(arr, dict):
                for sub_key, sub_arr in arr.items():
                    if isinstance(sub_arr, np.ndarray) and sub_arr.size > 0:
                        quality_data[f"{key}_{sub_key}"] = sub_arr
        if quality_data:
            quality_score = _rule_engine.get_data_quality_score(quality_data)
    except Exception:
        quality_score = 0.5

    # 4. ML model predictions
    predictions: Dict[str, float] = {}
    confidence_scores: Dict[str, float] = {}

    try:
        if _cardiac_model is not None:
            cardiac_probs = _cardiac_model.predict(X)
            # cardiac model outputs: [tachycardia, irregular_rhythm]
            predictions["tachycardia"] = float(cardiac_probs[0, 0])
            predictions["irregular_rhythm"] = float(cardiac_probs[0, 1])
    except Exception as exc:
        logger.warning("Cardiac prediction failed: %s", exc)
        predictions["tachycardia"] = 0.0
        predictions["irregular_rhythm"] = 0.0

    try:
        if _respiratory_model is not None:
            resp_probs = _respiratory_model.predict(X)
            predictions["low_spo2"] = float(resp_probs[0])
    except Exception as exc:
        logger.warning("Respiratory prediction failed: %s", exc)
        predictions["low_spo2"] = 0.0

    try:
        if _activity_model is not None:
            activity_probs = _activity_model.predict(X)
            # activity model outputs: [fall_detected, sleep_problem, fatigue]
            predictions["fall_detected"] = float(activity_probs[0, 0])
            predictions["sleep_problem"] = float(activity_probs[0, 1])
            predictions["fatigue"] = float(activity_probs[0, 2])
    except Exception as exc:
        logger.warning("Activity prediction failed: %s", exc)
        predictions["fall_detected"] = 0.0
        predictions["sleep_problem"] = 0.0
        predictions["fatigue"] = 0.0

    # 5. Rule-based fever detection (uses raw temperature, not ML)
    try:
        temp_arr = window_data.get("temperature", {}).get("stts22h_celsius", np.array([]))
        if temp_arr.size > 0:
            fever_detected, fever_confidence = _rule_engine.detect_fever(temp_arr)
            # Blend rule-based fever with ML (if ML gives anything)
            ml_fever = predictions.get("fever", 0.0)
            predictions["fever"] = max(ml_fever, fever_confidence)
        else:
            predictions.setdefault("fever", 0.0)
    except Exception:
        predictions.setdefault("fever", 0.0)

    # 6. Determine overall status and per-condition severity
    DETECTION_THRESHOLD = 0.5
    condition_results: Dict[str, Dict[str, Any]] = {}
    any_critical = False
    any_warning = False

    for cond in CONDITION_LABELS:
        prob = predictions.get(cond, 0.0)
        detected = prob >= DETECTION_THRESHOLD

        if prob >= 0.8:
            severity = "critical"
            any_critical = True
        elif prob >= 0.6:
            severity = "severe"
            any_critical = True
        elif detected:
            severity = "moderate"
            any_warning = True
        elif prob >= 0.3:
            severity = "mild"
            any_warning = True
        else:
            severity = "normal"

        condition_results[cond] = {
            "name": CONDITION_DISPLAY.get(cond, cond),
            "probability": round(prob, 4),
            "detected": detected,
            "severity": severity,
            "color": _SEVERITY_COLORS.get(severity, "#757575"),
        }

    # Overall status
    if any_critical:
        overall_status = "critical"
    elif any_warning:
        overall_status = "warning"
    else:
        overall_status = "normal"

    return {
        "status": "success",
        "overall_status": overall_status,
        "data_quality": round(quality_score, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conditions": condition_results,
        "raw_probabilities": {k: round(v, 4) for k, v in predictions.items()},
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_result_markdown(result: Dict[str, Any]) -> str:
    """Convert an inference result dict to a readable Markdown string."""
    if result.get("status") == "error":
        return (
            f"### Error\n\n"
            f"{result.get('error', 'Unknown error')}\n\n"
            f"*No analysis could be performed.*"
        )

    overall = result.get("overall_status", "normal")
    quality = result.get("data_quality", 0.0)
    ts = result.get("timestamp", "N/A")
    conditions = result.get("conditions", {})

    # Overall status banner
    if overall == "critical":
        status_line = "**OVERALL STATUS: CRITICAL -- Seek immediate medical attention**"
    elif overall == "warning":
        status_line = "**OVERALL STATUS: WARNING -- Abnormal readings detected**"
    else:
        status_line = "**OVERALL STATUS: NORMAL -- No significant abnormalities detected**"

    lines = [
        f"## Analysis Results",
        "",
        status_line,
        "",
        f"- **Data Quality**: {quality:.1%}",
        f"- **Analysis Time**: {ts}",
        "",
        "---",
        "",
        "### Condition Probabilities",
        "",
        "| Condition | Probability | Status | Severity |",
        "|-----------|-------------|--------|----------|",
    ]

    for cond_key in CONDITION_LABELS:
        info = conditions.get(cond_key, {})
        prob = info.get("probability", 0.0)
        detected = info.get("detected", False)
        severity = info.get("severity", "normal")
        name = info.get("name", cond_key)

        if detected:
            status_str = "**DETECTED**"
        else:
            status_str = "Not detected"

        # Color code severity
        if severity in ("critical", "severe"):
            sev_display = f"**{severity.upper()}**"
        elif severity == "moderate":
            sev_display = f"*{severity}*"
        else:
            sev_display = severity

        lines.append(
            f"| {name} | {prob:.1%} | {status_str} | {sev_display} |"
        )

    # Add any detected condition alerts
    detected_conds = [
        info["name"]
        for info in conditions.values()
        if info.get("detected")
    ]
    if detected_conds:
        lines.extend([
            "",
            "---",
            "",
            "### Alerts",
            "",
        ])
        for name in detected_conds:
            lines.append(f"- **{name}**")
        lines.append("")
    else:
        lines.extend([
            "",
            "---",
            "",
            "*No conditions detected above the threshold.*",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*This analysis is for informational purposes only. "
        "It is not a medical diagnosis. Consult a healthcare professional.*",
    ])

    return "\n".join(lines)


def _format_result_json(result: Dict[str, Any]) -> str:
    """Return a pretty-printed JSON string of the result."""
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def _generate_demo_data(condition: str, severity: str) -> str:
    """Generate synthetic sensor data JSON for a demo scenario."""
    try:
        from data.synthetic.generator import SyntheticDataGenerator

        gen = SyntheticDataGenerator(seed=42)

        condition_map = {
            "Normal": None,
            "Tachycardia": "tachycardia",
            "Low SpO2": "low_spo2",
            "Fever": "fever",
            "Fall": "fall_detected",
            "Fatigue": "fatigue",
            "Sleep Problem": "sleep_problem",
            "Irregular Rhythm": "irregular_rhythm",
        }

        cond_key = condition_map.get(condition)
        if cond_key is None:
            session = gen.generate_normal_session(duration_sec=30)
        else:
            session = gen.generate_condition_session(
                cond_key,
                duration_sec=30,
                severity=severity.lower(),
            )

        # Return first window's sensor data as formatted JSON
        windows = session.get("windows", [])
        if not windows:
            return json.dumps({"error": "No windows generated"}, indent=2)

        sensor_data = windows[0].get("sensor_data", {})
        return json.dumps(sensor_data, indent=2, default=str)

    except Exception as exc:
        return json.dumps({"error": f"Demo generation failed: {exc}"}, indent=2)


# ---------------------------------------------------------------------------
# Monitor dashboard state (simple in-memory log for demo)
# ---------------------------------------------------------------------------
_alert_history: List[Dict[str, Any]] = []


def _update_alert_history(result: Dict[str, Any]) -> str:
    """Append to the in-memory alert history and return formatted log."""
    ts = result.get("timestamp", datetime.now(timezone.utc).isoformat())
    overall = result.get("overall_status", "normal")
    conditions = result.get("conditions", {})

    detected = [
        info["name"]
        for info in conditions.values()
        if info.get("detected")
    ]

    entry = {
        "time": ts,
        "status": overall,
        "alerts": detected,
    }
    _alert_history.append(entry)

    # Keep last 50 entries
    if len(_alert_history) > 50:
        _alert_history.pop(0)

    # Format as text
    lines = [
        "=" * 60,
        "  HEALTH MONITOR -- ALERT LOG",
        "=" * 60,
        "",
    ]

    for entry in reversed(_alert_history):
        status = entry["status"].upper()
        time_str = entry["time"]
        alerts = entry["alerts"]
        if alerts:
            lines.append(f"[{time_str}]  {status}  --  {', '.join(alerts)}")
        else:
            lines.append(f"[{time_str}]  {status}  --  No alerts")

    lines.extend([
        "",
        "=" * 60,
        f"  Total entries: {len(_alert_history)}",
        "=" * 60,
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio callback functions
# ---------------------------------------------------------------------------

def analyze_sensor_data(json_input: str, file_upload: Optional[Any] = None) -> Tuple[str, str]:
    """Analyze raw sensor JSON from the Live Sensor Input tab.

    Returns (markdown_result, json_result).
    """
    try:
        # Use file upload if provided and text input is empty
        if file_upload is not None and (not json_input or not json_input.strip()):
            if hasattr(file_upload, "name"):
                with open(file_upload.name, "r", encoding="utf-8") as fh:
                    json_input = fh.read()
            elif isinstance(file_upload, str):
                json_input = file_upload

        if not json_input or not json_input.strip():
            return (
                "### No Input\n\nPlease paste sensor data JSON or upload a JSON file.",
                "{}",
            )

        data = json.loads(json_input)

        # If the input is a full session (with "windows" key), use first window
        if isinstance(data, dict) and "windows" in data:
            data = data["windows"][0].get("sensor_data", data)

        result = _run_inference(data)
        return _format_result_markdown(result), _format_result_json(result)

    except json.JSONDecodeError as exc:
        error_msg = f"Invalid JSON input: {exc}"
        return f"### Error\n\n{error_msg}", json.dumps({"error": error_msg}, indent=2)
    except Exception as exc:
        error_msg = f"Analysis failed: {exc}"
        logger.error(traceback.format_exc())
        return f"### Error\n\n{error_msg}", json.dumps({"error": error_msg}, indent=2)


def generate_and_analyze(condition: str, severity: str) -> Tuple[str, str, str]:
    """Generate demo data and run inference.

    Returns (sensor_json, markdown_result, json_result).
    """
    try:
        sensor_json = _generate_demo_data(condition, severity)

        # Parse and analyse
        data = json.loads(sensor_json)
        result = _run_inference(data)

        return (
            sensor_json,
            _format_result_markdown(result),
            _format_result_json(result),
        )
    except Exception as exc:
        error_msg = f"Demo generation/analysis failed: {exc}"
        logger.error(traceback.format_exc())
        empty_json = json.dumps({"error": error_msg}, indent=2)
        return empty_json, f"### Error\n\n{error_msg}", empty_json


def monitor_analyze(json_input: str) -> Tuple[str, str]:
    """Analyse sensor data and update the monitor dashboard.

    Returns (dashboard_text, alert_log_text).
    """
    try:
        if not json_input or not json_input.strip():
            return "No data to analyse.", _format_alert_log()

        data = json.loads(json_input)
        if isinstance(data, dict) and "windows" in data:
            data = data["windows"][0].get("sensor_data", data)

        result = _run_inference(data)
        _update_alert_history(result)

        dashboard = _format_dashboard(result)
        return dashboard, _format_alert_log()

    except Exception as exc:
        return f"Error: {exc}", _format_alert_log()


def _format_dashboard(result: Dict[str, Any]) -> str:
    """Format a simple text-based dashboard."""
    if result.get("status") == "error":
        return f"ERROR: {result.get('error', 'Unknown')}"

    overall = result.get("overall_status", "normal").upper()
    quality = result.get("data_quality", 0.0)
    conditions = result.get("conditions", {})

    lines = [
        "+" + "-" * 58 + "+",
        "|  HEALTH MONITOR -- LIVE DASHBOARD" + " " * 25 + "|",
        "+" + "-" * 58 + "+",
        "",
        f"  Overall Status : {overall}",
        f"  Data Quality   : {quality:.1%}",
        f"  Timestamp      : {result.get('timestamp', 'N/A')}",
        "",
        "  Condition Status:",
    ]

    for cond_key in CONDITION_LABELS:
        info = conditions.get(cond_key, {})
        prob = info.get("probability", 0.0)
        severity = info.get("severity", "normal")
        name = info.get("name", cond_key)

        # Build a simple bar
        bar_len = int(prob * 20)
        bar = "#" * bar_len + "." * (20 - bar_len)

        lines.append(f"    {name:<30s} [{bar}] {prob:.1%}  ({severity})")

    lines.extend([
        "",
        "+" + "-" * 58 + "+",
    ])

    return "\n".join(lines)


def _format_alert_log() -> str:
    """Format the in-memory alert history."""
    if not _alert_history:
        return "No alerts recorded yet."

    lines = [
        "=" * 60,
        "  HEALTH MONITOR -- ALERT LOG",
        "=" * 60,
        "",
    ]

    for entry in reversed(_alert_history):
        status = entry["status"].upper()
        time_str = entry["time"]
        alerts = entry["alerts"]
        if alerts:
            lines.append(f"  [{time_str}]  {status}  --  {', '.join(alerts)}")
        else:
            lines.append(f"  [{time_str}]  {status}  --  No alerts")

    lines.extend([
        "",
        "=" * 60,
        f"  Total entries: {len(_alert_history)}",
        "=" * 60,
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3D visualization callbacks
# ---------------------------------------------------------------------------


def _generate_3d_visualization(
    viz_type: str,
    condition: str,
    severity: int,
) -> Tuple:
    """Generate 3D Plotly figures for the selected visualisation type.

    Returns a tuple of 6 outputs:
      (primary_fig, accel_fig, gyro_fig, vitals_fig,
       combined_row_update, combined_row2_update)

    - For single viz types, ``primary_fig`` contains the plot; the 3 sensor
      plots are empty placeholders; combined rows are hidden.
    - For ``"All Combined"``, ``primary_fig`` is hidden and the 3 sensor
      plots are returned separately (each fullscreenable). Combined rows
      are visible.
    """
    try:
        from data.synthetic.generator import SyntheticDataGenerator

        severity_map = {1: "mild", 2: "moderate", 3: "severe"}
        severity_str = severity_map.get(int(severity), "moderate")

        gen = SyntheticDataGenerator(seed=42)

        # Generate a single window of sensor data
        condition_map: Dict[str, str] = {
            "Normal": "normal",
            "Tachycardia": "tachycardia",
            "Low SpO2": "low_spo2",
            "Fever": "fever",
            "Fall": "fall_detected",
            "Fatigue": "fatigue",
            "Sleep Problem": "sleep_problem",
            "Irregular Rhythm": "irregular_rhythm",
        }

        cond_key = condition_map.get(condition, "normal")
        if cond_key == "normal":
            session = gen.generate_normal_session(duration_sec=30)
        else:
            session = gen.generate_condition_session(
                cond_key, duration_sec=30, severity=severity_str,
            )

        windows = session.get("windows", [])
        if not windows:
            empty = _empty_figure("No windows generated")
            hide = gr.update(visible=False)
            return empty, empty, empty, empty, hide, hide

        sensor_data = windows[0].get("sensor_data", {})

        # Run inference for status coloring
        predictions: Optional[Dict[str, Any]] = None
        try:
            result = _run_inference(sensor_data)
            if result.get("status") == "success":
                predictions = result
        except Exception:
            pass

        empty = _empty_figure()
        show = gr.update(visible=True)
        hide = gr.update(visible=False)

        # --- Branch on visualisation type ---
        if viz_type == "Accelerometer 3D":
            fig = plot_accelerometer_3d(sensor_data, height=700)
            return fig, empty, empty, empty, hide, hide

        elif viz_type == "Gyroscope 3D":
            fig = plot_gyroscope_3d(sensor_data, height=700)
            return fig, empty, empty, empty, hide, hide

        elif viz_type == "Vital Signs 3D":
            fig = plot_vitals_3d(sensor_data, predictions, height=700)
            return fig, empty, empty, empty, hide, hide

        elif viz_type == "Feature Space (PCA)":
            if _feature_extractor is None:
                fig = _empty_figure("Feature extractor not loaded yet")
                return fig, empty, empty, empty, hide, hide
            fig = plot_feature_space_3d(
                gen,
                condition=cond_key,
                severity=severity_str,
                n_windows=30,
                feature_extractor=_feature_extractor,
                height=700,
            )
            return fig, empty, empty, empty, hide, hide

        elif viz_type == "All Combined":
            accel_fig = plot_accelerometer_3d(sensor_data, height=500)
            gyro_fig = plot_gyroscope_3d(sensor_data, height=500)
            vitals_fig = plot_vitals_3d(sensor_data, predictions, height=500)
            hide_primary = gr.update(visible=False)  # hide the empty primary plot
            return hide_primary, accel_fig, gyro_fig, vitals_fig, show, show

        else:
            fig = _empty_figure(f"Unknown type: {viz_type}")
            return fig, empty, empty, empty, hide, hide

    except Exception as exc:
        logger.error("3D visualisation failed: %s", exc)
        logger.error(traceback.format_exc())
        err = _empty_figure(f"Visualisation error: {exc}")
        hide = gr.update(visible=False)
        return err, err, err, err, hide, hide


def _empty_figure(message: str = "No data") -> go.Figure:
    """Return an empty Plotly figure with an error message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16, color="#757575"),
    )
    fig.update_layout(
        title=dict(text="No Data", x=0.5, xanchor="center"),
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
.health-app { max-width: 1000px; margin: auto; }
.status-normal { color: #2e7d32; font-weight: bold; }
.status-warning { color: #ef6c00; font-weight: bold; }
.status-critical { color: #c62828; font-weight: bold; }
.disclaimer-box {
    background-color: #fff3e0;
    border: 2px solid #ef6c00;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}
.model-status { font-size: 0.9em; color: #666; }
"""


def _build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    with gr.Blocks(
        title="Health Monitor -- AI Symptom Detection",
        css=_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="gray",
        ),
    ) as app:
        gr.Markdown(
            "# Health Monitor\n"
            "AI-powered health symptom detection from wearable sensor data\n\n"
            "---"
        )

        # ------------------------------------------------------------------
        # Tab 1: Live Sensor Input
        # ------------------------------------------------------------------
        with gr.Tab("Live Sensor Input"):
            gr.Markdown(
                "### Analyse Real Sensor Data\n"
                "Paste your wearable sensor data in JSON format, or upload a JSON file."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    sensor_input = gr.Textbox(
                        label="Sensor Data (JSON)",
                        placeholder=(
                            '{\n'
                            '  "accelerometer": [[0.1, 9.8, 0.2], ...],\n'
                            '  "gyroscope": [[0.01, -0.02, 0.005], ...],\n'
                            '  "heart_rate": [72, 73, 71, ...],\n'
                            '  "spo2": [97, 98, 97, ...],\n'
                            '  "temperature": [36.5, 36.6, ...],\n'
                            '  "ppg": [0.5, 0.8, 0.3, ...]\n'
                            '}'
                        ),
                        lines=12,
                        max_lines=30,
                    )
                    file_upload = gr.File(
                        label="Or Upload JSON File",
                        file_types=[".json"],
                        type="filepath",
                    )
                    analyze_btn = gr.Button("Analyse", variant="primary")

                with gr.Column(scale=5):
                    result_markdown = gr.Markdown(
                        label="Analysis Results",
                        value="*Paste sensor data and click Analyse.*",
                    )
                    result_json = gr.JSON(label="Raw Result")

            analyze_btn.click(
                fn=analyze_sensor_data,
                inputs=[sensor_input, file_upload],
                outputs=[result_markdown, result_json],
            )

        # ------------------------------------------------------------------
        # Tab 2: Simulated Demo
        # ------------------------------------------------------------------
        with gr.Tab("Simulated Demo"):
            gr.Markdown(
                "### Generate and Analyse Synthetic Sensor Data\n"
                "Select a health condition and severity to generate realistic "
                "sensor data and see the analysis results."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    condition_dropdown = gr.Dropdown(
                        label="Condition",
                        choices=[
                            "Normal",
                            "Tachycardia",
                            "Low SpO2",
                            "Fever",
                            "Fall",
                            "Fatigue",
                            "Sleep Problem",
                            "Irregular Rhythm",
                        ],
                        value="Normal",
                    )
                    severity_slider = gr.Slider(
                        label="Severity",
                        minimum=1,
                        maximum=3,
                        step=1,
                        value=2,
                        info="1 = Mild, 2 = Moderate, 3 = Severe",
                    )
                    generate_btn = gr.Button("Generate and Analyse", variant="primary")

                with gr.Column(scale=5):
                    demo_json_output = gr.Code(
                        label="Generated Sensor Data (JSON)",
                        language="json",
                    )

            with gr.Row():
                demo_result_markdown = gr.Markdown(
                    label="Analysis Results",
                    value="*Select a condition and click Generate.*",
                )

            with gr.Row():
                demo_result_json = gr.JSON(label="Raw Result")

            # Map slider int to severity string
            def _severity_from_slider(val):
                return {1: "mild", 2: "moderate", 3: "severe"}.get(int(val), "moderate")

            def _run_demo(condition, severity_val):
                severity_str = _severity_from_slider(severity_val)
                return generate_and_analyze(condition, severity_str)

            generate_btn.click(
                fn=_run_demo,
                inputs=[condition_dropdown, severity_slider],
                outputs=[demo_json_output, demo_result_markdown, demo_result_json],
            )

        # ------------------------------------------------------------------
        # Tab 3: About / Disclaimer
        # ------------------------------------------------------------------
        with gr.Tab("About / Disclaimer"):
            gr.Markdown(
                "---\n"
                "## IMPORTANT MEDICAL DISCLAIMER\n\n"
                "**This application is NOT a medical device and does NOT provide "
                "medical diagnoses. The outputs are for informational and "
                "educational purposes ONLY.**\n\n"
                "**Always consult a qualified healthcare professional for medical "
                "advice, diagnosis, or treatment. Never disregard professional "
                "medical advice or delay seeking it because of information "
                "provided by this application.**\n\n"
                "**In case of a medical emergency, call your local emergency "
                "services immediately.**\n\n"
                "---"
            )

            gr.Markdown(
                "### About Health Monitor\n\n"
                "Health Monitor is an AI-powered system that analyses data from "
                "wearable sensor devices (accelerometer, gyroscope, heart rate, "
                "SpO2, temperature, PPG) to detect potential health anomalies.\n\n"
                "The system uses a combination of:\n"
                "- **LightGBM machine learning models** trained on synthetic "
                "wearable sensor data\n"
                "- **Rule-based clinical thresholds** for conditions like fever\n"
                "- **Feature engineering** with 162 features across 7 categories "
                "(motion, heart rate, HRV, SpO2, temperature, cross-sensor, "
                "frequency domain)\n\n"
                "### Detectable Conditions\n\n"
                "| Condition | Detection Method | Clinical Threshold |\n"
                "|-----------|------------------|--------------------|\n"
                "| Tachycardia | ML (Cardiac Model) | HR > 100 BPM |\n"
                "| Irregular Rhythm | ML (Cardiac Model) | RR variability |\n"
                "| Low Blood Oxygen | ML (Respiratory Model) | SpO2 < 95% |\n"
                "| Fever | Rule-based | Temp >= 38.0 C |\n"
                "| Fall Detection | ML (Activity Model) + Rules | Acceleration spike |\n"
                "| Sleep Problem | ML (Activity Model) | Motion + HR patterns |\n"
                "| Fatigue | ML (Activity Model) | HRV + cross-sensor |\n\n"
                "### Sensor Information\n\n"
                "The system is designed for the following sensor configuration:\n"
                "- **Accelerometer**: MPU6500, 3-axis, 50 Hz\n"
                "- **Gyroscope**: MPU6500, 3-axis, 50 Hz\n"
                "- **Heart Rate / SpO2 / PPG**: MAX30102, 25 Hz\n"
                "- **Temperature**: STTS22H (digital) + LM35 (analog), 1 Hz\n\n"
                "### Model Performance\n\n"
                "Models are trained on synthetic data. Performance on real-world "
                "sensor data may differ. Current model metrics (on test set):\n"
                "- **Cardiac model**: AUC varies by condition\n"
                "- **Respiratory model**: AUC varies by condition\n"
                "- **Activity model**: AUC varies by condition\n\n"
                "Model files: `cardiac.joblib`, `respiratory.joblib`, "
                "`activity.joblib`\n\n"
                "### Technical Details\n\n"
                "- **Framework**: Python 3.10+, Gradio 4.x\n"
                "- **ML**: LightGBM 4.x, scikit-learn, NumPy, SciPy\n"
                "- **Features**: 162 engineered features\n"
                "- **Inference**: ~50ms per window on CPU\n"
                "- **Deployment**: Hugging Face Spaces (free CPU tier)"
            )

        # ------------------------------------------------------------------
        # Tab 4: Monitor Dashboard
        # ------------------------------------------------------------------
        with gr.Tab("Monitor Dashboard"):
            gr.Markdown(
                "### Real-Time Monitoring Mode\n"
                "Analyse sensor data to see a live dashboard view and "
                "alert history."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    monitor_input = gr.Textbox(
                        label="Sensor Data (JSON)",
                        placeholder="Paste sensor data JSON here...",
                        lines=8,
                    )
                    monitor_btn = gr.Button("Analyse and Update", variant="primary")

                with gr.Column(scale=5):
                    dashboard_output = gr.Textbox(
                        label="Live Dashboard",
                        lines=20,
                        interactive=False,
                    )

            alert_log_output = gr.Textbox(
                label="Alert History Log",
                lines=15,
                interactive=False,
            )

            monitor_btn.click(
                fn=monitor_analyze,
                inputs=[monitor_input],
                outputs=[dashboard_output, alert_log_output],
            )

        # ------------------------------------------------------------------
        # Tab 5: 3D Visualization
        # ------------------------------------------------------------------
        with gr.Tab("3D Visualization"):
            gr.Markdown(
                "### Interactive 3D Sensor & Model Visualisation\n"
                "Explore wearable sensor data in 3D space. Each plot has its "
                "own **fullscreen button** ("
                "\U000026F6 in the top-right toolbar) so you can "
                "expand any graph for detailed inspection."
            )

            with gr.Row():
                # -- Controls --
                with gr.Column(scale=2):
                    viz_type_dropdown = gr.Dropdown(
                        label="Visualisation Type",
                        choices=[
                            "Accelerometer 3D",
                            "Gyroscope 3D",
                            "Vital Signs 3D",
                            "Feature Space (PCA)",
                            "All Combined",
                        ],
                        value="Accelerometer 3D",
                    )
                    viz_condition_dropdown = gr.Dropdown(
                        label="Condition",
                        choices=[
                            "Normal",
                            "Tachycardia",
                            "Low SpO2",
                            "Fever",
                            "Fall",
                            "Fatigue",
                            "Sleep Problem",
                            "Irregular Rhythm",
                        ],
                        value="Normal",
                    )
                    viz_severity_slider = gr.Slider(
                        label="Severity",
                        minimum=1,
                        maximum=3,
                        step=1,
                        value=2,
                        info="1 = Mild, 2 = Moderate, 3 = Severe",
                    )
                    viz_generate_btn = gr.Button(
                        "Generate 3D Visualisation", variant="primary"
                    )

                    gr.Markdown(
                        "---\n"
                        "**Tips:**\n"
                        "- Use **fullscreen** ("
                        "\U000026F6 icon) on any plot to view it clearly.\n"
                        "- **Rotate** by dragging, **zoom** with scroll.\n"
                        "- **Hover** for exact values.\n"
                        "- *All Combined* shows 3 separate plots — each "
                        "can be fullscreened individually."
                    )

                # -- Plot area --
                with gr.Column(scale=5):
                    # Single-type primary plot (700px tall)
                    viz_primary_plot = gr.Plot(
                        label="3D Visualization",
                        value=_empty_figure(
                            "Select a visualisation type and click Generate"
                        ),
                        show_label=True,
                    )

                    # Combined plots row (hidden until "All Combined" is selected)
                    with gr.Row(visible=False) as viz_combined_row:
                        with gr.Column():
                            viz_accel_plot = gr.Plot(
                                label="Accelerometer 3D",
                                value=_empty_figure(),
                                show_label=True,
                            )
                        with gr.Column():
                            viz_gyro_plot = gr.Plot(
                                label="Gyroscope 3D",
                                value=_empty_figure(),
                                show_label=True,
                            )
                    with gr.Row(visible=False) as viz_combined_row2:
                        with gr.Column():
                            viz_vitals_plot = gr.Plot(
                                label="Vital Signs 3D",
                                value=_empty_figure(),
                                show_label=True,
                            )

            # Wire up the generate button
            viz_generate_btn.click(
                fn=_generate_3d_visualization,
                inputs=[viz_type_dropdown, viz_condition_dropdown, viz_severity_slider],
                outputs=[
                    viz_primary_plot,
                    viz_accel_plot,
                    viz_gyro_plot,
                    viz_vitals_plot,
                    viz_combined_row,
                    viz_combined_row2,
                ],
            )

        # ------------------------------------------------------------------
        # Footer
        # ------------------------------------------------------------------
        gr.Markdown(
            "---\n"
            "*Health Monitor v1.0 | "
            "[GitHub Repository](https://github.com/health-monitor) | "
            "Built with Gradio*"
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Load resources at module scope for HF Spaces cold-start optimisation.
# This runs when the module is first imported (during HF Spaces startup).
_load_resources()

# Build the Gradio app
app = _build_ui()

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
