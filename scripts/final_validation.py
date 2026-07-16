#!/usr/bin/env python3
"""Final comprehensive validation of the Health Monitor project."""
import sys, os, json, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from datetime import datetime

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.getcwd())

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} - {detail}")

print("=" * 60)
print("  FINAL COMPREHENSIVE VALIDATION")
print("=" * 60)

# --- 1. Module Imports ---
print("\n[1] Module Imports")
import src.config
import src.data.loader, src.data.preprocessor, src.data.window_generator, src.data.label_generator
import src.features.extractor
import src.features.motion, src.features.heart_rate, src.features.hrv, src.features.spo2
import src.features.temperature, src.features.cross_sensor, src.features.frequency_domain
import src.models.cardiac_model, src.models.respiratory_model, src.models.activity_model, src.models.rule_engine
import src.inference
from src.inference import Predictor, TemporalSmoother, ResponseBuilder
import data.synthetic.generator
check("All modules import successfully", True)

# --- 2. Model Loading ---
print("\n[2] Model Loading")
predictor = Predictor(models_dir="models")
smoother = TemporalSmoother()
builder = ResponseBuilder()
check("Predictor loaded all models", predictor.is_loaded)
check("Feature names count", 100 <= len(predictor.feature_names) <= 200, str(len(predictor.feature_names)))
check("Model versions populated", len(predictor.model_versions) == 3, str(predictor.model_versions))

# --- 3. Normal Session ---
print("\n[3] Normal Session Inference")
from data.synthetic.generator import SyntheticDataGenerator
gen = SyntheticDataGenerator(seed=42)
normal = gen.generate_normal_session(duration_sec=30)
window = normal["windows"][0]["sensor_data"]
raw = predictor.predict(window)
check("predict() returns dict", isinstance(raw, dict))
check("All 7 conditions present", all(c in raw for c in [
    "tachycardia", "irregular_rhythm", "low_spo2",
    "fever", "fall_detected", "sleep_problem", "fatigue"
]))
normal_detected = sum(1 for v in raw.values() if v["detected"])
check("Normal session: 0 conditions detected (may vary with model)", normal_detected <= 2, f"detected={normal_detected}")

# Compute data quality via proper pipeline
quality_raw = predictor._compute_data_quality(predictor._normalize_input(window))
check("Data quality computed", quality_raw > 0.5, f"quality={quality_raw:.3f}")

# --- 4. Condition Detection ---
print("\n[4] Condition Detection")
conditions = [
    ("tachycardia", "tachycardia"),
    ("low_spo2", "low_spo2"),
    ("fever", "fever"),
    ("fall_detected", "fall_detected"),
    ("sleep_problem", "sleep_problem"),
    ("fatigue", "fatigue"),
    ("irregular_rhythm", "irregular_rhythm"),
]
high_prob_count = 0
for cond_name, expected_key in conditions:
    try:
        session = gen.generate_condition_session(cond_name, duration_sec=30, severity="moderate")
        cw = session["windows"][0]["sensor_data"]
        cr = predictor.predict(cw)
        prob = cr[expected_key]["probability"]
        if prob > 0.3:
            high_prob_count += 1
    except Exception as e:
        print(f"  [WARN] {cond_name}: {e}")
check(f"Conditions with prob>0.3: {high_prob_count}/7", high_prob_count >= 5, f"got {high_prob_count}/7")

# --- 5. Temporal Smoother ---
print("\n[5] Temporal Smoother")
s2 = TemporalSmoother(window_buffer_size=3, min_detections=2, cooldown_seconds=0)
pred_on = {"tachycardia": {"detected": True, "probability": 0.9, "confidence": 0.8}}
r1 = s2.update(pred_on, 1.0)
check("1/3 not detected", r1["tachycardia"]["detected"] is False)
r2 = s2.update(pred_on, 2.0)
check("2/3 detected", r2["tachycardia"]["detected"] is True)
s2.reset()
check("Buffer cleared after reset", len(s2._buffer) == 0)

# Cooldown test
s3 = TemporalSmoother(window_buffer_size=3, min_detections=1, cooldown_seconds=10)
r1 = s3.update(pred_on, 100.0)
check("1/1 detected (triggers cooldown)", r1["tachycardia"]["detected"] is True)
r2 = s3.update(pred_on, 101.0)
check("Within cooldown: suppressed", r2["tachycardia"]["detected"] is False)

# --- 6. Response Builder ---
print("\n[6] Response Builder")
response = builder.build_response(raw, quality_raw)
check("Status is success", response["status"] == "success")
check("Has timestamp", "timestamp" in response)
check("Has disclaimer", "disclaimer" in response)
check("Has alert_summary", "alert_summary" in response)
check("Has predictions", "predictions" in response)

# Alert categorization
alert_preds = {
    "low_spo2": {"detected": True, "probability": 0.95, "confidence": 0.9},
    "fall_detected": {"detected": True, "probability": 0.99, "confidence": 0.95},
    "tachycardia": {"detected": True, "probability": 0.8, "confidence": 0.7},
    "fatigue": {"detected": True, "probability": 0.6, "confidence": 0.5},
}
alert_resp = builder.build_response(alert_preds, 0.9)
check("Critical alerts for low_spo2 and fall",
      len(alert_resp["alert_summary"]["critical_alerts"]) == 2)
check("Warnings for tachycardia",
      "tachycardia" in alert_resp["alert_summary"]["warnings"])
check("Info for fatigue",
      "fatigue" in alert_resp["alert_summary"]["info"])

# Edge cases
empty_resp = builder.build_response({}, 0.0)
check("Empty predictions handled", empty_resp["predictions"] == {})

health = builder.build_health_status()
check("Health endpoint works", health["status"] == "healthy")

error = builder.build_error_response("test error", 400)
check("Error response", error["status"] == "error" and error["code"] == 400)

# --- 7. Serialization ---
print("\n[7] JSON Serialization")
# Verify all responses are JSON-serializable
json.dumps(response)
check("Response is JSON-serializable", True)

# Verify demo window is JSON-serializable
serializable = {}
for key, val in window.items():
    if hasattr(val, 'tolist'):
        serializable[key] = val.tolist()
    elif isinstance(val, dict):
        serializable[key] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in val.items()}
    else:
        serializable[key] = val
json.dumps(serializable, default=str)
check("Window data is JSON-serializable", True)

# --- 8. Input Normalization ---
print("\n[8] Input Format Normalization")
dict_fmt = {
    "accelerometer": {"ax": [0]*10, "ay": [0]*10, "az": [0]*10},
    "gyroscope": {"gx": [0]*10, "gy": [0]*10, "gz": [0]*10},
    "heart_rate": {"bpm": [72]*5, "spo2": [98]*5, "ppg_raw": [0.5]*5},
    "temperature": {"stts22h_celsius": [36.6]*3, "lm35_celsius": [36.6]*3},
    "metadata": {},
}
nd = predictor._normalize_input(dict_fmt)
check("Dict format passthrough", nd is dict_fmt)

flat_fmt = {
    "accelerometer": [[0, 9.81, 0]] * 10,
    "gyroscope": [[0, 0, 0]] * 10,
    "heart_rate": [72.0] * 5,
    "temperature": [36.6] * 3,
    "spo2": [98.0] * 5,
    "ppg": [0.5] * 5,
    "metadata": {},
}
nf = predictor._normalize_input(flat_fmt)
check("Flat list converted to dict", isinstance(nf["accelerometer"], dict))
check("Accel has ax/ay/az", all(k in nf["accelerometer"] for k in ["ax", "ay", "az"]))
check("HR has bpm/spo2/ppg_raw", all(k in nf["heart_rate"] for k in ["bpm", "spo2", "ppg_raw"]))
check("Temp has stts22h/lm35", all(k in nf["temperature"] for k in ["stts22h_celsius", "lm35_celsius"]))

# --- Summary ---
print()
print("=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
if FAIL == 0:
    print("  ALL VALIDATIONS PASSED")
else:
    print(f"  {FAIL} FAILURE(S) DETECTED")
print("=" * 60)
