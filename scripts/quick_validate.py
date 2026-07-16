"""Quick validation script for Phase 4 inference modules."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def make_test_window():
    """Create a minimal normal window for testing."""
    np.random.seed(42)
    dur, fs_a, fs_h = 30, 50, 25
    n_a, n_h = dur * fs_a, dur * fs_h
    t_h = np.linspace(0, dur, n_h)
    ax = np.random.normal(0, 0.02, n_a)
    ay = np.random.normal(9.81, 0.03, n_a)
    az = np.random.normal(0, 0.02, n_a)
    hr = np.random.normal(72, 4, n_h).clip(60, 85)
    spo2 = np.random.normal(98, 0.5, n_h).clip(96, 100)
    ppg = 0.5*np.sin(2*np.pi*72/60*t_h) + 0.1*np.random.randn(n_h)
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min() + 1e-9) * 1024 + 512
    temp = np.random.normal(36.6, 0.05, dur)
    return {
        "accelerometer": {"ax": ax, "ay": ay, "az": az},
        "gyroscope": {"gx": np.random.normal(0,0.005,n_a), "gy": np.random.normal(0,0.005,n_a), "gz": np.random.normal(0,0.005,n_a)},
        "heart_rate": {"bpm": hr, "spo2": spo2, "ppg_raw": ppg},
        "temperature": {"stts22h_celsius": temp, "lm35_celsius": temp + 0.1},
        "metadata": {"activity_state": "resting", "window_duration_sec": dur},
    }

def main():
    # Test imports
    from src.inference import Predictor, TemporalSmoother, ResponseBuilder
    print("All inference classes imported OK")

    # Test TemporalSmoother
    smoother = TemporalSmoother(window_buffer_size=3, min_detections=2, cooldown_seconds=0)
    test_pred = {
        "tachycardia": {"detected": True, "probability": 0.8, "confidence": 0.7},
        "low_spo2": {"detected": False, "probability": 0.1, "confidence": 0.9},
        "fever": {"detected": False, "probability": 0.0, "confidence": 1.0},
    }
    for i in range(3):
        result = smoother.update(test_pred, float(i))
    assert result["tachycardia"]["detected"] is True  # 3/3 >= 2, cooldown disabled
    assert result["low_spo2"]["detected"] is False
    status = smoother.get_status()
    assert "buffer_size" in status
    print(f"TemporalSmoother OK (buffer={status['buffer_size']}, cooldown_state={status['cooldown_state']})")

    # Test ResponseBuilder
    builder = ResponseBuilder()
    response = builder.build_response(test_pred, 0.95, {"test": True})
    assert response["status"] == "success"
    assert "data_quality_score" in response
    assert "predictions" in response
    assert "disclaimer" in response
    assert "alert_summary" in response
    assert "critical_alerts" in response["alert_summary"]
    assert "tachycardia" in response["alert_summary"]["warnings"]
    print(f"ResponseBuilder OK (quality={response['data_quality_score']}, alerts={len(response['alert_summary']['warnings'])})")

    # Test health and error responses
    health = builder.build_health_status()
    assert health["status"] == "healthy"
    error = builder.build_error_response("Test error")
    assert error["status"] == "error"
    print("ResponseBuilder health/error endpoints OK")

    # Test Predictor (will show warnings if models don't exist)
    predictor = Predictor(models_dir="models", use_onnx=False)
    expected_conditions = [
        "tachycardia", "irregular_rhythm", "low_spo2",
        "fever", "fall_detected", "sleep_problem", "fatigue"
    ]
    assert hasattr(predictor, 'feature_names')
    assert hasattr(predictor, 'model_versions')
    assert hasattr(predictor, 'is_loaded')
    print(f"Predictor API OK (loaded={predictor.is_loaded}, features={len(predictor.feature_names)})")

    # Try a prediction (may return zeros if no models)
    window = make_test_window()
    result = predictor.predict(window)
    assert isinstance(result, dict)
    for cond in expected_conditions:
        assert cond in result, f"Missing condition: {cond}"
        assert "detected" in result[cond]
        assert "probability" in result[cond]
    print(f"Prediction OK ({sum(1 for c in result.values() if c['detected'])} conditions detected)")

    # Test feature alignment
    features = predictor._extract_features(window)
    aligned = predictor._align_features(features)
    assert len(aligned[0]) == len(predictor.feature_names)
    assert np.isfinite(aligned).all()
    print(f"Feature alignment OK ({len(aligned[0])} features)")

    # Test predict_proba
    proba = predictor.predict_proba(window)
    for cond in expected_conditions:
        assert cond in proba
        assert 0.0 <= proba[cond]["probability"] <= 1.0
    print("predict_proba OK")

    # Test TemporalSmoother with real predictions
    smoother.reset()
    for i in range(5):
        r = smoother.update(result, float(i))
    assert len(smoother._buffer) <= 3
    print("TemporalSmoother with real data OK")

    # Test full response pipeline
    quality = predictor._compute_data_quality(window)
    full = builder.build_response(result, quality, {"source": "test"})
    assert full["status"] == "success"
    assert full["data_quality_score"] >= 0.0
    print(f"Full response pipeline OK (status={full['status']})")

    print("\nAll Phase 4 validations passed!")

if __name__ == "__main__":
    main()
