"""Tests for all feature extraction modules."""
import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.motion import MotionFeatures
from src.features.heart_rate import HeartRateFeatures
from src.features.hrv import HRVFeatures
from src.features.spo2 import SpO2Features
from src.features.temperature import TemperatureFeatures
from src.features.cross_sensor import CrossSensorFeatures
from src.features.frequency_domain import FrequencyDomainFeatures
from src.features.extractor import FeatureExtractor


# ============================================================================
# Helper: build a baseline normal window dict
# ============================================================================

def _make_normal_window(seed=42):
    """Create a normal resting window dict for testing."""
    fs_accel = 50
    fs_hr = 25
    duration = 30
    n_accel = fs_accel * duration
    n_hr = fs_hr * duration

    t_hr = np.linspace(0, duration, n_hr, endpoint=False)
    rng = np.random.default_rng(seed)

    ax = rng.normal(0, 0.02, n_accel)
    ay = rng.normal(9.81, 0.03, n_accel)   # gravity on y-axis
    az = rng.normal(0, 0.02, n_accel)
    gx = rng.normal(0, 0.005, n_accel)
    gy = rng.normal(0, 0.005, n_accel)
    gz = rng.normal(0, 0.005, n_accel)

    hr_bpm = rng.normal(72, 4, n_hr).clip(60, 85)
    spo2 = rng.normal(98, 0.5, n_hr).clip(96, 100)

    # Temperature: 30 samples at 1 Hz (30s duration)
    temp_stts22h = rng.normal(36.6, 0.05, duration)
    temp_lm35 = rng.normal(36.7, 0.08, duration)

    # PPG waveform
    hr_hz = 72 / 60
    ppg_raw = (0.5 * np.sin(2 * np.pi * hr_hz * t_hr) +
               0.25 * np.sin(2 * np.pi * 2 * hr_hz * t_hr) +
               0.1 * np.sin(2 * np.pi * 3 * hr_hz * t_hr) +
               0.05 * rng.standard_normal(n_hr))
    ppg_raw = ppg_raw - ppg_raw.min()
    ppg_raw = ppg_raw / ppg_raw.max() * 1024 + 512

    return {
        "accelerometer": {"ax": ax, "ay": ay, "az": az},
        "gyroscope": {"gx": gx, "gy": gy, "gz": gz},
        "heart_rate": {"bpm": hr_bpm, "spo2": spo2, "ppg_raw": ppg_raw},
        "temperature": {
            "stts22h_celsius": temp_stts22h,
            "lm35_celsius": temp_lm35,
        },
        "metadata": {"activity_state": "resting", "window_duration_sec": duration},
    }


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def normal_resting_window():
    return _make_normal_window(seed=42)


@pytest.fixture
def tachycardia_window():
    """Window with elevated heart rate (≥100 BPM at rest)."""
    data = _make_normal_window(seed=99)
    n_hr = len(data["heart_rate"]["bpm"])
    rng = np.random.default_rng(99)
    data["heart_rate"]["bpm"] = rng.normal(115, 5, n_hr).clip(100, 130)
    data["heart_rate"]["spo2"] = rng.normal(97, 0.8, n_hr).clip(95, 100)
    data["metadata"]["activity_state"] = "resting"
    return data


@pytest.fixture
def low_spo2_window():
    """Window with low blood oxygen (<95%)."""
    data = _make_normal_window(seed=99)
    n_hr = len(data["heart_rate"]["bpm"])
    rng = np.random.default_rng(99)
    data["heart_rate"]["bpm"] = rng.normal(88, 5, n_hr).clip(78, 100)
    data["heart_rate"]["spo2"] = rng.normal(91, 2, n_hr).clip(85, 95)
    return data


@pytest.fixture
def fever_window():
    """Window with elevated temperature (≥38°C)."""
    data = _make_normal_window(seed=99)
    dur = 30
    rng = np.random.default_rng(99)
    data["temperature"]["stts22h_celsius"] = rng.normal(38.5, 0.1, dur)
    data["temperature"]["lm35_celsius"] = rng.normal(38.6, 0.15, dur)
    return data


@pytest.fixture
def empty_window():
    return {
        "accelerometer": {"ax": np.array([]), "ay": np.array([]), "az": np.array([])},
        "gyroscope": {"gx": np.array([]), "gy": np.array([]), "gz": np.array([])},
        "heart_rate": {"bpm": np.array([]), "spo2": np.array([]), "ppg_raw": np.array([])},
        "temperature": {},
        "metadata": {"activity_state": "unknown"},
    }


# ============================================================================
# Tests: MotionFeatures
# ============================================================================

class TestMotionFeatures:
    def test_extract_returns_dict(self, normal_resting_window):
        features = MotionFeatures().extract(normal_resting_window)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_contains_key_features(self, normal_resting_window):
        features = MotionFeatures().extract(normal_resting_window)
        expected = [
            "motion_ax_mean", "motion_ay_mean", "motion_az_mean",
            "motion_ax_std", "motion_ay_std", "motion_az_std",
            "motion_sma", "motion_movement_intensity", "motion_tilt_angle_mean",
        ]
        for key in expected:
            assert key in features, f"Missing feature: {key}"

    def test_gravity_on_y_axis(self, normal_resting_window):
        features = MotionFeatures().extract(normal_resting_window)
        assert abs(features["motion_ay_mean"] - 9.81) < 0.2, \
            f"y-axis should be ~9.81, got {features['motion_ay_mean']}"
        assert abs(features["motion_ax_mean"]) < 0.3, \
            f"x-axis should be near 0, got {features['motion_ax_mean']}"
        assert abs(features["motion_az_mean"]) < 0.3, \
            f"z-axis should be near 0, got {features['motion_az_mean']}"

    def test_empty_returns_defaults(self, empty_window):
        features = MotionFeatures().extract(empty_window)
        assert len(features) > 0
        assert all(v == 0.0 for v in features.values())

    def test_get_feature_names(self):
        names = MotionFeatures().get_feature_names()
        assert len(names) > 20
        assert all(isinstance(n, str) for n in names)


# ============================================================================
# Tests: HeartRateFeatures (feature names: hr_*)
# ============================================================================

class TestHeartRateFeatures:
    def test_extract_returns_dict(self, normal_resting_window):
        features = HeartRateFeatures().extract(normal_resting_window)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_normal_hr_range(self, normal_resting_window):
        features = HeartRateFeatures().extract(normal_resting_window)
        assert 55 <= features["hr_mean"] <= 90, features["hr_mean"]
        assert "hr_min" in features
        assert "hr_max" in features

    def test_tachycardia_detected(self, tachycardia_window):
        features = HeartRateFeatures().extract(tachycardia_window)
        assert features["hr_mean"] > 100, f"mean HR = {features['hr_mean']}"
        # Check for tachycardia indicator feature
        tach_keys = [k for k in features if "tachy" in k.lower() or "above_100" in k]
        if tach_keys:
            assert features[tach_keys[0]] > 0.3

    def test_empty_returns_defaults(self, empty_window):
        features = HeartRateFeatures().extract(empty_window)
        assert len(features) > 0
        assert all(v == 0.0 for v in features.values())

    def test_hr_std_present(self, normal_resting_window):
        features = HeartRateFeatures().extract(normal_resting_window)
        assert "hr_std" in features


# ============================================================================
# Tests: HRVFeatures (feature names: hrv_*)
# ============================================================================

class TestHRVFeatures:
    def test_extract_returns_dict(self, normal_resting_window):
        features = HRVFeatures().extract(normal_resting_window)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_hrv_contains_standard_metrics(self, normal_resting_window):
        features = HRVFeatures().extract(normal_resting_window)
        expected = ["hrv_rmssd", "hrv_std_rr", "hrv_mean_rr", "hrv_sd1", "hrv_sd2", "hrv_cv"]
        for key in expected:
            assert key in features, f"Missing HRV feature: {key}"
            assert isinstance(features[key], float), f"{key} should be float"

    def test_normal_hrv_values(self, normal_resting_window):
        features = HRVFeatures().extract(normal_resting_window)
        assert features["hrv_rmssd"] >= 0
        assert features["hrv_std_rr"] >= 0

    def test_empty_returns_defaults(self, empty_window):
        features = HRVFeatures().extract(empty_window)
        assert len(features) > 0
        assert all(v == 0.0 for v in features.values())


# ============================================================================
# Tests: SpO2Features (feature names: spo2_*)
# ============================================================================

class TestSpO2Features:
    def test_extract_returns_dict(self, normal_resting_window):
        features = SpO2Features().extract(normal_resting_window)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_normal_spo2_levels(self, normal_resting_window):
        features = SpO2Features().extract(normal_resting_window)
        assert features["spo2_mean"] > 96, f"mean SpO2 = {features['spo2_mean']}"
        assert features["spo2_hypoxemia_pct"] < 0.2, \
            f"hypoxemia pct = {features['spo2_hypoxemia_pct']}"

    def test_low_spo2_detected(self, low_spo2_window):
        features = SpO2Features().extract(low_spo2_window)
        assert features["spo2_mean"] < 95, f"mean SpO2 = {features['spo2_mean']}"
        assert features["spo2_hypoxemia_pct"] > 0.3, \
            f"hypoxemia pct = {features['spo2_hypoxemia_pct']}"

    def test_empty_returns_defaults(self, empty_window):
        features = SpO2Features().extract(empty_window)
        assert len(features) > 0
        assert all(v == 0.0 for v in features.values())


# ============================================================================
# Tests: TemperatureFeatures (feature names: temp_*)
# ============================================================================

class TestTemperatureFeatures:
    def test_extract_returns_dict(self, normal_resting_window):
        features = TemperatureFeatures().extract(normal_resting_window)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_normal_temp(self, normal_resting_window):
        features = TemperatureFeatures().extract(normal_resting_window)
        assert 35.0 <= features["temp_stts22h_mean"] <= 38.0
        assert features["temp_fever_indicator"] == 0.0

    def test_fever_detected(self, fever_window):
        features = TemperatureFeatures().extract(fever_window)
        assert features["temp_fever_indicator"] == 1.0, \
            f"fever_indicator = {features['temp_fever_indicator']}"
        assert features["temp_stts22h_mean"] >= 38.0, \
            f"mean temp = {features['temp_stts22h_mean']}"

    def test_empty_returns_defaults(self, empty_window):
        features = TemperatureFeatures().extract(empty_window)
        assert len(features) > 0
        assert all(v == 0.0 for v in features.values())


# ============================================================================
# Tests: CrossSensorFeatures (feature names: xsensor_*)
# ============================================================================

class TestCrossSensorFeatures:
    def test_extract_returns_dict(self, normal_resting_window):
        features = CrossSensorFeatures().extract(normal_resting_window)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_contains_expected_features(self, normal_resting_window):
        features = CrossSensorFeatures().extract(normal_resting_window)
        expected = [
            "xsensor_hr_temp_correlation",
            "xsensor_hr_spo2_ratio",
            "xsensor_activity_adjusted_hr",
            "xsensor_cardio_respiratory_index",
            "xsensor_cardiovascular_strain",
        ]
        for key in expected:
            assert key in features, f"Missing cross-sensor feature: {key}"

    def test_empty_returns_defaults(self, empty_window):
        features = CrossSensorFeatures().extract(empty_window)
        assert len(features) > 0
        # All values should be finite (no NaN/Inf) even for empty data
        for k, v in features.items():
            assert np.isfinite(v), f"{k} = {v} (not finite)"


# ============================================================================
# Tests: FrequencyDomainFeatures
# ============================================================================

class TestFrequencyDomainFeatures:
    def test_extract_returns_dict(self, normal_resting_window):
        features = FrequencyDomainFeatures().extract(normal_resting_window)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_contains_dominant_freq(self, normal_resting_window):
        features = FrequencyDomainFeatures().extract(normal_resting_window)
        dom_keys = [k for k in features if "dominant_freq" in k]
        assert len(dom_keys) >= 3, f"Found dom freq keys: {dom_keys}"

    def test_spectral_entropy_present(self, normal_resting_window):
        features = FrequencyDomainFeatures().extract(normal_resting_window)
        entropy_keys = [k for k in features if "entropy" in k]
        assert len(entropy_keys) > 0, f"No entropy keys in {list(features.keys())[:5]}"

    def test_empty_returns_defaults(self, empty_window):
        features = FrequencyDomainFeatures().extract(empty_window)
        assert len(features) > 0
        assert all(v == 0.0 for v in features.values())


# ============================================================================
# Tests: FeatureExtractor (Orchestrator)
# ============================================================================

class TestFeatureExtractor:
    def test_extract_all_returns_dict(self, normal_resting_window):
        features = FeatureExtractor().extract_all(normal_resting_window)
        assert isinstance(features, dict)
        assert len(features) > 50

    def test_feature_count(self):
        count = FeatureExtractor().feature_count()
        assert 80 < count < 300, f"feature count = {count}"

    def test_all_feature_names_are_strings(self):
        names = FeatureExtractor().get_all_feature_names()
        assert all(isinstance(n, str) for n in names)
        assert len(set(names)) == len(names)

    def test_extract_with_empty_window(self, empty_window):
        features = FeatureExtractor().extract_all(empty_window)
        assert isinstance(features, dict)

    def test_end_to_end_normal(self, normal_resting_window):
        features = FeatureExtractor().extract_all(normal_resting_window)
        assert "hr_mean" in features
        assert 55 <= features.get("hr_mean", 0) <= 90
        assert "spo2_mean" in features
        assert features.get("spo2_mean", 0) > 95
        assert "temp_stts22h_mean" in features
        assert features.get("temp_stts22h_mean", 0) < 38.0


# ============================================================================
# Tests: Data Pipeline Integration
# ============================================================================

class TestDataPipeline:
    @staticmethod
    def _wrap_as_window(data):
        """Wrap a flat fixture dict into the window format expected by LabelGenerator.

        The label generator expects flat numpy arrays under sensor_data:
          accelerometer: (N, 3) or (N,) array of magnitude
          gyroscope: (N, 3) array
          heart_rate: (N,) array of BPM values
          spo2: (N,) array of SpO2 values
          temperature: (N,) array of temperature values
          ppg: (N,) array of raw PPG values
        """
        accel = data["accelerometer"]
        # Convert dict-of-arrays to (N, 3) array in g-units (divide by 9.81)
        ax, ay, az = accel["ax"], accel["ay"], accel["az"]
        if ax.size > 0:
            accel_3d = np.column_stack([ax / 9.81, ay / 9.81, az / 9.81])
        else:
            accel_3d = np.array([])

        gyro = data["gyroscope"]
        gx, gy, gz = gyro["gx"], gyro["gy"], gyro["gz"]
        gyro_3d = np.column_stack([gx, gy, gz]) if gx.size > 0 else np.array([])

        temp = data["temperature"]
        # Temperature: use stts22h if available, else average both
        if "stts22h_celsius" in temp:
            stts = temp["stts22h_celsius"]
            temp_arr = stts if isinstance(stts, np.ndarray) else np.array([stts])
        else:
            temp_arr = np.array([])

        hr = data["heart_rate"]
        ppg = hr.get("ppg_raw", np.array([]))

        return {
            "sensor_data": {
                "accelerometer": accel_3d,
                "gyroscope": gyro_3d,
                "heart_rate": hr.get("bpm", np.array([])),
                "spo2": hr.get("spo2", np.array([])),
                "temperature": temp_arr,
                "ppg": ppg,
            },
            "metadata": data.get("metadata", {}),
            "start_time": 0.0,
            "end_time": 30.0,
        }

    def test_label_generator_produces_labels(self, normal_resting_window):
        from src.data.label_generator import LabelGenerator
        window = self._wrap_as_window(normal_resting_window)
        labels = LabelGenerator().generate_labels(window)
        assert isinstance(labels, dict)
        for cond in ["tachycardia", "irregular_rhythm", "low_spo2",
                      "fever", "fall_detected", "sleep_problem", "fatigue"]:
            assert cond in labels, f"Missing label: {cond}"
            assert "detected" in labels[cond]
            assert "confidence" in labels[cond]

    def test_normal_window_no_alarms(self, normal_resting_window):
        from src.data.label_generator import LabelGenerator
        window = self._wrap_as_window(normal_resting_window)
        labels = LabelGenerator().generate_labels(window)
        for cond in ["tachycardia", "low_spo2", "fever", "fall_detected"]:
            assert labels[cond]["detected"] is False, \
                f"{cond} should be False (got confidence={labels[cond]['confidence']})"
            assert labels[cond]["confidence"] < 0.5

    def test_tachycardia_label(self, tachycardia_window):
        from src.data.label_generator import LabelGenerator
        window = self._wrap_as_window(tachycardia_window)
        labels = LabelGenerator().generate_labels(window)
        # HR ~115 BPM, threshold 100, upper 150, confidence = (115-100)/50 = 0.30
        assert labels["tachycardia"]["detected"] is True
        assert labels["tachycardia"]["confidence"] >= 0.25

    def test_fever_label(self, fever_window):
        from src.data.label_generator import LabelGenerator
        window = self._wrap_as_window(fever_window)
        labels = LabelGenerator().generate_labels(window)
        # Temp ~38.5°C, threshold 38.0°C, confidence = (38.5-38.0)/2.0 = 0.25
        assert labels["fever"]["detected"] is True
        assert labels["fever"]["confidence"] >= 0.2

    def test_window_generator_output(self):
        """Verify WindowGenerator yields correct number of windows."""
        from src.data.window_generator import WindowGenerator
        session = {
            "session_id": "test",
            "windows": [{
                "window_id": 0,
                "timestamp": "2024-01-01T00:00:00Z",
                "duration_sec": 30.0,
                "sensor_data": {
                    "accelerometer": [[0.0, 9.81, 0.0]] * 1500,
                    "gyroscope": [[0.0, 0.0, 0.0]] * 1500,
                    "heart_rate": [72.0] * 750,
                    "temperature": [36.5] * 30,
                    "spo2": [98.0] * 750,
                    "ppg": [512] * 750,
                },
                "metadata": {},
            }]
        }
        wg = WindowGenerator(session, window_duration_sec=10, stride_sec=10)
        windows = wg.generate_windows()
        assert len(windows) >= 1

    def test_preprocessor_filters_signal(self):
        from src.data.preprocessor import SensorPreprocessor
        sp = SensorPreprocessor()
        fs = 50
        t = np.linspace(0, 10, fs * 10, endpoint=False)
        signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
        filtered = sp.filter_signal(signal, fs, lowcut=1, highcut=5)
        assert len(filtered) == len(signal)
        # High-frequency (30 Hz) component should be attenuated more than low-freq (2 Hz)
        # Compare amplitude at the output vs input
        # Input: 2 Hz sine (amp 1.0) + 30 Hz sine (amp 0.5)
        # After filtering (1-5 Hz bandpass), the 30 Hz should be reduced
        assert np.max(np.abs(filtered)) < np.max(np.abs(signal)) * 0.95 or \
               np.std(filtered) < np.std(signal) * 0.95

    def test_loader_validate_schema(self):
        from src.data.loader import SensorDataLoader
        loader = SensorDataLoader(strict=False)
        valid = loader.validate_schema({
            "session_id": "t1", "timestamp_start": "2024-01-01",
            "sampling_config": {"accel_sample_rate_hz": 50},
            "windows": [{"window_id": 0, "duration_sec": 30.0, "timestamp": "",
                         "sensor_data": {"accelerometer": [[0, 9.81, 0]],
                                         "gyroscope": [[0, 0, 0]],
                                         "heart_rate": [72],
                                         "temperature": [36.5]}}],
            "metadata": {}
        })
        assert valid is True

    def test_loader_rejects_invalid_schema(self):
        from src.data.loader import SensorDataLoader
        loader = SensorDataLoader(strict=True)
        with pytest.raises(Exception):
            loader.validate_schema({"bad": "data"})


# ============================================================================
# Tests: Synthetic Data Generator
# ============================================================================

class TestSyntheticDataGenerator:
    @pytest.fixture
    def generator(self):
        from data.synthetic.generator import SyntheticDataGenerator
        return SyntheticDataGenerator(seed=42)

    def test_generate_normal_session(self, generator):
        session = generator.generate_normal_session(duration_sec=30)
        assert "session_id" in session
        assert "windows" in session
        assert len(session["windows"]) > 0
        w = session["windows"][0]
        sd = w["sensor_data"]
        assert "accelerometer" in sd
        assert "heart_rate" in sd
        assert "temperature" in sd

    def test_generate_tachycardia_session(self, generator):
        session = generator.generate_condition_session("tachycardia", duration_sec=30)
        assert session["metadata"]["condition"] == "tachycardia"
        assert "ground_truth" in session
        assert session["ground_truth"]["tachycardia"] is True

    def test_generate_fever_session(self, generator):
        session = generator.generate_condition_session("fever", duration_sec=30)
        assert session["metadata"]["condition"] == "fever"
        assert session["ground_truth"]["fever"] is True

    def test_generate_fall_session(self, generator):
        session = generator.generate_condition_session("fall_detected", duration_sec=30)
        assert session["ground_truth"]["fall_detected"] is True

    def test_generate_dataset(self, generator):
        sessions = generator.generate_dataset(num_sessions=3, output_dir=None, include_labels=True)
        assert len(sessions) == 3
        for s in sessions:
            assert "ground_truth" in s


if __name__ == "__main__":
    pytest.main(["-v", __file__])
