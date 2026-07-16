# Health Symptom Detection AI — Complete Pipeline Architecture

## Table of Contents

1. [System Constraints Recap](#1-system-constraints-recap)
2. [Overall Pipeline Diagram](#2-overall-pipeline-diagram)
3. [Training Pipeline (Local GPU)](#3-training-pipeline-local-gpu)
4. [Inference Pipeline (HF CPU)](#4-inference-pipeline-hf-cpu)
5. [Data Pipeline](#5-data-pipeline)
6. [MLOps & Workflow](#6-mlops--workflow)
7. [Project Structure](#7-project-structure)
8. [Key Design Decisions](#8-key-design-decisions)
9. [Risk & Mitigation](#9-risk--mitigation)
10. [Appendix: Feature Catalog](#appendix-a-feature-catalog)
11. [Appendix: Sensor Specifications](#appendix-b-sensor-specifications)

---

## 1. System Constraints Recap

| Component | Specification | Implication |
|-----------|--------------|-------------|
| **Training HW** | NVIDIA RTX 2050 (4GB VRAM, ~10 TFLOPS) | Small models only; LightGBM GPU training feasible; batch size must be modest |
| **Inference HW** | HF Spaces CPU Basic (2 vCPU, 16GB RAM, 50GB disk) | No GPU; must stay under 512MB for model artifacts (see §9); cold start ~30-60s |
| **Language** | Python 3.10+ | Modern syntax, type hints, match/case |
| **Sensors** | MPU6500 (accel+gyro), MAX30102 (HR/SpO2/PPG), STTS22H (temp), LM35 (temp) | Mixed sampling rates (50-100Hz motion, 100Hz PPG, 1Hz temp) |
| **Model** | 3 LightGBM specialized models + rule-based | CPU-friendly, <50ms inference per window |
| **Conditions** | Tachycardia, Irregular Heart Rhythm, Low SpO2, Fever, Fall Detection, Sleep Problems, Fatigue | 7 binary classifiers (or multi-label) |

---

## 2. Overall Pipeline Diagram

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING PIPELINE (Local)                         │
│                     RTX 2050 GPU + 16-32GB RAM                             │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Synthetic│──▶│  Data    │──▶│ Feature  │──▶│  Model   │──▶│  Model   │ │
│  │   Data   │   │ Ingest & │   │Extract & │   │ Training │   │Registry  │ │
│  │Generator │   │ Clean    │   │ Window   │   │ (LGBM×3) │   │(joblib/  │ │
│  │          │   │          │   │ Generate │   │ GPU accel│   │ ONNX)    │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └────┬─────┘ │
│                                                                     │       │
└─────────────────────────────────────────────────────────────────────┼───────┘
                                                                      │
                          Push model artifacts via Git / HF Hub       │
                                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE (HF CPU)                         │
│                   HF Spaces CPU Basic (2 vCPU, 16GB RAM)                   │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Sensor  │──▶│ Feature  │──▶│ Ensemble │──▶│  Post-   │──▶│  JSON    │ │
│  │   JSON   │   │Extract & │   │ Aggregat.│   │Processing│   │ Response │ │
│  │  Input   │   │Normalize │   │ (3LGBM + │   │Threshold │   │  Output  │ │
│  │          │   │(identical│   │  Rules)  │   │ Temporal │   │          │ │
│  │          │   │to train) │   │          │   │ Smoothing│   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Detail

```
RAW SENSOR STREAMS (JSON batches every 1-5 seconds)
    │
    ├── MPU6500: {ax, ay, az, gx, gy, gz} @ 50-100 Hz
    ├── MAX30102: {ir, red} @ 100 Hz  (derived: HR, SpO2, PPG waveform)
    ├── STTS22H: {temperature} @ 1 Hz
    └── LM35:    {temperature} @ 1 Hz
         │
         ▼
    ┌─────────────────────────────┐
    │  1. JSON DESERIALIZATION    │
    │     Parse & validate schema │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  2. TIME ALIGNMENT          │
    │     Resample to common grid │
    │     (100 Hz for motion/PPG, │
    │      1 Hz for temperature)  │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  3. QUALITY CHECK           │
    │     Sensor validity flags   │
    │     Missing data detection  │
    │     Motion artifact flag    │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  4. SLIDING WINDOW          │
    │     Window: 30 sec          │
    │     Stride: 5 sec (overlap) │
    │     Per window → feature vec│
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  5. FEATURE EXTRACTION      │
    │     ~120 features per window│
    │     (see Appendix A)        │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  6. MODEL INFERENCE         │
    │     Model A: Cardiac        │
    │       → Tachycardia         │
    │       → Irregular Rhythm    │
    │     Model B: Oxygen & Temp  │
    │       → Low SpO2            │
    │       → Fever               │
    │     Model C: Motion & Sleep │
    │       → Fall Detection      │
    │       → Sleep Problems      │
    │       → Fatigue             │
    │     Rules: Hard thresholds  │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  7. ENSEMBLE AGGREGATION    │
    │     Weighted average of     │
    │     model + rule outputs    │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  8. POST-PROCESSING         │
    │     Confidence calibration  │
    │     Temporal smoothing      │
    │     (N-of-M voting)         │
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  9. JSON OUTPUT             │
    │     {condition: {detected,  │
    │      confidence}} × 7       │
    └─────────────────────────────┘
```

---

## 3. Training Pipeline (Local GPU)

### 3.1 Data Ingestion

**Sources:**
1. **Synthetic data** — Generated by `scripts/generate_synthetic.py` (see §5.1)
2. **Real sensor data** — JSON files from sensor device (future)

**Ingestion flow:**
```python
# src/data/ingest.py

def load_session(json_path: Path) -> pd.DataFrame:
    """Load a single sensor session from JSON.
    
    Expected JSON schema:
    {
        "session_id": "uuid",
        "start_time": "ISO8601",
        "sample_rate": {"motion": 100, "ppg": 100, "temp": 1},
        "sensors": {
            "mpu6500": {"timestamp": [...], "ax": [...], ...},
            "max30102": {"timestamp": [...], "ir": [...], "red": [...]},
            "stts22h": {"timestamp": [...], "temperature": [...]},
            "lm35":    {"timestamp": [...], "temperature": [...]}
        }
    }
    """
    ...

def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all sessions, return (features_df, labels_df).
    
    - Scans data_dir for *.json files
    - Loads each session via load_session()
    - Extracts features per window (§3.3)
    - Applies labels (§3.5)
    - Returns aligned DataFrames
    """
    ...
```

**File format on disk:**
```
data/
├── raw/
│   ├── synthetic/
│   │   ├── session_001.json
│   │   ├── session_002.json
│   │   └── ...
│   └── real/           # future
│       └── ...
├── processed/
│   ├── features.parquet    # precomputed feature matrix
│   ├── labels.parquet      # corresponding labels
│   ├── feature_names.json  # ordered feature names
│   └── scaler_params.json  # normalization parameters
└── splits/
    ├── train_ids.json
    ├── val_ids.json
    └── test_ids.json
```

### 3.2 Preprocessing

**Step-by-step cleaning:**

1. **Unit conversion**
   - MPU6500: raw ADC → g (accel) and °/s (gyro) using datasheet sensitivity
   - MAX30102: raw counts → PPG amplitude (no absolute unit needed for ML)
   - STTS22H: raw → °C (already digital, just scale)
   - LM35: ADC voltage → °C: `T = voltage / 0.01`

2. **Missing data handling**
   - Interpolate gaps < 500ms using linear interpolation
   - Mark gaps ≥ 500ms as NaN (feature extraction handles gracefully)
   - For motion: if >20% of window is missing → flag window as low quality

3. **Noise filtering**
   - Motion (MPU6500): 4th-order Butterworth low-pass at 20Hz cutoff
   - PPG (MAX30102): 0.5-5Hz bandpass for cardiac component
   - Temperature: median filter (window=5) to remove spikes

4. **Normalization** (per-window z-score for features, global scaler for raw)
   - Training: fit StandardScaler on training set → save mean/std
   - Inference: load saved scaler params, apply identically
   - **Critical**: Raw sensor values are NOT normalized globally (different sessions have different baselines). Features ARE normalized.

### 3.3 Feature Extraction

**Total: ~120 features per 30-second window**

The feature extraction is organized into 6 modules. Each module is a pure function that takes a window of raw data and returns a dict of features.

#### Module 1: Motion Features (MPU6500) — ~35 features

```python
# src/features/motion.py

def extract_motion_features(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,  # @100Hz, 3000 samples
    gx: np.ndarray, gy: np.ndarray, gz: np.ndarray
) -> dict[str, float]:
    """Extract acceleration and gyroscope features from a 30s window."""
    
    features = {}
    
    # --- Accelerometer time-domain (12 features) ---
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)  # magnitude
    features["acc_mean_x"] = np.mean(ax)
    features["acc_mean_y"] = np.mean(ay)
    features["acc_mean_z"] = np.mean(az)
    features["acc_std_x"] = np.std(ax)
    features["acc_std_y"] = np.std(ay)
    features["acc_std_z"] = np.std(az)
    features["acc_mag_mean"] = np.mean(acc_mag)
    features["acc_mag_std"] = np.std(acc_mag)
    features["acc_mag_max"] = np.max(acc_mag)
    features["acc_mag_min"] = np.min(acc_mag)
    features["acc_mag_range"] = features["acc_mag_max"] - features["acc_mag_min"]
    features["acc_rms"] = np.sqrt(np.mean(acc_mag**2))
    
    # --- Accelerometer frequency-domain (6 features) ---
    freqs, psd = signal.welch(acc_mag, fs=100, nperseg=512)
    features["acc_psd_peak_freq"] = freqs[np.argmax(psd)]
    features["acc_psd_peak_power"] = np.max(psd)
    features["acc_psd_mean"] = np.mean(psd)
    features["acc_energy_low"] = np.sum(psd[(freqs >= 0.5) & (freqs <= 5)])   # activity
    features["acc_energy_mid"] = np.sum(psd[(freqs >= 5) & (freqs <= 15)])    # motion
    features["acc_energy_high"] = np.sum(psd[(freqs >= 15) & (freqs <= 50)])  # vibration
    
    # --- Gyroscope features (8 features) ---
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    features["gyro_mean"] = np.mean(gyro_mag)
    features["gyro_std"] = np.std(gyro_mag)
    features["gyro_max"] = np.max(gyro_mag)
    features["gyro_rms"] = np.sqrt(np.mean(gyro_mag**2))
    features["gyro_psd_peak_freq"] = freqs[np.argmax(signal.welch(gyro_mag, fs=100, nperseg=512)[1])]
    features["gyro_energy"] = np.sum(signal.welch(gyro_mag, fs=100, nperseg=512)[1])
    features["gyro_entropy"] = _spectral_entropy(gyro_mag, fs=100)
    features["gyro_kurtosis"] = scipy.stats.kurtosis(gyro_mag)
    
    # --- Cross-axis features (4 features) ---
    features["acc_corr_xy"] = np.corrcoef(ax, ay)[0, 1]
    features["acc_corr_xz"] = np.corrcoef(ax, az)[0, 1]
    features["acc_corr_yz"] = np.corrcoef(ay, az)[0, 1]
    features["acc_jerk_mean"] = np.mean(np.diff(acc_mag))  # rate of change
    
    # --- Fall detection features (5 features) ---
    features["acc_impact_peak"] = np.max(acc_mag)  # spike detection
    features["acc_freefall_duration"] = np.sum(acc_mag < 0.5) / 100  # seconds < 0.5g
    features["acc_impact_rise_time"] = _compute_rise_time(acc_mag, fs=100)
    features["acc_post_impact_std"] = _post_impact_variability(acc_mag, fs=100)
    features["acc_tilt_angle"] = _compute_tilt_angle(ax, ay, az)
    
    return features
```

#### Module 2: Cardiac Features (MAX30102 PPG) — ~30 features

```python
# src/features/cardiac.py

def extract_cardiac_features(
    ir_signal: np.ndarray,     # @100Hz, 3000 samples
    red_signal: np.ndarray,    # @100Hz, 3000 samples
    fs: int = 100
) -> dict[str, float]:
    """Extract heart rate, HRV, SpO2 features from PPG window."""
    
    features = {}
    
    # --- PPG preprocessing ---
    ppg_filtered = _bandpass_filter(ir_signal, low=0.5, high=5.0, fs=fs, order=4)
    peaks, _ = signal.find_peaks(ppg_filtered, distance=fs*0.5, prominence=0.1)
    
    if len(peaks) < 3:
        # Too few peaks for reliable analysis — fill with NaN
        return {k: np.nan for k in CARDIAC_FEATURE_TEMPLATE}
    
    # --- Heart rate features (5 features) ---
    rr_intervals = np.diff(peaks) / fs * 1000  # in ms
    hr = 60000 / rr_intervals  # bpm
    
    features["hr_mean"] = np.mean(hr)
    features["hr_std"] = np.std(hr)
    features["hr_min"] = np.min(hr)
    features["hr_max"] = np.max(hr)
    features["hr_range"] = features["hr_max"] - features["hr_min"]
    
    # --- HRV time-domain (8 features) ---
    features["hrv_sdnn"] = np.std(rr_intervals, ddof=1)
    features["hrv_rmssd"] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    features["hrv_sdsd"] = np.std(np.diff(rr_intervals), ddof=1)
    features["hrv_nn50"] = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)
    features["hrv_pnn50"] = features["hrv_nn50"]  # alias
    features["hrv_median_rr"] = np.median(rr_intervals)
    features["hrv_range_rr"] = np.max(rr_intervals) - np.min(rr_intervals)
    features["hrv_cv_rr"] = features["hrv_sdnn"] / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
    
    # --- HRV frequency-domain (8 features) ---
    # Interpolate RR intervals to uniform grid, then Welch PSD
    rr_times = np.cumsum(np.diff(peaks)) / fs
    rr_uniform = np.interp(
        np.arange(rr_times[0], rr_times[-1], 1/4),  # 4 Hz grid
        rr_times, rr_intervals
    )
    freqs, psd = signal.welch(rr_uniform, fs=4, nperseg=min(256, len(rr_uniform)))
    
    vlf_mask = (freqs >= 0.003) & (freqs <= 0.04)
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
    
    total_power = np.sum(psd)
    features["hrv_vlf_power"] = np.sum(psd[vlf_mask]) / total_power if total_power > 0 else 0
    features["hrv_lf_power"] = np.sum(psd[lf_mask]) / total_power if total_power > 0 else 0
    features["hrv_hf_power"] = np.sum(psd[hf_mask]) / total_power if total_power > 0 else 0
    features["hrv_lf_hf_ratio"] = (
        features["hrv_lf_power"] / features["hrv_hf_power"]
        if features["hrv_hf_power"] > 0 else 0
    )
    features["hrv_total_power"] = total_power
    features["hrv_lf_nu"] = features["hrv_lf_power"] / (features["hrv_lf_power"] + features["hrv_hf_power"]) if (features["hrv_lf_power"] + features["hrv_hf_power"]) > 0 else 0
    features["hrv_hf_nu"] = 1 - features["hrv_lf_nu"]
    features["hrv_peak_lf_freq"] = freqs[lf_mask][np.argmax(psd[lf_mask])] if np.any(lf_mask) else 0
    
    # --- SpO2 features (3 features) ---
    red_ac = np.max(red_signal) - np.min(red_signal)
    ir_ac = np.max(ir_signal) - np.min(ir_signal)
    red_dc = np.mean(red_signal)
    ir_dc = np.mean(ir_signal)
    
    r_ratio = (red_ac / red_dc) / (ir_ac / ir_dc) if ir_ac > 0 and ir_dc > 0 else 0
    features["spo2_ratio"] = r_ratio
    features["spo2_estimate"] = max(0, min(100, 110 - 25 * r_ratio))  #简化公式
    features["spo2_snr"] = 20 * np.log10(ir_ac / (np.std(ir_signal) + 1e-10))
    
    # --- PPG morphology (4 features) ---
    features["ppg_pulse_width"] = _compute_pulse_width(ppg_filtered, peaks, fs)
    features["ppg_systolic_slope"] = _compute_systolic_slope(ppg_filtered, peaks, fs)
    features["ppg_dicrotic_notch"] = _detect_dicrotic_notch(ppg_filtered, peaks, fs)
    features["ppg_pulse_amplitude_var"] = _pulse_amplitude_variability(ppg_filtered, peaks)
    
    # --- Irregularity features (2 features) ---
    features["rr_irregularity_index"] = _compute_rr_irregularity(rr_intervals)
    features["ppg_morphological_entropy"] = _signal_morphological_entropy(ppg_filtered)
    
    return features
```

#### Module 3: Temperature Features — ~8 features

```python
# src/features/temperature.py

def extract_temperature_features(
    stts22h_temp: np.ndarray,   # @1Hz, 30 samples
    lm35_temp: np.ndarray,      # @1Hz, 30 samples
    fs: int = 1
) -> dict[str, float]:
    """Extract temperature features from both sensors."""
    
    features = {}
    
    # Basic stats from each sensor
    features["temp_stts22h_mean"] = np.mean(stts22h_temp)
    features["temp_stts22h_max"] = np.max(stts22h_temp)
    features["temp_stts22h_min"] = np.min(stts22h_temp)
    features["temp_stts22h_slope"] = np.polyfit(range(len(stts22h_temp)), stts22h_temp, 1)[0] if len(stts22h_temp) > 1 else 0
    
    features["temp_lm35_mean"] = np.mean(lm35_temp)
    features["temp_lm35_max"] = np.max(lm35_temp)
    
    # Cross-sensor consistency
    features["temp_sensor_diff"] = np.mean(stts22h_temp) - np.mean(lm35_temp)
    features["temp_sensor_corr"] = np.corrcoef(stts22h_temp[:len(lm35_temp)], lm35_temp[:len(stts22h_temp)])[0, 1] if min(len(stts22h_temp), len(lm35_temp)) > 2 else 0
    
    return features
```

#### Module 4: Cross-Sensor Features — ~8 features

```python
# src/features/cross_sensor.py

def extract_cross_sensor_features(
    motion_features: dict,
    cardiac_features: dict,
    temperature_features: dict
) -> dict[str, float]:
    """Features that combine multiple sensor modalities."""
    
    features = {}
    
    # --- HR-motion coupling ---
    features["hr_motion_correlation"] = _compute_hr_motion_coupling(
        cardiac_features.get("hr_mean", 0),
        motion_features.get("acc_mag_mean", 0)
    )
    
    # --- Activity-adjusted HR ---
    acc_activity = motion_features.get("acc_energy_low", 0)
    hr = cardiac_features.get("hr_mean", 0)
    features["hr_per_activity"] = hr / (acc_activity + 1e-6)
    
    # --- Sleep quality indicators ---
    features["motion_during_low_hr"] = _motion_during_rest(
        motion_features.get("acc_mag_std", 0),
        cardiac_features.get("hr_mean", 0)
    )
    
    # --- Fatigue index ---
    features["fatigue_score"] = _compute_fatigue_index(
        hr_variability=cardiac_features.get("hrv_sdnn", 0),
        motion_fatigue=motion_features.get("acc_rms", 0),
        temperature=temperature_features.get("temp_stts22h_mean", 36.5)
    )
    
    # --- Composite vital sign score ---
    features["vital_sign_composite"] = _compute_vital_composite(
        hr=cardiac_features.get("hr_mean", 72),
        spo2=cardiac_features.get("spo2_estimate", 98),
        temp=temperature_features.get("temp_stts22h_mean", 36.5)
    )
    
    return features
```

#### Module 5: Statistical Summary Features — ~10 features

```python
# src/features/statistical.py

def extract_statistical_features(
    all_signals: dict[str, np.ndarray]
) -> dict[str, float]:
    """General statistical features across all signals."""
    
    features = {}
    
    for name, signal_data in all_signals.items():
        features[f"{name}_skewness"] = scipy.stats.skew(signal_data)
        features[f"{name}_kurtosis"] = scipy.stats.kurtosis(signal_data)
        features[f"{name}_iqr"] = np.percentile(signal_data, 75) - np.percentile(signal_data, 25)
    
    return features
```

#### Module 6: Derived Quality Features — ~5 features

```python
# src/features/quality.py

def extract_quality_features(
    raw_data: dict,
    sensor_coverage: dict
) -> dict[str, float]:
    """Features indicating data quality and reliability."""
    
    features = {}
    
    features["motion_artifact_score"] = _compute_motion_artifacts(raw_data)
    features["signal_coverage_ratio"] = _compute_coverage(sensor_coverage)
    features["ppg_sqi"] = _signal_quality_index(raw_data.get("ppg", None))
    features["sensor_agreement_score"] = _cross_sensor_agreement(raw_data)
    features["window_completeness"] = _window_completeness(sensor_coverage)
    
    return features
```

### 3.4 Window Generation

```python
# src/features/windows.py

@dataclass
class WindowConfig:
    """Configuration for sliding window generation."""
    window_size_sec: float = 30.0      # 30-second windows
    stride_sec: float = 5.0            # 5-second stride (6x overlap)
    motion_fs: int = 100               # motion/PPG sample rate
    temp_fs: int = 1                   # temperature sample rate
    
    @property
    def motion_window_samples(self) -> int:
        return int(self.window_size_sec * self.motion_fs)  # 3000
    
    @property
    def temp_window_samples(self) -> int:
        return int(self.window_size_sec * self.temp_fs)  # 30
    
    @property
    def motion_stride_samples(self) -> int:
        return int(self.stride_sec * self.motion_fs)  # 500
    
    @property
    def temp_stride_samples(self) -> int:
        return int(self.stride_sec * self.temp_fs)  # 5


def generate_windows(
    session_data: pd.DataFrame,
    config: WindowConfig = WindowConfig()
) -> list[dict]:
    """Generate overlapping windows from a session.
    
    Returns list of dicts, each containing:
    - window_id: unique identifier
    - session_id: parent session
    - start_idx: start index in session
    - end_idx: end index in session
    - motion_data: dict of np.arrays for motion channels
    - ppg_data: dict of np.arrays for PPG channels
    - temp_data: dict of np.arrays for temperature channels
    - quality_flags: dict of booleans
    """
    windows = []
    
    n_motion_samples = len(session_data)
    n_windows = (n_motion_samples - config.motion_window_samples) // config.motion_stride_samples + 1
    
    for i in range(n_windows):
        start = i * config.motion_stride_samples
        end = start + config.motion_window_samples
        
        window = {
            "window_id": f"{session_data['session_id'].iloc[0]}_w{i:04d}",
            "session_id": session_data["session_id"].iloc[0],
            "start_idx": start,
            "end_idx": end,
            "motion_data": {
                "ax": session_data["ax"].values[start:end],
                "ay": session_data["ay"].values[start:end],
                "az": session_data["az"].values[start:end],
                "gx": session_data["gx"].values[start:end],
                "gy": session_data["gy"].values[start:end],
                "gz": session_data["gz"].values[start:end],
            },
            "ppg_data": {
                "ir": session_data["ir"].values[start:end],
                "red": session_data["red"].values[start:end],
            },
            "temp_data": {
                "stts22h": session_data["temp_stts22h"].values[
                    start // config.motion_fs : end // config.motion_fs
                ],
                "lm35": session_data["temp_lm35"].values[
                    start // config.motion_fs : end // config.motion_fs
                ],
            },
            "quality_flags": _assess_window_quality(session_data, start, end),
        }
        windows.append(window)
    
    return windows
```

### 3.5 Label Generation

For synthetic data, labels are generated rule-based during data creation. For real data, labels come from clinical annotations.

```python
# src/data/labels.py

def label_window(
    window_features: dict,
    window_raw: dict,
    config: LabelConfig
) -> dict[str, bool]:
    """Generate labels for a window using clinical rules.
    
    These are used for:
    1. Synthetic data: ground-truth labels baked into generation
    2. Real data: initial pseudo-labels for bootstrapping
    3. Validation: sanity checks against known thresholds
    """
    
    labels = {}
    
    # Tachycardia: HR > 100 bpm sustained for > 5 seconds
    hr_mean = window_features.get("hr_mean", 72)
    labels["tachycardia"] = hr_mean > config.tachycardia_threshold  # default: 100 bpm
    
    # Irregular Heart Rhythm: HRV irregularity + RR interval variance
    rr_irregularity = window_features.get("rr_irregularity_index", 0)
    hrv_pnn50 = window_features.get("hrv_pnn50", 0)
    labels["irregular_rhythm"] = (
        rr_irregularity > config.rr_irregularity_threshold and
        hrv_pnn50 > config.pnn50_threshold
    )
    
    # Low SpO2: SpO2 < 95%
    spo2 = window_features.get("spo2_estimate", 98)
    labels["low_spo2"] = spo2 < config.spo2_threshold  # default: 95%
    
    # Fever: temperature > 37.5°C (oral) or > 38.0°C (axillary)
    temp = window_features.get("temp_stts22h_mean", 36.5)
    labels["fever"] = temp > config.fever_threshold  # default: 37.5°C
    
    # Fall Detection: impact spike + post-impact stillness
    acc_impact = window_features.get("acc_impact_peak", 0)
    acc_freefall = window_features.get("acc_freefall_duration", 0)
    labels["fall"] = (
        acc_impact > config.fall_impact_threshold and  # default: 3g
        acc_freefall > config.fall_freefall_threshold  # default: 0.3s
    )
    
    # Sleep Problems: low HRV + abnormal motion during sleep period
    # (requires time-of-day context, simplified here)
    labels["sleep_problems"] = _evaluate_sleep_quality(window_features, config)
    
    # Fatigue: elevated resting HR + reduced HRV + low activity
    labels["fatigue"] = _evaluate_fatigue(window_features, config)
    
    return labels


@dataclass
class LabelConfig:
    """Thresholds for rule-based labeling."""
    tachycardia_threshold: float = 100.0    # bpm
    rr_irregularity_threshold: float = 0.3
    pnn50_threshold: float = 0.05
    spo2_threshold: float = 95.0            # %
    fever_threshold: float = 37.5           # °C
    fall_impact_threshold: float = 3.0      # g
    fall_freefall_threshold: float = 0.3    # seconds
    sleep_motion_threshold: float = 0.5     # g std
    fatigue_hr_threshold: float = 85.0      # bpm resting
    fatigue_hrv_threshold: float = 20.0     # ms SDNN
```

### 3.6 Train/Val/Test Split Strategy

```
Split Strategy:
───────────────
1. SESSION-LEVEL split (never split within a session)
   - Each session is a continuous recording from one subject
   - Splitting within a session causes data leakage (overlapping windows)
   
2. Ratios: 70% train / 15% validation / 15% test

3. Stratification: by dominant condition in session
   - Ensure each split has representation of all 7 conditions
   - Use session-level labels (any positive window → positive session)

4. Subject-level separation (if multiple subjects):
   - All windows from subject X go to same split
   - Prevents model memorizing subject-specific patterns

5. Temporal separation:
   - Train: sessions from time period T1
   - Val: sessions from T2 (T2 > T1)
   - Test: sessions from T3 (T3 > T2)
   - Mimics real deployment (future data)
```

```python
# src/data/splits.py

def create_splits(
    sessions: list[str],
    labels: dict[str, dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> dict[str, list[str]]:
    """Create session-level train/val/test splits.
    
    Returns dict with keys "train", "val", "test" mapping to session ID lists.
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    # Get session-level labels (1 if any window is positive)
    session_labels = {
        sid: max(labels[sid].values()) for sid in sessions
    }
    
    # Stratified split preserving session grouping
    gss = GroupShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=seed)
    train_idx, temp_idx = next(gss.split(sessions, [session_labels[s] for s in sessions]))
    
    # Split temp into val + test
    temp_sessions = [sessions[i] for i in temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_ratio/(val_ratio+test_ratio), random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_sessions))
    
    return {
        "train": [sessions[i] for i in train_idx],
        "val": [temp_sessions[i] for i in val_idx],
        "test": [temp_sessions[i] for i in test_idx],
    }
```

### 3.7 Model Training

**LightGBM Configuration for RTX 2050:**

```python
# src/training/train.py

import lightgbm as lgb

# === Model A: Cardiac (Tachycardia + Irregular Rhythm) ===
CARDIAC_FEATURES = [
    # HR features
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_range",
    # HRV time-domain
    "hrv_sdnn", "hrv_rmssd", "hrv_sdsd", "hrv_nn50", "hrv_pnn50",
    "hrv_median_rr", "hrv_range_rr", "hrv_cv_rr",
    # HRV frequency-domain
    "hrv_vlf_power", "hrv_lf_power", "hrv_hf_power", "hrv_lf_hf_ratio",
    "hrv_total_power", "hrv_lf_nu", "hrv_hf_nu", "hrv_peak_lf_freq",
    # PPG morphology
    "ppg_pulse_width", "ppg_systolic_slope", "ppg_dicrotic_notch",
    "ppg_pulse_amplitude_var",
    # Irregularity
    "rr_irregularity_index", "ppg_morphological_entropy",
    # Cross-sensor
    "hr_motion_correlation", "hr_per_activity",
    # Quality
    "ppg_sqi", "signal_coverage_ratio",
]

CARDIAC_TARGETS = ["tachycardia", "irregular_rhythm"]

CARDIAC_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "device": "gpu",                    # RTX 2050 GPU acceleration
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "num_leaves": 31,                   # Conservative for 4GB VRAM
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "feature_fraction": 0.8,            # Column subsampling
    "bagging_fraction": 0.8,            # Row subsampling
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_bin": 63,                      # Recommended for GPU (see research)
    "gpu_use_dp": False,                # Single precision (faster on consumer GPU)
    "verbose": -1,
    "n_jobs": -1,                       # Use all CPU cores for data loading
    "random_state": 42,
    "early_stopping_rounds": 50,
}

# === Model B: Oxygen & Temperature (Low SpO2 + Fever) ===
O2_TEMP_FEATURES = [
    # SpO2
    "spo2_ratio", "spo2_estimate", "spo2_snr",
    # PPG signal
    "ppg_pulse_width", "ppg_systolic_slope", "ppg_pulse_amplitude_var",
    # Temperature
    "temp_stts22h_mean", "temp_stts22h_max", "temp_stts22h_min",
    "temp_stts22h_slope", "temp_lm35_mean", "temp_lm35_max",
    "temp_sensor_diff", "temp_sensor_corr",
    # Quality
    "ppg_sqi", "signal_coverage_ratio", "window_completeness",
]

O2_TEMP_TARGETS = ["low_spo2", "fever"]

O2_TEMP_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "num_leaves": 24,                   # Simpler model (fewer features)
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "gpu_use_dp": False,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}

# === Model C: Motion & Sleep (Fall + Sleep + Fatigue) ===
MOTION_FEATURES = [
    # Accelerometer
    "acc_mean_x", "acc_mean_y", "acc_mean_z",
    "acc_std_x", "acc_std_y", "acc_std_z",
    "acc_mag_mean", "acc_mag_std", "acc_mag_max", "acc_mag_min",
    "acc_mag_range", "acc_rms",
    # Acc frequency
    "acc_psd_peak_freq", "acc_psd_peak_power", "acc_psd_mean",
    "acc_energy_low", "acc_energy_mid", "acc_energy_high",
    # Gyroscope
    "gyro_mean", "gyro_std", "gyro_max", "gyro_rms",
    "gyro_psd_peak_freq", "gyro_energy", "gyro_entropy", "gyro_kurtosis",
    # Cross-axis
    "acc_corr_xy", "acc_corr_xz", "acc_corr_yz", "acc_jerk_mean",
    # Fall features
    "acc_impact_peak", "acc_freefall_duration", "acc_impact_rise_time",
    "acc_post_impact_std", "acc_tilt_angle",
    # Cross-sensor
    "motion_during_low_hr", "fatigue_score",
    # Quality
    "motion_artifact_score", "signal_coverage_ratio",
]

MOTION_TARGETS = ["fall", "sleep_problems", "fatigue"]

MOTION_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "num_leaves": 31,
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_bin": 63,
    "gpu_use_dp": False,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_rounds": 50,
}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    target_name: str,
    output_dir: Path
) -> lgb.Booster:
    """Train a single LightGBM model with GPU acceleration."""
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [
        lgb.early_stopping(params.pop("early_stopping_rounds", 50)),
        lgb.log_evaluation(period=100),
    ]
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"model_{target_name}.txt"
    model.save_model(str(model_path))
    
    return model
```

**GPU Memory Budget on RTX 2050:**
```
LightGBM GPU memory usage:
- Dataset binning: ~50-100MB for our feature matrix (~10K windows × 120 features)
- Tree building: ~200-500MB depending on num_leaves and max_depth
- Total: ~300-600MB (well within 4GB VRAM)

With max_bin=63 and num_leaves=31, we use ~200MB peak.
Training 3 models sequentially: each uses <1GB peak.
```

### 3.8 Model Evaluation

```python
# src/training/evaluate.py

from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, average_precision_score
)

def evaluate_model(
    model: lgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    target_name: str
) -> dict:
    """Comprehensive model evaluation."""
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    results = {
        "target": target_name,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "average_precision": average_precision_score(y_test, y_pred_proba),
    }
    
    # Optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    results["optimal_threshold"] = float(thresholds[optimal_idx])
    
    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    results["top_10_features"] = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1], reverse=True
    )[:10]
    
    # Per-class metrics for multi-target models
    for i, target in enumerate(CARDIAC_TARGETS if target_name == "cardiac" else 
                               O2_TEMP_TARGETS if target_name == "o2_temp" else
                               MOTION_TARGETS):
        results[f"{target}_auc"] = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
    
    return results
```

### 3.9 Model Serialization

**Dual format strategy: joblib for development, ONNX for production:**

```python
# src/training/serialize.py

import joblib
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from onnxconverter_common import FloatTensorType

def save_model_development(model: lgb.Booster, path: Path):
    """Save for local development/testing (joblib)."""
    joblib.dump(model, path / "model.joblib", compress=3)

def save_model_production(model: lgb.Booster, n_features: int, path: Path):
    """Save for HF Spaces deployment (ONNX).
    
    Benefits:
    - Smaller file size (typically 2-5x smaller than joblib)
    - Faster loading (no unpickling overhead)
    - Version-independent (no joblib/scikit-learn version issues)
    - ONNX Runtime optimized for CPU inference
    """
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_types)
    
    path.mkdir(parents=True, exist_ok=True)
    onnx_path = path / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    return onnx_path

def save_model_full_pipeline(
    models: dict[str, lgb.Booster],
    scaler_params: dict,
    feature_names: list[str],
    label_config: dict,
    output_dir: Path
):
    """Save complete model package for deployment.
    
    Directory structure:
    output_dir/
    ├── model_cardiac.onnx      (~50KB each)
    ├── model_o2_temp.onnx
    ├── model_motion.onnx
    ├── feature_names.json       (~5KB)
    ├── scaler_params.json       (~2KB)
    ├── label_config.json        (~1KB)
    └── metadata.json            (~1KB, version, timestamp, metrics)
    """
    # Save each model as ONNX
    for name, model in models.items():
        save_model_production(model, len(feature_names), output_dir / name)
    
    # Save metadata
    import json
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names))
    (output_dir / "scaler_params.json").write_text(json.dumps(scaler_params))
    (output_dir / "label_config.json").write_text(json.dumps(label_config))
    (output_dir / "metadata.json").write_text(json.dumps({
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "models": list(models.keys()),
        "n_features": len(feature_names),
        "training_samples": "TBD",
        "metrics": "TBD",
    }))
```

---

## 4. Inference Pipeline (HF CPU)

### 4.1 Model Loading

```python
# src/inference/loader.py

import onnxruntime as ort
import json
from pathlib import Path
from functools import lru_cache

class ModelLoader:
    """Lazy-loading model manager for HF Spaces.
    
    Models are loaded at module scope (not lazily) to avoid cold-start
    penalty on first request. ONNX Runtime is faster to initialize than
    joblib/pickle.
    """
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._models: dict[str, ort.InferenceSession] = {}
        self._feature_names: list[str] = []
        self._scaler_params: dict = {}
        self._label_config: dict = {}
        self._loaded = False
    
    def load_all(self):
        """Load all models and configs. Called once at startup."""
        if self._loaded:
            return
        
        # Load ONNX models (each ~50KB, total ~150KB)
        for model_name in ["cardiac", "o2_temp", "motion"]:
            model_path = self.model_dir / f"model_{model_name}.onnx"
            self._models[model_name] = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"]
            )
        
        # Load configs
        self._feature_names = json.loads(
            (self.model_dir / "feature_names.json").read_text()
        )
        self._scaler_params = json.loads(
            (self.model_dir / "scaler_params.json").read_text()
        )
        self._label_config = json.loads(
            (self.model_dir / "label_config.json").read_text()
        )
        
        self._loaded = True
    
    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Run prediction on a single model.
        
        Args:
            model_name: "cardiac", "o2_temp", or "motion"
            features: shape (1, n_features) or (n_features,)
        
        Returns:
            probabilities array
        """
        if not self._loaded:
            self.load_all()
        
        session = self._models[model_name]
        input_name = session.get_inputs()[0].name
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        result = session.run(None, {input_name: features.astype(np.float32)})
        return result[0]  # probabilities
    
    @property
    def feature_names(self) -> list[str]:
        return self._feature_names
    
    @property
    def scaler_params(self) -> dict:
        return self._scaler_params
    
    @property
    def label_config(self) -> dict:
        return self._label_config
```

### 4.2 Feature Consistency Guarantee

The single most critical requirement: **training and inference feature extraction must be bit-identical.**

```python
# src/features/consistency.py

"""
STRATEGY: Single FeatureExtractor class shared between training and inference.

The training pipeline imports and uses the SAME code as inference.
Feature order is enforced by a canonical feature_names list saved during training.

During inference:
1. Load feature_names.json (order is fixed)
2. Extract features using the exact same functions
3. Order features according to feature_names.json
4. Apply scaler using saved mean/std
5. Feed to model
"""

# Canonical feature order (saved during training)
FEATURE_ORDER_PATH = "feature_names.json"

class FeatureExtractor:
    """Shared feature extraction for both training and inference.
    
    This class is the SINGLE SOURCE OF TRUTH for feature computation.
    It is imported by both:
    - src/training/pipeline.py (training)
    - src/inference/pipeline.py (inference)
    
    The feature_names list is serialized during training and loaded
    during inference to guarantee ordering consistency.
    """
    
    def __init__(self, feature_names: list[str] = None):
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = self._get_default_feature_names()
    
    def extract(self, window_data: dict) -> np.ndarray:
        """Extract features from a window, return ordered numpy array.
        
        The output vector is ordered according to self.feature_names.
        """
        # Extract from each module
        motion_feats = extract_motion_features(
            window_data["motion_data"]["ax"],
            window_data["motion_data"]["ay"],
            window_data["motion_data"]["az"],
            window_data["motion_data"]["gx"],
            window_data["motion_data"]["gy"],
            window_data["motion_data"]["gz"],
        )
        cardiac_feats = extract_cardiac_features(
            window_data["ppg_data"]["ir"],
            window_data["ppg_data"]["red"],
        )
        temp_feats = extract_temperature_features(
            window_data["temp_data"]["stts22h"],
            window_data["temp_data"]["lm35"],
        )
        cross_feats = extract_cross_sensor_features(
            motion_feats, cardiac_feats, temp_feats
        )
        stat_feats = extract_statistical_features({
            "acc_mag": np.sqrt(
                window_data["motion_data"]["ax"]**2 + 
                window_data["motion_data"]["ay"]**2 + 
                window_data["motion_data"]["az"]**2
            ),
            "ppg_ir": window_data["ppg_data"]["ir"],
        })
        quality_feats = extract_quality_features(
            window_data, window_data.get("sensor_coverage", {})
        )
        
        # Merge all features
        all_features = {}
        all_features.update(motion_feats)
        all_features.update(cardiac_feats)
        all_features.update(temp_feats)
        all_features.update(cross_feats)
        all_features.update(stat_feats)
        all_features.update(quality_feats)
        
        # Order according to canonical feature names
        feature_vector = np.array([
            all_features.get(name, 0.0) for name in self.feature_names
        ], dtype=np.float32)
        
        return feature_vector
    
    def normalize(self, features: np.ndarray, scaler_params: dict) -> np.ndarray:
        """Apply z-score normalization using saved training parameters."""
        mean = np.array(scaler_params["mean"])
        std = np.array(scaler_params["std"])
        std[std < 1e-8] = 1.0  # avoid division by zero
        return (features - mean) / std
    
    def _get_default_feature_names(self) -> list[str]:
        """Generate default feature name list.
        
        Order must match training. This is computed once during training
        and saved to feature_names.json.
        """
        # This is defined at module level to ensure consistency
        return FEATURE_NAMES_CANONICAL  # ~120 names, defined in constants.py
```

### 4.3 Ensemble Aggregation

```python
# src/inference/ensemble.py

from dataclasses import dataclass

@dataclass
class ConditionResult:
    detected: bool
    confidence: float  # 0.0 - 1.0
    contributing_models: list[str]
    
class EnsembleAggregator:
    """Combine outputs from 3 LightGBM models + rule-based checks.
    
    Architecture:
    ┌──────────────┐
    │ Model Cardiac │──▶ tachycardia_prob, irregular_rhythm_prob
    └──────────────┘
    ┌──────────────┐
    │ Model O2Temp  │──▶ low_spo2_prob, fever_prob
    └──────────────┘
    ┌──────────────┐
    │ Model Motion  │──▶ fall_prob, sleep_prob, fatigue_prob
    └──────────────┘
    ┌──────────────┐
    │ Rule Engine   │──▶ hard overrides (safety-critical)
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │  Aggregator   │──▶ final 7 condition results
    └──────────────┘
    """
    
    # Model weights (tuned during validation)
    MODEL_WEIGHTS = {
        "cardiac": 0.5,
        "o2_temp": 0.5,
        "motion": 0.5,
        "rules": 0.3,  # Rules can override with high confidence
    }
    
    # Thresholds (per-condition, calibrated on validation set)
    DEFAULT_THRESHOLDS = {
        "tachycardia": 0.5,
        "irregular_rhythm": 0.5,
        "low_spo2": 0.5,
        "fever": 0.5,
        "fall": 0.4,          # Lower threshold for safety-critical
        "sleep_problems": 0.5,
        "fatigue": 0.5,
    }
    
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
    
    def aggregate(
        self,
        model_outputs: dict[str, np.ndarray],
        rule_outputs: dict[str, bool],
        feature_vector: np.ndarray
    ) -> dict[str, ConditionResult]:
        """Aggregate model and rule outputs into final predictions.
        
        Args:
            model_outputs: {"cardiac": probs, "o2_temp": probs, "motion": probs}
            rule_outputs: {"tachycardia": bool, ...} from rule engine
            feature_vector: raw features (for confidence calibration)
        
        Returns:
            Dict of condition_name → ConditionResult
        """
        results = {}
        
        # Map model outputs to conditions
        model_condition_map = {
            "cardiac": ["tachycardia", "irregular_rhythm"],
            "o2_temp": ["low_spo2", "fever"],
            "motion": ["fall", "sleep_problems", "fatigue"],
        }
        
        for condition in self.DEFAULT_THRESHOLDS.keys():
            # Gather model probabilities for this condition
            model_probs = []
            model_names = []
            
            for model_name, conditions in model_condition_map.items():
                if condition in conditions:
                    idx = conditions.index(condition)
                    prob = float(model_outputs[model_name][0][idx])
                    model_probs.append(prob * self.MODEL_WEIGHTS[model_name])
                    model_names.append(model_name)
            
            # Weighted average of model outputs
            model_score = sum(model_probs) / sum(self.MODEL_WEIGHTS[m] for m in model_names) if model_probs else 0
            
            # Rule-based override (safety-critical conditions)
            rule_flag = rule_outputs.get(condition, False)
            
            # Ensemble: max of model score and rule flag
            # Rules can force detection even if model disagrees
            if rule_flag:
                final_score = max(model_score, 0.9)  # Rules boost confidence
                contributing = model_names + ["rules"]
            else:
                final_score = model_score
                contributing = model_names
            
            # Threshold application
            detected = final_score >= self.thresholds[condition]
            
            results[condition] = ConditionResult(
                detected=detected,
                confidence=min(1.0, final_score),
                contributing_models=contributing,
            )
        
        return results
```

### 4.4 Rule Engine (Safety-Critical Checks)

```python
# src/inference/rules.py

class RuleEngine:
    """Rule-based checks for safety-critical conditions.
    
    These rules catch obvious cases that models might miss,
    and serve as a safety net. Rules are NOT meant to replace
    models — they supplement them.
    """
    
    def evaluate(
        self,
        features: dict[str, float],
        config: dict
    ) -> dict[str, bool]:
        """Evaluate rules against extracted features."""
        
        results = {}
        
        # Rule 1: Severe Tachycardia (HR > 150 bpm → always flag)
        hr = features.get("hr_mean", 72)
        results["tachycardia"] = hr > 150.0
        
        # Rule 2: Critical SpO2 (SpO2 < 90% → always flag)
        spo2 = features.get("spo2_estimate", 98)
        results["low_spo2"] = spo2 < 90.0
        
        # Rule 3: High Fever (temp > 39°C → always flag)
        temp = features.get("temp_stts22h_mean", 36.5)
        results["fever"] = temp > 39.0
        
        # Rule 4: Extreme Impact (fall with >5g → always flag)
        impact = features.get("acc_impact_peak", 0)
        freefall = features.get("acc_freefall_duration", 0)
        results["fall"] = impact > 5.0 and freefall > 0.2
        
        # Rule 5: Irregular rhythm with high confidence
        rr_irr = features.get("rr_irregularity_index", 0)
        pnn50 = features.get("hrv_pnn50", 0)
        results["irregular_rhythm"] = rr_irr > 0.5 and pnn50 > 0.1
        
        return results
```

### 4.5 Post-Processing

```python
# src/inference/postprocess.py

from collections import deque
from dataclasses import dataclass

@dataclass
class TemporalConfig:
    """Configuration for temporal smoothing."""
    window_count: int = 3              # N-of-M voting
    min_detections: int = 2            # need 2 out of 3 windows
    cooldown_seconds: float = 30.0     # min time between alerts
    max_alerts_per_hour: int = 10      # rate limiting

class PostProcessor:
    """Post-processing for temporal smoothing and confidence calibration.
    
    Key insight: A single 30-second window with elevated HR doesn't mean
    tachycardia. We need CONSISTENT detection across multiple windows.
    
    N-of-M voting: If N of the last M windows detect a condition,
    then we flag it. This prevents false positives from transient spikes.
    """
    
    def __init__(self, config: TemporalConfig = TemporalConfig()):
        self.config = config
        # Per-condition history buffers
        self._history: dict[str, deque] = {}
        self._last_alert_time: dict[str, float] = {}
        self._alert_count_hour: dict[str, int] = {}
    
    def process(
        self,
        raw_results: dict[str, ConditionResult],
        timestamp: float
    ) -> dict[str, ConditionResult]:
        """Apply temporal smoothing to raw predictions."""
        
        smoothed = {}
        
        for condition, result in raw_results.items():
            # Initialize history buffer
            if condition not in self._history:
                self._history[condition] = deque(maxlen=self.config.window_count)
            
            # Add current detection to history
            self._history[condition].append(result.detected)
            
            # N-of-M voting
            recent_detections = sum(self._history[condition])
            sustained_detection = recent_detections >= self.config.min_detections
            
            # Cooldown check
            last_alert = self._last_alert_time.get(condition, 0)
            cooldown_ok = (timestamp - last_alert) >= self.config.cooldown_seconds
            
            # Rate limiting
            hour_count = self._alert_count_hour.get(condition, 0)
            rate_ok = hour_count < self.config.max_alerts_per_hour
            
            # Final decision
            if sustained_detection and cooldown_ok and rate_ok:
                final_detected = True
                # Boost confidence for sustained detections
                confidence = min(1.0, result.confidence * (1 + 0.1 * recent_detections))
                self._last_alert_time[condition] = timestamp
                self._alert_count_hour[condition] = hour_count + 1
            else:
                final_detected = False
                confidence = result.confidence * 0.8  # Reduce confidence for non-sustained
            
            smoothed[condition] = ConditionResult(
                detected=final_detected,
                confidence=confidence,
                contributing_models=result.contributing_models + ["temporal_smooth"],
            )
        
        return smoothed
    
    def reset_hourly_counts(self):
        """Call once per hour to reset rate limits."""
        self._alert_count_hour.clear()
```

### 4.6 Output Schema

```json
{
    "session_id": "abc123",
    "window_timestamp": "2025-01-15T10:30:00Z",
    "processing_time_ms": 42,
    "conditions": {
        "tachycardia": {
            "detected": false,
            "confidence": 0.12,
            "details": {
                "heart_rate_bpm": 78,
                "model_score": 0.12,
                "rule_override": false
            }
        },
        "irregular_heart_rhythm": {
            "detected": true,
            "confidence": 0.87,
            "details": {
                "hrv_pnn50": 0.08,
                "rr_irregularity": 0.42,
                "model_score": 0.87,
                "rule_override": false
            }
        },
        "low_spo2": {
            "detected": false,
            "confidence": 0.03,
            "details": {
                "spo2_estimate": 97.2,
                "model_score": 0.03,
                "rule_override": false
            }
        },
        "fever": {
            "detected": false,
            "confidence": 0.05,
            "details": {
                "temperature_c": 36.8,
                "model_score": 0.05,
                "rule_override": false
            }
        },
        "fall_detected": {
            "detected": false,
            "confidence": 0.01,
            "details": {
                "impact_peak_g": 1.2,
                "freefall_duration_s": 0.0,
                "model_score": 0.01,
                "rule_override": false
            }
        },
        "sleep_problems": {
            "detected": false,
            "confidence": 0.15,
            "details": {
                "model_score": 0.15,
                "rule_override": false
            }
        },
        "fatigue": {
            "detected": false,
            "confidence": 0.22,
            "details": {
                "fatigue_score": 0.31,
                "model_score": 0.22,
                "rule_override": false
            }
        }
    },
    "data_quality": {
        "window_completeness": 0.98,
        "motion_artifact_score": 0.12,
        "ppg_signal_quality": 0.85
    },
    "meta": {
        "model_version": "1.0.0",
        "feature_count": 120,
        "models_used": ["cardiac", "o2_temp", "motion", "rules"]
    }
}
```

### 4.7 Performance Optimization

```python
# src/inference/optimizations.py

"""
Performance optimization strategies for HF CPU free tier.

TARGET: <100ms total inference time per window on 2 vCPU.
"""

# 1. Model loading: Load once at startup, keep in memory
# ONNX models are ~50KB each, total ~150KB in RAM

# 2. Feature computation: Use numpy vectorized operations
# Avoid Python loops in hot path

# 3. ONNX Runtime session options
ONNX_OPTIONS = ort.SessionOptions()
ONNX_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ONNX_OPTIONS.intra_op_num_threads = 2  # Match 2 vCPU
ONNX_OPTIONS.inter_op_num_threads = 1
ONNX_OPTIONS.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# 4. Input validation: Fast-fail on malformed input
# Reject windows with >30% missing data before feature extraction

# 5. Feature caching: Cache last N windows' features
# (Not applicable for streaming, but useful for batch re-analysis)

# 6. Batch inference: Not needed for real-time (single window at a time)

# 7. Float32: Use float32 throughout (not float64)
# Reduces memory bandwidth, faster on CPU

# 8. Pre-allocated buffers: Reuse numpy arrays across windows
class InferenceBuffer:
    """Pre-allocated buffers to avoid repeated memory allocation."""
    
    def __init__(self, n_features: int = 120):
        self.feature_buffer = np.zeros(n_features, dtype=np.float32)
        self.motion_buffer = {ch: np.zeros(3000, dtype=np.float32) 
                              for ch in ["ax", "ay", "az", "gx", "gy", "gz"]}
        self.ppg_buffer = {ch: np.zeros(3000, dtype=np.float32) 
                           for ch in ["ir", "red"]}
        self.temp_buffer = {ch: np.zeros(30, dtype=np.float32) 
                            for ch in ["stts22h", "lm35"]}
```

---

## 5. Data Pipeline

### 5.1 Synthetic Data Generation

```python
# scripts/generate_synthetic.py

"""
Synthetic data generator for training bootstrapping.

Generates realistic sensor data with known ground-truth labels.
Each session simulates 5-30 minutes of continuous monitoring.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class SessionProfile:
    """Defines the physiological state for a synthetic session."""
    base_hr: float = 72.0          # bpm
    hr_variability: float = 5.0    # ms std
    base_spo2: float = 98.0        # %
    base_temp: float = 36.5        # °C
    activity_level: float = 0.1    # 0=sedentary, 1=active
    conditions: list[str] = None   # e.g., ["tachycardia", "fever"]
    
def generate_session(profile: SessionProfile, duration_sec: int = 300) -> dict:
    """Generate a single synthetic session.
    
    Process:
    1. Generate base physiological signals
    2. Inject conditions at specific time points
    3. Add realistic noise and artifacts
    4. Generate corresponding sensor outputs
    """
    
    # Motion signal (100 Hz)
    motion_fs = 100
    n_motion = duration_sec * motion_fs
    t_motion = np.arange(n_motion) / motion_fs
    
    # Base motion: low-level noise + activity bursts
    base_motion = profile.activity_level * (
        0.5 * np.sin(2 * np.pi * 0.3 * t_motion) +  # slow sway
        0.2 * np.random.randn(n_motion) * profile.activity_level  # noise
    )
    
    # PPG signal (100 Hz) - simulated cardiac waveform
    ppg_fs = 100
    n_ppg = duration_sec * ppg_fs
    
    # Generate RR intervals with variability
    hr = profile.base_hr + profile.hr_variability * np.random.randn(n_ppg // ppg_fs)
    
    # Inject tachycardia if requested
    if "tachycardia" in (profile.conditions or []):
        tacho_start = np.random.randint(0, duration_sec // 2)
        tacho_end = tacho_start + np.random.randint(10, 30)
        hr[tacho_start:tacho_end] = np.random.uniform(110, 140, tacho_end - tacho_start)
    
    # Generate PPG waveform from HR
    ppg_signal = _synthesize_ppg(hr, ppg_fs, n_ppg)
    
    # SpO2 simulation
    spo2 = profile.base_spo2 * np.ones(n_ppg // ppg_fs)
    if "low_spo2" in (profile.conditions or []):
        spo2_start = np.random.randint(0, duration_sec // 2)
        spo2[spo2_start:spo2_start+20] = np.random.uniform(88, 94, 20)
    
    # Temperature simulation (1 Hz)
    temp_fs = 1
    n_temp = duration_sec
    temp = profile.base_temp * np.ones(n_temp)
    if "fever" in (profile.conditions or []):
        fever_start = np.random.randint(0, duration_sec // 2)
        temp[fever_start:] = np.random.uniform(37.8, 39.5, n_temp - fever_start)
    
    return {
        "session_id": f"synth_{uuid.uuid4().hex[:8]}",
        "duration_sec": duration_sec,
        "sample_rate": {"motion": motion_fs, "ppg": ppg_fs, "temp": temp_fs},
        "sensors": {
            "mpu6500": {
                "ax": base_motion + 9.81 * np.ones(n_motion),  # gravity
                "ay": base_motion * 0.1 + np.random.randn(n_motion) * 0.01,
                "az": np.random.randn(n_motion) * 0.01,
                "gx": np.random.randn(n_motion) * 0.5,
                "gy": np.random.randn(n_motion) * 0.5,
                "gz": np.random.randn(n_motion) * 0.5,
            },
            "max30102": {
                "ir": ppg_signal,
                "red": ppg_signal * 0.6 + np.random.randn(n_ppg) * 0.01,
                "hr_derived": hr,
                "spo2_derived": spo2,
            },
            "stts22h": {"temperature": temp},
            "lm35": {"temperature": temp + np.random.randn(n_temp) * 0.1},
        },
        "labels": {cond: True for cond in (profile.conditions or [])},
        "ground_truth": {
            "hr_series": hr,
            "spo2_series": spo2,
            "temp_series": temp,
        }
    }

def generate_dataset(
    n_sessions: int = 1000,
    output_dir: Path = Path("data/raw/synthetic")
):
    """Generate complete synthetic dataset."""
    
    # Distribution of conditions across sessions
    condition_profiles = [
        # Normal (40% of sessions)
        *[SessionProfile(conditions=[]) for _ in range(int(n_sessions * 0.4))],
        # Tachycardia (10%)
        *[SessionProfile(base_hr=105, conditions=["tachycardia"]) for _ in range(int(n_sessions * 0.1))],
        # Low SpO2 (10%)
        *[SessionProfile(base_spo2=92, conditions=["low_spo2"]) for _ in range(int(n_sessions * 0.1))],
        # Fever (10%)
        *[SessionProfile(base_temp=38.2, conditions=["fever"]) for _ in range(int(n_sessions * 0.1))],
        # Fall (10%)
        *[SessionProfile(conditions=["fall"]) for _ in range(int(n_sessions * 0.1))],
        # Multi-condition (10%)
        *[SessionProfile(base_hr=110, base_temp=38.0, conditions=["tachycardia", "fever"]) 
          for _ in range(int(n_sessions * 0.1))],
        # Sleep problems (5%)
        *[SessionProfile(conditions=["sleep_problems"]) for _ in range(int(n_sessions * 0.05))],
        # Fatigue (5%)
        *[SessionProfile(conditions=["fatigue"]) for _ in range(int(n_sessions * 0.05))],
    ]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, profile in enumerate(condition_profiles[:n_sessions]):
        session = generate_session(profile, duration_sec=np.random.randint(120, 600))
        session_path = output_dir / f"{session['session_id']}.json"
        
        with open(session_path, "w") as f:
            json.dump(session, f)
        
        if i % 100 == 0:
            print(f"Generated {i}/{n_sessions} sessions")
```

### 5.2 Data Versioning

```
Strategy: DVC (Data Version Control) + Git

data/
├── raw/
│   ├── synthetic/          # DVC tracked (large files)
│   │   ├── .gitignore      # *.json (tracked by DVC)
│   │   └── *.json.dvc      # DVC pointer files
│   └── real/               # Future
├── processed/              # DVC tracked
│   ├── features.parquet.dvc
│   ├── labels.parquet.dvc
│   └── ...
└── splits/                 # Git tracked (small JSON files)
    ├── train_ids.json
    ├── val_ids.json
    └── test_ids.json

# dvc.yaml defines data pipeline stages
stages:
  generate_data:
    cmd: python scripts/generate_synthetic.py --n-sessions 1000
    deps:
      - scripts/generate_synthetic.py
    outs:
      - data/raw/synthetic
  
  extract_features:
    cmd: python -m src.features.pipeline --input data/raw/synthetic --output data/processed
    deps:
      - src/features/
      - data/raw/synthetic
    outs:
      - data/processed/features.parquet
      - data/processed/labels.parquet
    params:
      - config/features.yaml:
          - window_size_sec
          - stride_sec
```

### 5.3 Feature Store (Lightweight)

For this project scale, a full feature store (Feast, Tecton) is overkill. Instead:

```
Feature Store = Parquet files + JSON config

data/processed/
├── features.parquet           # Feature matrix (rows=windows, cols=features)
├── labels.parquet             # Label matrix
├── feature_metadata.json      # Feature definitions, dtypes, ranges
└── normalization_stats.json   # Mean/std for each feature (from training set)

# Feature metadata tracks:
{
    "features": {
        "hr_mean": {"dtype": "float32", "range": [30, 200], "unit": "bpm", "module": "cardiac"},
        "acc_mag_std": {"dtype": "float32", "range": [0, 10], "unit": "g", "module": "motion"},
        ...
    },
    "created": "2025-01-15",
    "n_features": 120,
    "n_samples": 50000
}
```

---

## 6. MLOps & Workflow

### 6.1 Training → Deployment Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    MLOps Workflow                            │
│                                                             │
│  1. DATA PREPARATION                                        │
│     scripts/generate_synthetic.py                           │
│     → data/raw/synthetic/*.json                             │
│                                                             │
│  2. FEATURE EXTRACTION                                      │
│     python -m src.features.pipeline                         │
│     → data/processed/features.parquet                       │
│                                                             │
│  3. TRAINING (Local, RTX 2050)                              │
│     python -m src.training.train                            │
│     → models/v1.0.0/                                        │
│       ├── model_cardiac.onnx                                │
│       ├── model_o2_temp.onnx                                │
│       ├── model_motion.onnx                                 │
│       └── metadata.json                                     │
│                                                             │
│  4. EVALUATION                                              │
│     python -m src.training.evaluate                         │
│     → reports/v1.0.0/                                       │
│       ├── evaluation_report.json                             │
│       └── confusion_matrices.png                             │
│                                                             │
│  5. VALIDATION GATE                                         │
│     Check: AUC > 0.85 for all conditions                    │
│     Check: No condition has recall < 0.80                    │
│     Check: Model artifacts < 512KB total                     │
│                                                             │
│  6. REGISTRY (File-based)                                   │
│     models/registry.json updated                             │
│     → models/v1.0.0/ marked as "production"                  │
│                                                             │
│  7. DEPLOYMENT                                              │
│     Copy models to hf_space/models/                         │
│     Git commit + push → HF Spaces rebuilds automatically    │
│                                                             │
│  8. MONITORING (Lightweight)                                │
│     Log prediction distribution to JSON                     │
│     Compare against training distribution                   │
│     Alert if drift detected                                 │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Model Registry (File-Based)

```json
// models/registry.json
{
    "models": {
        "v1.0.0": {
            "created": "2025-01-15T10:30:00Z",
            "status": "production",
            "artifact_path": "models/v1.0.0/",
            "artifact_size_bytes": 145000,
            "metrics": {
                "cardiac": {
                    "tachycardia_auc": 0.92,
                    "irregular_rhythm_auc": 0.88
                },
                "o2_temp": {
                    "low_spo2_auc": 0.95,
                    "fever_auc": 0.97
                },
                "motion": {
                    "fall_auc": 0.94,
                    "sleep_problems_auc": 0.82,
                    "fatigue_auc": 0.79
                }
            },
            "training_config": {
                "n_sessions": 1000,
                "n_windows": 45000,
                "feature_count": 120,
                "lgbm_version": "4.3.0"
            }
        },
        "v0.9.0": {
            "created": "2025-01-10T14:00:00Z",
            "status": "archived",
            "artifact_path": "models/v0.9.0/",
            "artifact_size_bytes": 132000,
            "metrics": { "...": "..." }
        }
    },
    "current_production": "v1.0.0",
    "pending": null
}
```

### 6.3 Config Management

```yaml
# config/pipeline.yaml
# Central configuration for the entire pipeline

training:
  random_seed: 42
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  
  window:
    size_sec: 30.0
    stride_sec: 5.0
  
  lightgbm:
    max_bin: 63
    gpu_enabled: true
    early_stopping_rounds: 50

features:
  motion_fs: 100
  ppg_fs: 100
  temp_fs: 1
  n_features: 120
  
  # Feature flags for ablation studies
  enabled_modules:
    - motion
    - cardiac
    - temperature
    - cross_sensor
    - statistical
    - quality

inference:
  model_dir: "models/v1.0.0/"
  confidence_threshold: 0.5
  
  temporal_smoothing:
    window_count: 3
    min_detections: 2
    cooldown_seconds: 30.0
    max_alerts_per_hour: 10
  
  # Per-condition thresholds (overridden by model registry)
  thresholds:
    tachycardia: 0.5
    irregular_rhythm: 0.5
    low_spo2: 0.5
    fever: 0.5
    fall: 0.4
    sleep_problems: 0.5
    fatigue: 0.5

deployment:
  hf_space_repo: "health-monitor-space"
  max_artifact_size_bytes: 512000  # 512KB limit
  
  # Feature flags for gradual rollout
  features:
    enable_temporal_smoothing: true
    enable_rule_engine: true
    enable_confidence_calibration: true
```

### 6.4 Drift Monitoring (Lightweight)

```python
# src/monitoring/drift.py

"""
Lightweight drift detection for HF free tier.

Strategy: Compare runtime feature distributions against training baseline.
No heavy libraries (no evidently, no alibi-detect).
Simple statistical tests only.
"""

import json
from pathlib import Path
from collections import deque

class LightweightDriftMonitor:
    """Monitor for data and concept drift using simple statistics.
    
    Runs entirely on CPU with minimal memory overhead.
    Stores last 1000 predictions in memory.
    """
    
    def __init__(self, training_stats_path: Path, window_size: int = 1000):
        # Load training baseline statistics
        self.baseline = json.loads(training_stats_path.read_text())
        self.window_size = window_size
        
        # Rolling buffers for runtime statistics
        self._feature_buffer: dict[str, deque] = {}
        self._prediction_buffer: deque = deque(maxlen=window_size)
        
        # Alert thresholds
        self.drift_threshold = 0.1  # 10% distribution shift
    
    def record_prediction(self, features: np.ndarray, predictions: dict):
        """Record a prediction for drift monitoring."""
        
        # Buffer features (sample every 10th feature to save memory)
        for i in range(0, len(features), 10):
            fname = f"feature_{i}"
            if fname not in self._feature_buffer:
                self._feature_buffer[fname] = deque(maxlen=self.window_size)
            self._feature_buffer[fname].append(float(features[i]))
        
        # Buffer prediction distribution
        self._prediction_buffer.append(predictions)
    
    def check_drift(self) -> dict:
        """Check for drift using simple z-test on feature means.
        
        Returns dict with drift status and details.
        """
        alerts = []
        
        for fname, buffer in self._feature_buffer.items():
            if len(buffer) < 100:  # Need enough samples
                continue
            
            runtime_mean = np.mean(buffer)
            runtime_std = np.std(buffer)
            
            # Compare against training baseline
            baseline_mean = self.baseline.get(fname, {}).get("mean", 0)
            baseline_std = self.baseline.get(fname, {}).get("std", 1)
            
            if baseline_std < 1e-8:
                continue
            
            # Z-test for mean shift
            z_score = abs(runtime_mean - baseline_mean) / (baseline_std / np.sqrt(len(buffer)))
            
            if z_score > 3.0:  # p < 0.001
                alerts.append({
                    "feature": fname,
                    "type": "mean_shift",
                    "z_score": float(z_score),
                    "runtime_mean": float(runtime_mean),
                    "baseline_mean": float(baseline_mean),
                })
        
        # Check prediction distribution shift
        if len(self._prediction_buffer) >= 100:
            recent_conditions = {}
            for pred in list(self._prediction_buffer)[-100:]:
                for cond, result in pred.items():
                    if cond not in recent_conditions:
                        recent_conditions[cond] = []
                    recent_conditions[cond].append(result.get("detected", False))
            
            for cond, detections in recent_conditions.items():
                detection_rate = sum(detections) / len(detections)
                baseline_rate = self.baseline.get(f"pred_{cond}_rate", 0.1)
                
                if abs(detection_rate - baseline_rate) > 0.2:  # >20% shift
                    alerts.append({
                        "condition": cond,
                        "type": "prediction_shift",
                        "runtime_rate": float(detection_rate),
                        "baseline_rate": float(baseline_rate),
                    })
        
        return {
            "drift_detected": len(alerts) > 0,
            "n_alerts": len(alerts),
            "alerts": alerts,
            "buffer_sizes": {k: len(v) for k, v in self._feature_buffer.items()},
        }
    
    def save_runtime_stats(self, output_path: Path):
        """Save current runtime statistics for analysis."""
        stats = {}
        for fname, buffer in self._feature_buffer.items():
            stats[fname] = {
                "mean": float(np.mean(buffer)),
                "std": float(np.std(buffer)),
                "min": float(np.min(buffer)),
                "max": float(np.max(buffer)),
                "n_samples": len(buffer),
            }
        output_path.write_text(json.dumps(stats, indent=2))
```

---

## 7. Project Structure

```
health-monitor/
│
├── README.md                          # Project overview
├── PIPELINE_ARCHITECTURE.md           # This document
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata + tool config
├── .gitignore
├── .dvcignore
│
├── config/                            # Configuration files
│   ├── pipeline.yaml                  # Main pipeline config (§6.3)
│   ├── features.yaml                  # Feature extraction config
│   ├── training.yaml                  # Model training hyperparameters
│   ├── inference.yaml                 # Inference settings
│   └── thresholds.yaml                # Per-condition detection thresholds
│
├── data/                              # Data directory (DVC tracked)
│   ├── raw/
│   │   ├── synthetic/                 # Generated synthetic data
│   │   │   ├── .gitignore             # *.json (DVC tracked)
│   │   │   └── *.json.dvc
│   │   └── real/                      # Future real sensor data
│   │       └── .gitkeep
│   ├── processed/                     # Processed features
│   │   ├── features.parquet.dvc
│   │   ├── labels.parquet.dvc
│   │   ├── feature_names.json
│   │   ├── scaler_params.json
│   │   └── feature_metadata.json
│   ├── splits/                        # Data splits (Git tracked)
│   │   ├── train_ids.json
│   │   ├── val_ids.json
│   │   └── test_ids.json
│   └── dvc.yaml                       # DVC pipeline definition
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── data/                          # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── ingest.py                  # JSON data loading
│   │   ├── preprocess.py              # Cleaning, filtering, alignment
│   │   ├── labels.py                  # Label generation rules
│   │   ├── splits.py                  # Train/val/test splitting
│   │   └── schemas.py                 # JSON schemas for validation
│   │
│   ├── features/                      # Feature extraction
│   │   ├── __init__.py
│   │   ├── pipeline.py                # End-to-end feature pipeline
│   │   ├── motion.py                  # MPU6500 features (§3.3 Module 1)
│   │   ├── cardiac.py                 # MAX30102 PPG features (§3.3 Module 2)
│   │   ├── temperature.py             # Temperature features (§3.3 Module 3)
│   │   ├── cross_sensor.py            # Cross-modal features (§3.3 Module 4)
│   │   ├── statistical.py             # Statistical features (§3.3 Module 5)
│   │   ├── quality.py                 # Quality features (§3.3 Module 6)
│   │   ├── windows.py                 # Sliding window generation
│   │   ├── consistency.py             # FeatureExtractor class (§4.2)
│   │   ├── signal_processing.py       # Shared DSP utilities
│   │   └── constants.py               # FEATURE_NAMES_CANONICAL list
│   │
│   ├── training/                      # Model training
│   │   ├── __init__.py
│   │   ├── train.py                   # LightGBM training (§3.7)
│   │   ├── evaluate.py                # Model evaluation (§3.8)
│   │   ├── hyperparameter_search.py   # Optuna/grid search
│   │   ├── serialize.py               # Model saving (§3.9)
│   │   └── configs/                   # Per-model training configs
│   │       ├── cardiac.yaml
│   │       ├── o2_temp.yaml
│   │       └── motion.yaml
│   │
│   ├── inference/                     # Inference pipeline
│   │   ├── __init__.py
│   │   ├── app.py                     # HF Spaces Gradio app entry point
│   │   ├── loader.py                  # Model loading (§4.1)
│   │   ├── pipeline.py                # End-to-end inference pipeline
│   │   ├── ensemble.py                # Ensemble aggregation (§4.3)
│   │   ├── rules.py                   # Rule engine (§4.4)
│   │   ├── postprocess.py             # Temporal smoothing (§4.5)
│   │   └── optimizations.py           # CPU performance optimizations
│   │
│   ├── monitoring/                    # Drift monitoring
│   │   ├── __init__.py
│   │   ├── drift.py                   # Lightweight drift detection (§6.4)
│   │   └── logging.py                 # Prediction logging
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       ├── config.py                  # Config loading
│       ├── io.py                      # File I/O helpers
│       └── validation.py              # Input validation
│
├── models/                            # Model artifacts
│   ├── registry.json                  # Model registry (§6.2)
│   └── v1.0.0/                        # Versioned model package
│       ├── model_cardiac.onnx         # ~50KB each
│       ├── model_o2_temp.onnx
│       ├── model_motion.onnx
│       ├── feature_names.json         # ~5KB
│       ├── scaler_params.json         # ~2KB
│       ├── label_config.json          # ~1KB
│       └── metadata.json              # ~1KB
│
├── scripts/                           # Standalone scripts
│   ├── generate_synthetic.py          # Synthetic data generation
│   ├── extract_features.py            # Batch feature extraction
│   ├── train_all.py                   # Train all 3 models
│   ├── evaluate_all.py                # Evaluate all models
│   ├── export_onnx.py                 # Convert models to ONNX
│   ├── validate_inference.py          # End-to-end inference test
│   └── setup_hf_space.py             # Setup HF Spaces deployment
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_data/
│   │   ├── test_ingest.py
│   │   ├── test_preprocess.py
│   │   └── test_labels.py
│   ├── test_features/
│   │   ├── test_motion.py
│   │   ├── test_cardiac.py
│   │   ├── test_temperature.py
│   │   ├── test_cross_sensor.py
│   │   ├── test_consistency.py        # Critical: feature order test
│   │   └── test_windows.py
│   ├── test_training/
│   │   ├── test_train.py
│   │   └── test_evaluate.py
│   ├── test_inference/
│   │   ├── test_loader.py
│   │   ├── test_ensemble.py
│   │   ├── test_rules.py
│   │   ├── test_postprocess.py
│   │   └── test_pipeline.py           # End-to-end inference test
│   └── test_monitoring/
│       └── test_drift.py
│
├── reports/                           # Training evaluation reports
│   └── v1.0.0/
│       ├── evaluation_report.json
│       ├── confusion_matrices.png
│       └── feature_importance.png
│
├── hf_space/                          # Hugging Face Spaces deployment
│   ├── README.md                      # HF Spaces metadata
│   ├── app.py                         # Gradio app (entry point)
│   ├── requirements.txt               # Minimal dependencies for HF
│   ├── models/                        # Symlink or copy of production models
│   │   └── (copied from models/v1.0.0/)
│   └── Dockerfile                     # (Optional) Custom Docker if needed
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Run tests on push
│       ├── train.yml                  # Manual training trigger
│       └── deploy.yml                 # Auto-deploy to HF Spaces
│
└── docs/
    ├── architecture.md                # High-level architecture
    ├── feature_catalog.md             # Feature documentation
    ├── sensor_guide.md                # Sensor setup guide
    └── deployment.md                  # Deployment instructions
```

### Dependencies

```txt
# requirements.txt (full, for training)
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
lightgbm>=4.3.0
joblib>=1.3.0
onnxmltools>=1.12.0
onnxconverter-common>=1.13.0
onnx>=1.15.0
pyyaml>=6.0
dvc>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
```

```txt
# hf_space/requirements.txt (minimal, for inference)
numpy>=1.24.0
onnxruntime>=1.16.0
scipy>=1.10.0
gradio>=4.0.0
```

---

## 8. Key Design Decisions

### 8.1 Precompute on Sensor vs Server-Side?

**Decision: Server-side feature extraction.**

| Factor | On-Sensor | Server-Side | Winner |
|--------|-----------|-------------|--------|
| Latency | +0ms (already computed) | +20-40ms | Sensor |
| Accuracy | Limited by MCU compute | Full numpy/scipy | Server |
| Flexibility | Must reflash to change | Update server code | Server |
| Power | Higher MCU usage | No impact on battery | Server |
| Complexity | Embedded C/C++ | Python (simpler) | Server |

**Rationale:** The MCU (likely ESP32 or similar) has limited compute. Feature extraction requires FFT, PSD estimation, and statistical computations that are expensive on microcontrollers. Sending raw JSON windows (30s × 100Hz × 6 axes = 18,000 floats ≈ 72KB uncompressed) is feasible over BLE/WiFi. Server-side extraction allows rapid iteration without firmware updates.

**Exception:** Simple features like mean, std, and magnitude can be precomputed on-sensor to reduce bandwidth. This is a future optimization.

### 8.2 ONNX vs Joblib for Deployment?

**Decision: ONNX for production, joblib for development.**

| Factor | ONNX | Joblib | Winner |
|--------|------|--------|--------|
| File size | ~50KB per model | ~200KB per model | ONNX |
| Load time | ~5ms | ~50ms | ONNX |
| Inference speed | ~1ms | ~2ms | ONNX |
| Version independence | Yes | No (sklearn version coupling) | ONNX |
| Dependency | onnxruntime only | lightgbm + sklearn | ONNX |
| Debugging | Harder (graph) | Easier (Python objects) | Joblib |

**Rationale:** ONNX is the clear winner for HF Spaces deployment. The 512MB disk limit (see §9) means every KB counts. ONNX models are 2-4x smaller and load 10x faster. Joblib is retained for local development/debugging where convenience matters more.

### 8.3 Heartbeat/Keep-Alive Endpoint?

**Decision: Yes, include a `/health` endpoint.**

```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": loader._loaded,
        "uptime_seconds": time.time() - start_time,
        "model_version": "1.0.0",
        "last_prediction": last_prediction_time,
    }
```

**Rationale:** HF Spaces free tier sleeps after 48 hours of inactivity. A simple health endpoint:
1. Lets external monitoring (if added later) verify the service is alive
2. Can be called by a cron job to prevent sleep (if needed)
3. Costs almost nothing (no model inference, just return JSON)

### 8.4 Multiple Users / Concurrent Requests?

**Decision: Sequential processing with Gradio queue.**

HF Spaces free tier has:
- 2 vCPU cores
- 16GB RAM
- No GPU
- Gradio handles request queuing natively

**Strategy:**
1. Gradio's built-in queue handles concurrency (requests wait in line)
2. Set `queue()` with `max_size=10` to limit queue depth
3. Each inference takes ~50ms, so we can handle ~20 requests/second theoretically
4. In practice, feature extraction dominates (~30ms), so ~15 req/sec

```python
import gradio as gr

demo = gr.Interface(
    fn=inference_pipeline,
    inputs=gr.JSON(label="Sensor Data Window"),
    outputs=gr.JSON(label="Health Predictions"),
    title="Health Symptom Detection",
)
demo.queue(max_size=10)  # Limit concurrent queue
```

**If concurrent load exceeds capacity:** Upgrade to HF Spaces PRO ($9/month) for more resources, or add a simple Redis queue (not applicable on free tier).

### 8.5 Batch vs Stream?

**Decision: Single-window streaming inference.**

| Factor | Batch | Stream | Winner |
|--------|-------|--------|--------|
| Latency | Higher (wait for batch) | Lower (process immediately) | Stream |
| Throughput | Higher (amortized overhead) | Lower | Batch |
| Memory | Higher (hold batch) | Lower | Stream |
| Complexity | More complex | Simpler | Stream |
| HF Spaces fit | Needs batching logic | Natural fit | Stream |

**Rationale:** The use case is real-time health monitoring. Users expect near-instant feedback. Each window is independent, so streaming is natural. The 50ms inference time means we can process each window as it arrives without noticeable delay.

**Exception:** If processing historical data (e.g., "analyze my last hour"), batch mode is useful. The pipeline supports both via a flag:

```python
def process(data, mode="stream"):
    if mode == "stream":
        return process_single_window(data)
    elif mode == "batch":
        return process_batch(data)  # parallel feature extraction
```

---

## 9. Risk & Mitigation

### 9.1 RTX 2050 (4GB VRAM) Limits

**Risk:** GPU out-of-memory during LightGBM training.

**Mitigation:**
| Strategy | Implementation | Impact |
|----------|---------------|--------|
| `max_bin=63` | Reduces histogram bins from 255 to 63 | -60% GPU memory, <1% accuracy loss |
| `num_leaves=31` | Conservative tree complexity | -40% memory vs 127 leaves |
| `gpu_use_dp=false` | Single precision | -50% memory vs double precision |
| Sequential model training | Train one model at a time | Peak usage <1GB |
| Feature subsampling | `feature_fraction=0.8` | -20% per-iteration memory |
| Row subsampling | `bagging_fraction=0.8` | -20% per-iteration memory |

**Actual VRAM usage (estimated):**
```
Dataset binning:     ~80MB  (50K windows × 120 features × 63 bins)
Tree building:       ~150MB (31 leaves × 500 trees × batch)
Gradient stats:      ~50MB
Total peak:          ~280MB (well within 4GB)
```

**If VRAM still insufficient:** Fall back to CPU training with `device="cpu"` and `num_threads=8`. LightGBM CPU training is still fast for our dataset size (~5 minutes for 50K samples).

### 9.2 HF Spaces Free Tier Constraints

**Risk:** Cold starts, memory limits, disk limits.

| Constraint | Value | Mitigation |
|-----------|-------|------------|
| Cold start | 30-60 seconds | Load models at module scope (not lazily) |
| RAM | 16GB | Total model footprint <1MB; feature extraction uses ~50MB working memory |
| Disk | 50GB ephemeral | Models ~150KB total; code ~5MB; no large dependencies |
| Artifact limit | 512MB (soft) | Our models are 150KB total — 0.03% of limit |
| Sleep after | 48 hours idle | Health endpoint + optional external keep-alive |
| Concurrent | 8 CPU units | Single Space, sequential processing |

**Cold start optimization:**
```python
# app.py - Module scope loading
import onnxruntime as ort
from src.inference.loader import ModelLoader

# Load models ONCE at import time (not on first request)
LOADER = ModelLoader(Path("models/v1.0.0"))
LOADER.load_all()  # Takes ~100ms total for 3 ONNX models

# Gradio app uses pre-loaded models
def predict(sensor_data):
    return PIPELINE.run(sensor_data, LOADER)
```

### 9.3 Model Update Strategy

**Risk:** Updating models without downtime.

**Strategy: Blue-Green deployment via Git:**

```
1. Train new model locally → models/v1.1.0/
2. Test thoroughly on local machine
3. Update hf_space/models/ symlink to v1.1.0
4. Git commit + push
5. HF Spaces rebuilds automatically (takes ~60 seconds)
6. During rebuild, old version serves requests
7. After rebuild, new version is live
```

**Rollback:** Revert the Git commit → HF Spaces rebuilds with old version.

**Zero-downtime alternative (if needed):** Use Gradio's model hot-reload:
```python
# In app.py
MODELS = {"v1.0.0": loader_v1, "v1.1.0": loader_v1_1}

def predict(data, model_version="v1.1.0"):
    return MODELS[model_version].predict(data)
```

---

## Appendix A: Feature Catalog

### Complete Feature List (~120 features)

| # | Feature Name | Module | Type | Range | Description |
|---|-------------|--------|------|-------|-------------|
| 1 | acc_mean_x | Motion | float32 | [-16, 16] g | Mean acceleration X |
| 2 | acc_mean_y | Motion | float32 | [-16, 16] g | Mean acceleration Y |
| 3 | acc_mean_z | Motion | float32 | [-16, 16] g | Mean acceleration Z |
| 4 | acc_std_x | Motion | float32 | [0, 16] g | Std acceleration X |
| 5 | acc_std_y | Motion | float32 | [0, 16] g | Std acceleration Y |
| 6 | acc_std_z | Motion | float32 | [0, 16] g | Std acceleration Z |
| 7 | acc_mag_mean | Motion | float32 | [0, 16] g | Mean acceleration magnitude |
| 8 | acc_mag_std | Motion | float32 | [0, 16] g | Std acceleration magnitude |
| 9 | acc_mag_max | Motion | float32 | [0, 16] g | Max acceleration magnitude |
| 10 | acc_mag_min | Motion | float32 | [0, 16] g | Min acceleration magnitude |
| 11 | acc_mag_range | Motion | float32 | [0, 16] g | Max-min acceleration |
| 12 | acc_rms | Motion | float32 | [0, 16] g | RMS acceleration |
| 13 | acc_psd_peak_freq | Motion | float32 | [0, 50] Hz | Dominant frequency |
| 14 | acc_psd_peak_power | Motion | float32 | [0, ∞) | Peak PSD power |
| 15 | acc_psd_mean | Motion | float32 | [0, ∞) | Mean PSD |
| 16 | acc_energy_low | Motion | float32 | [0, ∞) | Energy 0.5-5Hz |
| 17 | acc_energy_mid | Motion | float32 | [0, ∞) | Energy 5-15Hz |
| 18 | acc_energy_high | Motion | float32 | [0, ∞) | Energy 15-50Hz |
| 19 | gyro_mean | Motion | float32 | [0, 2000] °/s | Mean gyroscope |
| 20 | gyro_std | Motion | float32 | [0, 2000] °/s | Std gyroscope |
| 21 | gyro_max | Motion | float32 | [0, 2000] °/s | Max gyroscope |
| 22 | gyro_rms | Motion | float32 | [0, 2000] °/s | RMS gyroscope |
| 23 | gyro_psd_peak_freq | Motion | float32 | [0, 50] Hz | Gyro dominant frequency |
| 24 | gyro_energy | Motion | float32 | [0, ∞) | Gyro total energy |
| 25 | gyro_entropy | Motion | float32 | [0, ∞) | Spectral entropy |
| 26 | gyro_kurtosis | Motion | float32 | [-∞, ∞) | Distribution shape |
| 27 | acc_corr_xy | Motion | float32 | [-1, 1] | XY correlation |
| 28 | acc_corr_xz | Motion | float32 | [-1, 1] | XZ correlation |
| 29 | acc_corr_yz | Motion | float32 | [-1, 1] | YZ correlation |
| 30 | acc_jerk_mean | Motion | float32 | [-∞, ∞) | Mean acceleration change |
| 31 | acc_impact_peak | Motion | float32 | [0, 16] g | Peak impact force |
| 32 | acc_freefall_duration | Motion | float32 | [0, 30] s | Time below 0.5g |
| 33 | acc_impact_rise_time | Motion | float32 | [0, 1] s | Rise time to peak |
| 34 | acc_post_impact_std | Motion | float32 | [0, 16] g | Post-impact variability |
| 35 | acc_tilt_angle | Motion | float32 | [0, 180] ° | Tilt from vertical |
| 36 | hr_mean | Cardiac | float32 | [30, 200] bpm | Mean heart rate |
| 37 | hr_std | Cardiac | float32 | [0, 50] bpm | HR variability |
| 38 | hr_min | Cardiac | float32 | [30, 200] bpm | Min HR |
| 39 | hr_max | Cardiac | float32 | [30, 200] bpm | Max HR |
| 40 | hr_range | Cardiac | float32 | [0, 170] bpm | HR range |
| 41 | hrv_sdnn | Cardiac | float32 | [0, 200] ms | SDNN |
| 42 | hrv_rmssd | Cardiac | float32 | [0, 200] ms | RMSSD |
| 43 | hrv_sdsd | Cardiac | float32 | [0, 200] ms | SDSD |
| 44 | hrv_nn50 | Cardiac | float32 | [0, 1] | NN50 ratio |
| 45 | hrv_pnn50 | Cardiac | float32 | [0, 1] | pNN50 |
| 46 | hrv_median_rr | Cardiac | float32 | [300, 2000] ms | Median RR interval |
| 47 | hrv_range_rr | Cardiac | float32 | [0, 1700] ms | RR range |
| 48 | hrv_cv_rr | Cardiac | float32 | [0, 1] | CV of RR |
| 49 | hrv_vlf_power | Cardiac | float32 | [0, 1] | VLF power (normalized) |
| 50 | hrv_lf_power | Cardiac | float32 | [0, 1] | LF power (normalized) |
| 51 | hrv_hf_power | Cardiac | float32 | [0, 1] | HF power (normalized) |
| 52 | hrv_lf_hf_ratio | Cardiac | float32 | [0, ∞) | LF/HF ratio |
| 53 | hrv_total_power | Cardiac | float32 | [0, ∞) | Total spectral power |
| 54 | hrv_lf_nu | Cardiac | float32 | [0, 1] | LF normalized units |
| 55 | hrv_hf_nu | Cardiac | float32 | [0, 1] | HF normalized units |
| 56 | hrv_peak_lf_freq | Cardiac | float32 | [0, 0.15] Hz | Peak LF frequency |
| 57 | spo2_ratio | Cardiac | float32 | [0, ∞) | R-value |
| 58 | spo2_estimate | Cardiac | float32 | [70, 100] % | Estimated SpO2 |
| 59 | spo2_snr | Cardiac | float32 | [-∞, ∞) dB | PPG signal SNR |
| 60 | ppg_pulse_width | Cardiac | float32 | [0, 1] s | Pulse width |
| 61 | ppg_systolic_slope | Cardiac | float32 | [0, ∞) | Systolic upstroke slope |
| 62 | ppg_dicrotic_notch | Cardiac | float32 | [0, 1] | Dicrotic notch presence |
| 63 | ppg_pulse_amplitude_var | Cardiac | float32 | [0, ∞) | Pulse amplitude CV |
| 64 | rr_irregularity_index | Cardiac | float32 | [0, 1] | RR irregularity |
| 65 | ppg_morphological_entropy | Cardiac | float32 | [0, ∞) | Signal entropy |
| 66 | temp_stts22h_mean | Temp | float32 | [20, 45] °C | Mean skin temp |
| 67 | temp_stts22h_max | Temp | float32 | [20, 45] °C | Max skin temp |
| 68 | temp_stts22h_min | Temp | float32 | [20, 45] °C | Min skin temp |
| 69 | temp_stts22h_slope | Temp | float32 | [-∞, ∞) °C/s | Temp trend |
| 70 | temp_lm35_mean | Temp | float32 | [20, 45] °C | Mean LM35 temp |
| 71 | temp_lm35_max | Temp | float32 | [20, 45] °C | Max LM35 temp |
| 72 | temp_sensor_diff | Temp | float32 | [-∞, ∞) °C | Sensor difference |
| 73 | temp_sensor_corr | Temp | float32 | [-1, 1] | Sensor correlation |
| 74 | hr_motion_correlation | Cross | float32 | [-1, 1] | HR-motion coupling |
| 75 | hr_per_activity | Cross | float32 | [0, ∞) | HR normalized by activity |
| 76 | motion_during_low_hr | Cross | float32 | [0, ∞) | Motion during rest |
| 77 | fatigue_score | Cross | float32 | [0, 1] | Composite fatigue index |
| 78 | vital_sign_composite | Cross | float32 | [0, 1] | Overall vital score |
| 79 | acc_mag_skewness | Stat | float32 | [-∞, ∞) | Distribution skew |
| 80 | acc_mag_kurtosis | Stat | float32 | [-∞, ∞) | Distribution kurtosis |
| 81 | acc_mag_iqr | Stat | float32 | [0, ∞) | Interquartile range |
| 82 | ppg_ir_skewness | Stat | float32 | [-∞, ∞) | PPG skew |
| 83 | ppg_ir_kurtosis | Stat | float32 | [-∞, ∞) | PPG kurtosis |
| 84 | ppg_ir_iqr | Stat | float32 | [0, ∞) | PPG IQR |
| 85 | motion_artifact_score | Quality | float32 | [0, 1] | Motion artifact level |
| 86 | signal_coverage_ratio | Quality | float32 | [0, 1] | Data completeness |
| 87 | ppg_sqi | Quality | float32 | [0, 1] | PPG signal quality |
| 88 | sensor_agreement_score | Quality | float32 | [0, 1] | Cross-sensor consistency |
| 89 | window_completeness | Quality | float32 | [0, 1] | Window data coverage |

**Note:** Features 90-120 are additional statistical and derived features from the statistical module. The exact count depends on the number of raw signals included in `extract_statistical_features`.

---

## Appendix B: Sensor Specifications

### MPU6500 (6-Axis IMU)

| Parameter | Value |
|-----------|-------|
| Accelerometer Range | ±2/4/8/16g (configurable) |
| Gyroscope Range | ±250/500/1000/2000 °/s |
| Sample Rate | Up to 8kHz (we use 100Hz) |
| Noise (Accel) | 300μg/√Hz |
| Noise (Gyro) | 0.01°/s/√Hz |
| Interface | I2C (400kHz) or SPI (1MHz) |
| FIFO | 512 bytes |

### MAX30102 (Pulse Oximeter + HR)

| Parameter | Value |
|-----------|-------|
| LED wavelengths | Red (660nm) + IR (940nm) |
| ADC resolution | 18-bit |
| Sample rate | Up to 3200Hz (we use 100Hz) |
| Dynamic range | >60dB |
| Interface | I2C |
| Supply voltage | 1.8V (logic) + 5.0V (LEDs) |

### STTS22H (Temperature)

| Parameter | Value |
|-----------|-------|
| Range | -40°C to +125°C |
| Accuracy | ±0.5°C (-10°C to +60°C) |
| Resolution | 0.01°C |
| Sample rate | 1Hz (we use 1Hz) |
| Interface | I2C |

### LM35 (Temperature)

| Parameter | Value |
|-----------|-------|
| Range | -55°C to +150°C |
| Accuracy | ±0.5°C at 25°C |
| Output | 10mV/°C (analog) |
| Interface | ADC (we use 12-bit) |

---

## Appendix C: Critical Implementation Notes

### Feature Order Consistency Checklist

The #1 source of bugs in ML pipelines is feature order mismatch between training and inference. Here is the checklist:

- [ ] `FEATURE_NAMES_CANONICAL` list defined in `src/features/constants.py`
- [ ] `FeatureExtractor.extract()` returns features in canonical order
- [ ] Training saves `feature_names.json` with exact order
- [ ] Inference loads `feature_names.json` and validates order
- [ ] Test: `test_feature_order.py` verifies training and inference produce identical feature vectors for same input
- [ ] Test: `test_feature_consistency.py` loads a training sample, runs it through inference pipeline, compares output

### ONNX Conversion Checklist

- [ ] All 3 models converted via `onnxmltools`
- [ ] Input type: `FloatTensorType([None, 120])`
- [ ] Output type: probabilities (not class labels)
- [ ] Test: ONNX predictions match LightGBM predictions within 1e-5 tolerance
- [ ] File sizes verified < 100KB each

### HF Spaces Deployment Checklist

- [ ] `hf_space/requirements.txt` has only: numpy, onnxruntime, scipy, gradio
- [ ] No PyTorch, no TensorFlow (saves ~2GB)
- [ ] Models loaded at module scope, not lazily
- [ ] Gradio queue enabled with max_size
- [ ] `/health` endpoint implemented
- [ ] Total artifact size < 512KB
- [ ] Test: `validate_inference.py` runs end-to-end on HF Spaces

---

*Document version: 1.0.0*
*Last updated: 2025-01-15*
*Author: Pipeline Architecture Research*
